import numpy

import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_

from torch_ac.utils import DictList, ParallelEnv

class FEAlgo:
    def __init__(self,
        envs,
        acmodel,
        device=None,
        num_frames_per_proc=None,
        explorer_rl_algo='ppo',
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.1,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        horizon=None,
        preprocess_obss=None,
    ):
        
        # store parameters
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc or 128
        self.explorer_rl_algo = explorer_rl_algo
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon or self.env.envs[0].max_steps
        self.preprocess_obss = preprocess_obss
        
        # configure model
        self.acmodel.to(self.device)
        self.acmodel.train()
        
        # store helper values
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * shape[0]

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # initialize log values
        self.log_episode_return = torch.zeros(
            self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(
            self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # build the optimizer
        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(), lr, eps=adam_eps)

        self.batch_num = 0
    
    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            # convert observations to tensors
            preprocessed_obs = self.preprocess_obss(
                self.obs, device=self.device)
            
            # model forward
            with torch.no_grad():
                f_dist, f_value, e_dist, e_value = self.acmodel(
                    preprocessed_obs)
            
            # determine when to use the follower
            use_follower = (
                preprocessed_obs.step >= preprocessed_obs.switching_time)
            
            # sample an action
            f_action = f_dist.sample()
            e_action = e_dist.sample()
            action = f_action * use_follower + e_action * ~use_follower
            
            # step
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            # select follower/explorer
            value = f_value * use_follower + e_value * ~use_follower
            log_prob = (
                f_dist.log_prob(action) * use_follower +
                e_dist.log_prob(action) * ~use_follower
            )
            
            # record values for training
            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(
                done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            reward_tensor = torch.tensor(
                reward, device=self.device, dtype=torch.float)
            self.rewards[i] = reward_tensor
            self.log_probs[i] = log_prob

            # update log values
            self.log_episode_return += reward_tensor
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)
            
            for i, d in enumerate(done):
                if d:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(
                        self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[i].item())
            
            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
        
        # add advantages and return to experiences
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            #_, next_value = self.acmodel(preprocessed_obs)
            f_dist, f_value, e_dist, e_value = self.acmodel(preprocessed_obs)
            use_follower = (
                preprocessed_obs.step >= preprocessed_obs.switching_time)
            next_value = f_value * use_follower + e_value * ~use_follower
        
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = (
                self.masks[i+1]
                if i < self.num_frames_per_proc - 1
                else self.mask
            )
            next_value = (
                self.values[i+1]
                if i < self.num_frames_per_proc - 1
                else next_value
            )
            next_advantage = (
                self.advantages[i+1]
                if i < self.num_frames_per_proc - 1
                else 0
            )

            delta = (
                self.rewards[i] +
                self.discount * next_value * next_mask -
                self.values[i]
            )
            self.advantages[i] = (
                delta +
                self.discount * self.gae_lambda * next_advantage * next_mask
            )
        
        # concatenate the per-process experience
        exps = DictList()
        exps.obs = [
            self.obss[i][j]
            for j in range(self.num_procs)
            for i in range(self.num_frames_per_proc)
        ]

        exps.action = self.actions.transpose(0,1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # log
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
    
    def update_parameters(self, exps):
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []
        for _ in range(self.epochs):

            for inds in self._get_batches_starting_indexes():

                sb = exps[inds]

                # forward
                f_dist, f_value, e_dist, e_value = self.acmodel(sb.obs)
                use_follower = sb.obs.step >= sb.obs.switching_time
                
                # compute follower policy_loss
                follower_policy_loss = cross_entropy(
                    f_dist.logits, sb.obs.expert)
                
                if self.explorer_rl_algo == 'ppo':
                    # compute explorer policy loss
                    ratio = torch.exp(e_dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(
                        ratio, 1. - self.clip_eps, 1. + self.clip_eps
                    ) * sb.advantage
                    explorer_policy_loss = -torch.min(surr1, surr2)
                    
                    # compute explorer value loss
                    e_value_clipped = sb.value + torch.clamp(
                        e_value - sb.value,
                        -self.clip_eps,
                        self.clip_eps
                    )
                    e_surr1 = (e_value - sb.returnn).pow(2)
                    e_surr2 = (e_value_clipped - sb.returnn).pow(2)
                    explorer_value_loss = torch.max(e_surr1, e_surr2).mean()
                    
                    # combine explorer policy and value and entropy losses
                    explorer_entropy = e_dist.entropy()
                    explorer_loss = (
                        explorer_policy_loss -
                        self.entropy_coef * explorer_entropy +
                        self.value_loss_coef * explorer_value_loss
                    )
                    
                    # compute follower value loss
                    f_value_clipped = sb.value + torch.clamp(
                        f_value - sb.value,
                        -self.clip_eps,
                        self.clip_eps,
                    )
                    f_surr1 = (f_value - sb.returnn).pow(2)
                    f_surr2 = (f_value_clipped - sb.returnn).pow(2)
                    follower_value_loss = torch.max(f_surr1, f_surr2).mean()
                
                else:
                    raise ValueError(
                        'Unknown explorer RL algorithm: %s'%
                        self.explorer_rl_algo
                    )
                
                # combine follower policy and value losses
                follower_loss = (
                    follower_policy_loss +
                    self.value_loss_coef * follower_value_loss
                )
                
                # compute final loss
                loss = (
                    follower_loss * use_follower +
                    explorer_loss * ~use_follower
                ).mean()
                
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.acmodel.parameters()
                    if p.grad is not None
                ) ** 0.5
                clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # log
                follower_entropy = f_dist.entropy()
                entropy = (
                    follower_entropy * use_follower +
                    explorer_entropy * ~use_follower
                ).mean()
                
                value = (
                    f_value * use_follower +
                    e_value * ~use_follower
                ).mean()
                
                policy_loss = (
                    follower_policy_loss * use_follower +
                    explorer_policy_loss * ~use_follower
                ).mean()
                
                value_loss = (
                    follower_value_loss * use_follower +
                    explorer_value_loss * ~use_follower
                ).mean()
                
                log_entropies.append(entropy.item())
                log_values.append(value.item())
                log_policy_losses.append(policy_loss.item())
                log_value_losses.append(value_loss.item())
                log_grad_norms.append(grad_norm)

        logs = {
            'entropy' : numpy.mean(log_entropies),
            'value' : numpy.mean(log_values),
            'policy_loss' : numpy.mean(log_policy_losses),
            'value_loss' : numpy.mean(log_value_losses),
            'grad_norm' : numpy.mean(log_grad_norms),
        }

        return logs
    
    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a
        step of `self.recurrence`, shifted by `self.recurrence//2` one time
        in two for having more diverse batches. Then, the indexes are splited
        into the different batches.
        (NO RECURRENCE, IGNORE)
        
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, 1)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        #if self.batch_num % 2 == 1:
        #    indexes = indexes[
        #        (indexes + self.recurrence) % self.num_frames_per_proc != 0]
        #    indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size # // self.recurrence
        batches_starting_indexes = [
            indexes[i:i+num_indexes]
            for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes
