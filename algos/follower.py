import numpy

import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_

from torch_ac.utils import DictList, ParallelEnv

class FollowerAlgo:
    def __init__(self,
        envs,
        acmodel,
        explorer_model,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        optimizer=None,
    ):
        
        # store parameters
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.explorer_model = explorer_model
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc or 1024
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocess_obss = preprocess_obss
        
        # configure model
        self.acmodel.to(self.device)
        self.explorer_model.to(self.device)
        self.acmodel.train()
        self.explorer_model.train()
        
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
        self.use_follower = torch.zeros(*shape, device=self.device)
        
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
        self.optimizer = optimizer or torch.optim.Adam(
            self.acmodel.parameters(), lr, eps=adam_eps)
        
        self.batch_num = 0
    
    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(
                self.obs, device=self.device)
            with torch.no_grad():
                follower_dist, follower_value = self.acmodel(preprocessed_obs)
                explorer_dist, explorer_value = self.explorer_model(
                    preprocessed_obs)
            
            # decide when to use the follower
            use_follower = (
                preprocessed_obs.step >= preprocessed_obs.switching_time)
            use_explorer = ~use_follower
            
            # compute the action
            follower_action = follower_dist.sample()
            explorer_action = explorer_dist.sample()
            action = (
                follower_action * use_follower +
                explorer_action * use_explorer
            )
            
            # compute the value
            #value = (
            #    follower_value * use_follower +
            #    explorer_value * use_explorer
            #)
            
            # compute the log_prob
            follower_log_prob = follower_dist.log_prob(action)
            explorer_log_prob = explorer_dist.log_prob(action)
            log_prob = (
                follower_log_prob * use_follower +
                explorer_log_prob * use_explorer
            )
            
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(
                done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = follower_value #value
            reward_tensor = torch.tensor(
                reward, device=self.device, dtype=torch.float)
            self.rewards[i] = reward_tensor
            self.log_probs[i] = log_prob
            self.use_follower[i] = use_follower
            
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
            _, follower_next_value = self.acmodel(preprocessed_obs)
            _, explorer_next_value = self.explorer_model(preprocessed_obs)
            use_follower = (
                preprocessed_obs.step >= preprocessed_obs.switching_time)
            use_explorer = ~use_follower
            next_value = (
                follower_next_value * use_follower +
                explorer_next_value * use_explorer
            )
        
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
        exps.use_follower = self.use_follower.transpose(0, 1).reshape(-1)
        
        # preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        
        # log
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "follower_return_per_episode": self.log_return[-keep:],
            "follower_reshaped_return_per_episode":
                self.log_reshaped_return[-keep:],
            "follower_num_frames_per_episode": self.log_num_frames[-keep:],
            "follower_num_frames": self.num_frames
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
                dist, value = self.acmodel(sb.obs)
                
                # compute policy loss
                policy_loss = cross_entropy(dist.logits, sb.obs.expert)
                
                # compute value loss
                value_clipped = sb.value + torch.clamp(
                    value - sb.value, -self.clip_eps, self.clip_eps)
                surr1 = (value - sb.returnn).pow(2)
                surr2 = (value_clipped - sb.returnn).pow(2)
                max_surr = torch.max(surr1, surr2)
                filtered_max_surr = torch.sum(max_surr * sb.use_follower)
                divisor = torch.sum(sb.use_follower) + 1e-6
                value_loss = filtered_max_surr / divisor
                
                #breakpoint()
                
                # combine loss
                loss = policy_loss + self.value_loss_coef * value_loss
                
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
                
                entropy = dist.entropy().mean()
                log_entropies.append(entropy.item())
                log_values.append(value.mean().item())
                log_policy_losses.append(loss.item())
                log_value_losses.append(value_loss.item())
                log_grad_norms.append(grad_norm)
        
        logs = {
            'follower_entropy' : numpy.mean(log_entropies),
            'follower_value' : numpy.mean(log_values),
            'follower_policy_loss' : numpy.mean(log_policy_losses),
            'follower_value_loss' : numpy.mean(log_value_losses),
            'follower_grad_norm' : numpy.mean(log_grad_norms),
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
