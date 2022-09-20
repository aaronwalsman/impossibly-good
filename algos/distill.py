import numpy

import torch
from torch.nn.functional import cross_entropy

from torch_ac.utils import DictList, ParallelEnv

class Distill:
    def __init__(self,
        envs,
        model,
        reward_maximizer='ppo',
        l_term='zero', # zero | cross_entropy | reversed_cross_entropy
        r_term='zero', # zero | log_pi | future_log_pi | future_cross_entropy | value_shaping
        plus_R=False,
        on_policy=True,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        expert_loss_coef=1.0,
        entropy_loss_coef=0.01,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        #reshape_reward=None,
    ):
        
        if num_frames_per_proc is None:
            if reward_maximizer == 'ppo':
                num_frames_per_proc = 128
            else:
                num_frames_per_proc = 8
        
        # store parameters
        self.env = ParallelEnv(envs)
        self.model = model
        self.reward_maximizer = reward_maximizer
        self.l_term = l_term
        self.r_term = r_term
        self.plus_R = plus_R
        self.on_policy = on_policy
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.policy_loss_coef = policy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.expert_loss_coef = expert_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.adam_eps=adam_eps
        self.clip_eps=clip_eps
        self.epochs=epochs
        self.batch_size=batch_size
        self.preprocess_obss = preprocess_obss #or default_preprocess_obss
        #self.reshape_reward = reshape_reward
        
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        
        # error checking
        assert self.model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0
        assert self.batch_size % self.recurrence == 0

        # configure the model
        self.model.to(self.device)
        self.model.train()
        
        # initialize experience
        shape = (self.num_frames_per_proc, self.num_procs)
        self.obs = self.env.reset()
        self.obss = [None] * shape[0]
        if self.model.recurrent:
            self.memory = torch.zeros(
                shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(
                *shape, self.model.memory_size, device=self.device)
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
        
        # initialize optimzier
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
    
    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            
            # convert observations to tensors
            preprocessed_obs = self.preprocess_obss(
                self.obs, device=self.device)
            
            # forward pass
            with torch.no_grad():
                if self.model.recurrent:
                    dist, value, memory = self.model(
                        preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.model(preprocessed_obs)
            
            if self.on_policy:
                action = dist.sample()
            else:
                action = preprocessed_obs.expert
            
            # step
            pre_obs = self.obs
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            # compute reward surrogate
            surrogate_reward = numpy.zeros((len(reward),))
            if self.plus_R:
                surrogate_reward += reward
            if self.r_term == 'log_pi':
                raise Exception('TODO')
            elif self.r_term == 'future_log_pi':
                raise Exception('TODO')
            elif self.r_term == 'future_cross_entropy':
                raise Exception('TODO')
            elif self.r_term == 'value shaping':
                raise Exception('TODO')
            
            # update experiences
            self.obss[i] = self.obs
            self.obs = obs
            if self.model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(
                done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = torch.tensor(surrogate_reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # logging
            self.log_episode_return += torch.tensor(
                reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)
            
            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(
                        self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[i].item())
            
            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
        
        # add advantage and return to experiences
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.model.recurrent:
                _, next_value, _ = self.model(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.model(preprocessed_obs)

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
        
        # concatenate the experience of each process
        # below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.model.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(
                -1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

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
        
        for _ in range(self.epochs):
            
            # initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                
                # initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # initialize memory
                if self.model.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    
                    # create a sub-batch of experience
                    sb = exps[inds + i]

                    # model forward pass
                    if self.model.recurrent:
                        dist, value, memory = self.model(
                            sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.model(sb.obs)
                    
                    # compute reward-based losses
                    if self.reward_maximizer == 'vpg':
                        policy_loss = self.vpg_loss(dist, value, sb)
                        value_loss = torch.zeros(1, device=self.device)
                    elif self.reward_maximizer == 'a2c':
                        policy_loss, value_loss = self.a2c_losses(
                            dist, value, sb)
                    elif self.reward_maximizer == 'ppo':
                        policy_loss, value_loss = self.ppo_losses(
                            dist, value, sb)
                    
                    # compute expert matching losses
                    if self.l_term == 'zero':
                        expert_loss = 0.
                    elif self.l_term == 'cross_entropy':
                        expert_loss = cross_entropy(dist.logits, sb.obs.expert)
                    elif self.l_term == 'reveresed_cross_entropy':
                        raise Exception('TODO')
                    
                    # compute entropy
                    entropy = dist.entropy().mean()
                    
                    # combine loss signals
                    loss = (
                        self.policy_loss_coef * policy_loss +
                        self.value_loss_coef * value_loss +
                        self.expert_loss_coef * expert_loss +
                        self.entropy_loss_coef * -entropy
                    )

                    # update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # update memories for next epoch
                    if self.model.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()
                
                # update batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                
                # update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model.parameters()
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # update log
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
        
        # log values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs
    
    def vpg_loss(self, action_dist, value, sb):
        mean = sb.returnn.mean()
        std = sb.returnn.std() + 1e-5
        normalized_return = (sb.returnn - mean) / std
        policy_loss = -(action_dist.log_prob(sb.action) * normalized_return)
        return policy_loss.mean()
    
    def a2c_losses(self, action_dist, value, sb):
        policy_loss = -(action_dist.log_prob(sb.action) * sb.advantage).mean()
        value_loss = (value - sb.returnn).pow(2).mean()
        
        return policy_loss, value_loss
    
    def ppo_losses(self, action_dist, value, sb):
        ratio = torch.exp(action_dist.log_prob(sb.action) - sb.log_prob)
        surr1 = ratio * sb.advantage
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * sb.advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = sb.value + torch.clamp(
            value - sb.value, -self.clip_eps, self.clip_eps)
        surr1 = (value - sb.returnn).pow(2)
        surr2 = (value_clipped - sb.returnn).pow(2)
        value_loss = torch.max(surr1, surr2).mean()
        
        return policy_loss, value_loss
    
    def _get_batches_starting_indexes(self):
        
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[
                (indexes + self.recurrence) % self.num_frames_per_proc != 0
            ]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i:i+num_indexes]
            for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes

