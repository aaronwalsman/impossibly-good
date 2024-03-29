import time

import numpy

import torch
from torch.nn.functional import cross_entropy, kl_div
from torch.distributions import Categorical

from torch_ac.utils import DictList, ParallelEnv

class Distill:
    def __init__(self,
        envs,
        model,
        reward_maximizer='ppo',
        value_loss_model=None,
        # zero | cross_entropy | reverse_cross_entropy
        l_term='zero',
        # zero | log_p | cross_entropy | value_shaping
        r_term='zero',
        plus_R=False,
        on_policy=True,
        value_model=None,
        explorer_model=None,
        skip_immediate_reward=False,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        expert_matching_reward_pos=0.1,
        expert_matching_reward_neg=-0.1,
        expert_smoothing=0.01,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        expert_loss_coef=1.0,
        entropy_loss_coef=0.01,
        true_reward_coef=1.0,
        surrogate_reward_coef=1.0,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        log_prefix='',
        render=False,
        pause=0.,
        use_advisor=False,
        advisor_alpha=10.,
        override_switching_horizon=None,
        uniform_exploration=False,
        winning_target=0.75,
    ):
        
        if num_frames_per_proc is None:
            if reward_maximizer == 'a2c':
                num_frames_per_proc = 8
            else:
                num_frames_per_proc = 128
        
        if value_loss_model is None:
            if reward_maximizer == 'ppo':
                value_loss_model = 'ppo'
            elif reward_maximizer == 'a2c':
                value_loss_model = 'a2c'
            else:
                value_loss_model = 'zero'
        
        # store parameters
        self.env = ParallelEnv(envs)
        self.model = model
        self.reward_maximizer = reward_maximizer
        self.value_loss_model = value_loss_model
        self.l_term = l_term
        self.r_term = r_term
        self.plus_R = plus_R
        self.on_policy = on_policy
        self.value_model = value_model
        self.explorer_model = explorer_model
        self.skip_immediate_reward = skip_immediate_reward
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.expert_matching_reward_pos = expert_matching_reward_pos
        self.expert_matching_reward_neg = expert_matching_reward_neg
        self.expert_smoothing = expert_smoothing
        self.policy_loss_coef = policy_loss_coef
        self.value_loss_coef = value_loss_coef
        self.expert_loss_coef = expert_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.true_reward_coef = true_reward_coef
        self.surrogate_reward_coef = surrogate_reward_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.adam_eps = adam_eps
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocess_obss = preprocess_obss #or default_preprocess_obss
        self.log_prefix = log_prefix
        self.render = render
        self.pause = pause
        self.use_advisor = use_advisor
        self.advisor_alpha = advisor_alpha
        self.uniform_exploration = uniform_exploration
        self.winning_target = winning_target
        
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        
        # initialize switching horizon
        if self.explorer_model is not None:
            if override_switching_horizon is not None:
                self.switching_horizon = override_switching_horizon
            else:
                self.switching_horizon = self.env.envs[0].max_steps
        
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
        self.true_rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        if self.explorer_model is not None:
            self.switching_time = torch.randint(
                0, self.switching_horizon, (shape[1],), device=device)
            self.use_explorer = torch.zeros(
                *shape, dtype=torch.bool, device=device)
        
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
        
        # initialize value_mean
        self.model.value_mean = torch.zeros(1, device=self.device)
    
    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            
            # convert observations to tensors
            preprocessed_obs = self.preprocess_obss(
                self.obs, device=self.device)
            
            # forward pass
            with torch.no_grad():
                if self.model.recurrent:
                    if self.use_advisor:
                        dist, value, aux_dist, memory = self.model(
                            preprocessed_obs,
                            self.memory*self.mask.unsqueeze(1)
                        )
                    else:
                        dist, value, memory = self.model(
                            preprocessed_obs,
                            self.memory*self.mask.unsqueeze(1)
                        )
                else:
                    if self.use_advisor:
                        dist, value, aux_dist = self.model(preprocessed_obs)
                    else:
                        dist, value = self.model(preprocessed_obs)
            
            if self.on_policy:
                if self.explorer_model is None:
                    action = dist.sample()
                else:
                    with torch.no_grad():
                        if self.explorer_model.recurrent:
                            explorer_dist, *_ = self.explorer_model(
                                preprocessed_obs,
                                self.memory*self.mask.unsqueeze(1),
                            )
                        else:
                            explorer_dist, *_ = self.explorer_model(
                                preprocessed_obs)
                    use_explorer = (
                        preprocessed_obs.step < self.switching_time)
                    if self.uniform_exploration:
                        batch_size, max_action = dist.logits.shape
                        explorer_action = torch.ones(
                            (batch_size,), device=self.device).long()
                    else:
                        explorer_action = explorer_dist.sample()
                    policy_action = dist.sample()
                    action = (
                        explorer_action * use_explorer +
                        policy_action * ~use_explorer
                    )
            else:
                if self.explorer_model is None:
                    action = preprocessed_obs.expert
                else:
                    with torch.no_grad():
                        if self.explorer_model.recurrent:
                            explorer_dist, *_ = self.explorer_model(
                                preprocessed_obs,
                                self.memory*self.mask.unsqueeze(1)
                            )
                        else:
                            explorer_dist, *_ = self.explorer_model(
                                preprocessed_obs)
                    use_explorer = (
                        preprocessed_obs.step < self.switching_time)
                    explorer_action = explorer_dist.sample()
                    action = (
                        explorer_action * use_explorer +
                        preprocessed_obs.expert * ~use_explorer
                    )
            
            # get value before
            if self.r_term == 'value_shaping':
                if self.value_model is None:
                    value_before_update = preprocessed_obs.value.cpu().numpy()
                else:
                    with torch.no_grad():
                        if self.value_model.recurrent:
                            _, value_before_update, *_ = self.value_model(
                                preprocessed_obs,
                                self.memory*self.mask.unsqueeze(1)
                            )
                        else:
                            _, value_before_update, *_ = self.value_model(
                                preprocessed_obs)
                    value_before_update = (
                        value_before_update.detach().cpu().numpy())
            
            if self.render:
                self.env.envs[0].render('human')
                if self.explorer_model and use_explorer[0]:
                    print('EXPLORING:',
                        preprocessed_obs.step[0].item(),
                        self.switching_time[0].item()
                    )
                    time.sleep(0.25)
                print('Expert:', self.obs[0]['expert'])
                if hasattr(self.env.envs[0], 'Actions'):
                    print('Action:', self.env.envs[0].Actions(action[0].item()))
                else:
                    print('Action:', action[0].item())
            if self.pause:
                command = input()
                if command == 'breakpoint':
                    breakpoint()
                try:
                    if hasattr(self.env.envs[0], 'Actions'):
                        override_action = getattr(
                            self.env.envs[0].Actions, command)
                    else:
                        override_action = int(command)
                    print('OVERRIDE ACTION: %s'%override_action)
                except AttributeError:
                    print('INVALID OVERRIDE ACTION: %s'%command)
                    override_action = None
                
                if override_action is not None:
                    action[0] = int(override_action)
            
            # step
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            # compute reward surrogate
            surrogate_reward = numpy.zeros((len(reward),))
            if self.plus_R:
                surrogate_reward += numpy.array(reward) * self.true_reward_coef
            if self.r_term == 'log_p':
                expert_match = preprocessed_obs.expert == action
                n = dist.probs.shape[-1]
                assert self.expert_smoothing <= (1./n)
                log_p = (
                    numpy.log(self.expert_smoothing) * ~expert_match +
                    numpy.log(1. - (n-1)*self.expert_smoothing) * expert_match
                )
                surrogate_reward += (
                    log_p.detach().cpu().numpy() * self.surrogate_reward_coef)
            elif self.r_term == 'expert_matching_reward':
                expert_match = preprocessed_obs.expert == action
                surrogate_reward += (
                    expert_match * self.expert_matching_reward_pos + 
                    ~expert_match * self.expert_matching_reward_neg
                ).cpu().numpy() * self.surrogate_reward_coef
            elif self.r_term == 'cross_entropy':
                ce = -cross_entropy(dist.logits, preprocessed_obs.expert)
                surrogate_reward += (
                    ce.detach().cpu().numpy() * self.surrogate_reward_coef)
            elif self.r_term == 'kl_divergence':
                expert_dist = torch.zeros_like(dist.logits)
                b = expert_dist.shape[0]
                expert_dist[range(b), preprocessed_obs.expert] = 1
                kl = -kl_div(dist.logits, expert_dist, reduction='none')
                kl = torch.sum(kl, dim=1)
                surrogate_reward += (
                    kl.detach().cpu().numpy() * self.surrogate_reward_coef)
            elif self.r_term == 'value_shaping':
                if self.value_model is None:
                    value_after_update = preprocessed_obs.value.cpu().numpy()
                else:
                    post_obs = self.preprocess_obss(obs, device=self.device)
                    if self.value_model.recurrent:
                        post_mask = 1 - torch.tensor(
                            done, device=self.device, dtype=torch.float)
                        post_mask = post_mask.unsqueeze(1)
                        _, value_after_update, *_ = self.value_model(
                            post_obs, memory*post_mask)
                    else:
                        _, value_after_update, *_ = self.value_model(post_obs)
                    value_after_update = (
                        value_after_update.detach().cpu().numpy())
                value_shaping = value_after_update - value_before_update
                value_shaping = value_shaping * ~numpy.array(done)
                surrogate_reward += value_shaping * self.surrogate_reward_coef
            elif self.r_term == 'zero':
                pass
            else:
                raise ValueError('bad r_term')
            
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
            self.true_rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)
            if self.explorer_model is not None:
                d = torch.BoolTensor(done).to(self.device)
                new_switching_time = torch.randint(
                    0, self.switching_horizon, self.switching_time.shape,
                    device=self.device)
                self.switching_time = (
                    new_switching_time * d +
                    self.switching_time * ~d
                )
                self.use_explorer[i] = use_explorer
            
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
                _, next_value, *_ = self.model(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value, *_ = self.model(preprocessed_obs)

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
        
        # remove immediate reward if skip_immediate_reward is True
        if self.skip_immediate_reward:
            self.advantages -= self.rewards
            if self.plus_R:
                self.advantages += self.true_rewards
        
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
        if self.explorer_model is not None: # and self.on_policy:
            exps.use_explorer = self.use_explorer.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "%sreturn_per_episode"%self.log_prefix:
                self.log_return[-keep:],
            "%sreshaped_return_per_episode"%self.log_prefix:
                self.log_reshaped_return[-keep:],
            "%snum_frames_per_episode"%self.log_prefix:
                self.log_num_frames[-keep:],
            "%snum_frames"%self.log_prefix:
                self.num_frames
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
                        if self.use_advisor:
                            dist, value, aux_dist, memory = self.model(
                                sb.obs, memory*sb.mask)
                        else:
                            dist, value, memory = self.model(
                                sb.obs, memory*sb.mask)
                    else:
                        if self.use_advisor:
                            dist, value, aux_dist = self.model(sb.obs)
                        else:
                            dist, value = self.model(sb.obs)
                    
                    b, n = dist.probs.shape
                    
                    # compute advisor aux loss
                    if self.use_advisor:
                        advisor_aux_loss = cross_entropy(
                            aux_dist.logits, sb.obs.expert)
                        
                        expert_dist = torch.zeros_like(aux_dist.logits)
                        expert_dist[range(b), sb.obs.expert] = 1.
                        w = kl_div(
                            aux_dist.logits, expert_dist, reduction='none')
                        w = torch.sum(w, dim=1)
                        advisor_weights = torch.exp(-self.advisor_alpha * w)
                        loss_reduction='none'
                    else:
                        loss_reduction='mean'
                    
                    # compute policy loss
                    if self.reward_maximizer == 'zero':
                        if loss_reduction == 'none':
                            policy_loss = torch.zeros(b, device=self.device)
                        elif loss_reduction == 'mean':
                            policy_loss = torch.zeros(1, device=self.device)
                        else:
                            raise ValueError
                    elif self.reward_maximizer == 'vpg':
                        policy_loss = self.vpg_policy_loss(dist, sb)
                    elif self.reward_maximizer == 'a2c':
                        policy_loss = self.a2c_policy_loss(dist, sb)
                    elif self.reward_maximizer == 'ppo':
                        policy_loss = self.ppo_policy_loss(
                            dist, sb, reduction=loss_reduction)
                    
                    # compute value loss
                    if self.value_loss_model == 'zero':
                        if loss_reduction == 'mean':
                            value_loss = torch.zeros(1, device=self.device)
                        elif loss_reduction == 'none':
                            value_loss = torch.zeros(b, device=self.device)
                    elif self.value_loss_model == 'ppo':
                        value_loss = self.ppo_value_loss(
                            value, sb, reduction=loss_reduction)
                        #if self.log_prefix == 'follower_':
                        #    breakpoint()
                    elif self.value_loss_model == 'a2c':
                        value_loss = self.a2c_value_loss(value, sb)
                    
                    if self.use_advisor:
                        policy_loss = (
                            policy_loss * (1. - advisor_weights)).mean()
                        value_loss = (
                            value_loss * (1. - advisor_weights)).mean()
                    
                    self.model.value_mean = (
                        self.model.value_mean * 0.99 +
                        torch.mean(sb.returnn) * 0.01
                    )
                    
                    # compute expert matching losses
                    if self.l_term == 'zero':
                        expert_loss = 0.
                    elif self.l_term == 'cross_entropy':
                        expert_loss = cross_entropy(
                            dist.logits,
                            sb.obs.expert,
                            reduction=loss_reduction
                        )
                    elif self.l_term == 'reverse_cross_entropy':
                        assert self.expert_smoothing <= (1./n)
                        p_expert = torch.full_like(
                            dist.probs, self.expert_smoothing)
                        p_expert[range(b), sb.obs.expert.long()] = (
                            1. - (n-1)*self.expert_smoothing)
                        expert_loss = -(torch.log(p_expert) * dist.probs)
                        expert_loss = torch.sum(expert_loss, dim=-1).mean()
                    elif self.l_term == 'expert_if_winning':
                        with torch.no_grad():
                            if self.value_model.recurrent:
                                value_dist, value_value, *_ = self.value_model(
                                    sb.obs, memory*sb.mask)
                            else:
                                value_dist, value_value = self.value_model(
                                    sb.obs)
                        
                        do_ce = value_value > self.winning_target
                        ce = -torch.sum(dist.logits * value_dist.probs, dim=-1)
                        ce = ce * do_ce
                        denominator = torch.sum(do_ce).float() + 1e-6
                        expert_loss = torch.sum(ce) / denominator
                        
                    else:
                        raise Exception('bad l_term')
                    
                    if self.use_advisor:
                        expert_loss = (advisor_weights * expert_loss).mean()
                    
                    # compute entropy
                    entropy = dist.entropy().mean()
                    
                    # combine loss signals
                    if self.use_advisor:
                        loss = (
                            self.policy_loss_coef * policy_loss +
                            self.value_loss_coef * value_loss +
                            self.expert_loss_coef * expert_loss +
                            self.expert_loss_coef * advisor_aux_loss +
                            self.entropy_loss_coef * -entropy
                        )
                    else:
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
                    for p in self.model.parameters() if p.grad is not None
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
            "%sentropy"%self.log_prefix: numpy.mean(log_entropies),
            "%svalue"%self.log_prefix: numpy.mean(log_values),
            "%spolicy_loss"%self.log_prefix: numpy.mean(log_policy_losses),
            "%svalue_loss"%self.log_prefix: numpy.mean(log_value_losses),
            "%sgrad_norm"%self.log_prefix: numpy.mean(log_grad_norms)
        }

        return logs
    
    def vpg_policy_loss(self, dist, sb):
        mean = sb.returnn.mean()
        std = sb.returnn.std() + 1e-5
        normalized_return = (sb.returnn - mean) / std
        policy_loss = -(dist.log_prob(sb.action) * normalized_return)
        return policy_loss.mean()
    
    def a2c_policy_loss(self, dist, sb):
        policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()
        return policy_loss
    
    def a2c_value_loss(self, value, sb):
        value_loss = (value - sb.returnn).pow(2)
        if self.explorer_model is not None: # and self.on_policy:
            filtered_value_loss = torch.sum(value_loss * ~sb.use_explorer)
            divisor = torch.sum(~sb.use_explorer) + 1e-6
            value_loss = filtered_value_loss / divisor
        else:
            value_loss = value_loss.mean()
        return value_loss
    
    def ppo_policy_loss(self, dist, sb, reduction='mean'):
        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
        surr1 = ratio * sb.advantage
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * sb.advantage
        policy_loss = -torch.min(surr1, surr2)
        
        if reduction == 'mean':
            return policy_loss.mean()
        elif reduction == 'none':
            return policy_loss
        else:
            raise ValueError
    
    def ppo_value_loss(self, value, sb, reduction='mean'):
        value_clipped = sb.value + torch.clamp(
            value - sb.value, -self.clip_eps, self.clip_eps)
        surr1 = (value - sb.returnn).pow(2)
        surr2 = (value_clipped - sb.returnn).pow(2)
        max_surr = torch.max(surr1, surr2)
        if self.explorer_model is not None: # and self.on_policy:
            filtered_max_surr = torch.sum(max_surr * ~sb.use_explorer)
            divisor = torch.sum(~sb.use_explorer) + 1e-6
            value_loss = filtered_max_surr / divisor
        else:
            if reduction == 'mean':
                value_loss = max_surr.mean()
            elif reduction == 'none':
                value_loss = max_surr
            else:
                raise ValueError
        
        return value_loss
    
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
        #print(batches_starting_indexes)
        return batches_starting_indexes
    
    def cleanup(self):
        self.env.cleanup()
