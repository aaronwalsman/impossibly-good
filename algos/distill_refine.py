import torch

from algos.distill import Distill

class DistillRefine:
    def __init__(self,
        distill_envs,
        refine_envs,
        model,
        
        distill_reward_maximizer = 'ppo',
        distill_value_loss_model = None,
        distill_l_term = 'zero',
        distill_r_term = 'zero',
        distill_plus_R = False,
        distill_on_policy = True,
        distill_skip_immediate_reward = False,
        
        refine_reward_maximizer = 'ppo',
        refine_value_loss_model = None,
        refine_l_term = 'zero',
        refine_r_term = 'zero',
        refine_plus_R = False,
        refine_skip_immediate_reward = False,
        
        device=None,
        
        distill_frames_per_proc=None,
        refine_frames_per_proc=None,
        
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        
        distill_expert_matching_reward_pos=0.1,
        distill_expert_matching_reward_neg=-0.1,
        distill_expert_smoothing=0.01,
        
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
        switch_frames=80000,
        render=False,
        pause=0.
    ):
        
        self.distill_algo = Distill(
            distill_envs,
            model,
            reward_maximizer = distill_reward_maximizer,
            value_loss_model = distill_value_loss_model,
            l_term = distill_l_term,
            r_term = distill_r_term,
            plus_R = distill_plus_R,
            on_policy = distill_on_policy,
            skip_immediate_reward = distill_skip_immediate_reward,
            device = device,
            num_frames_per_proc = distill_frames_per_proc,
            discount = discount,
            lr = lr,
            gae_lambda = gae_lambda,
            expert_matching_reward_pos = distill_expert_matching_reward_pos,
            expert_matching_reward_neg = distill_expert_matching_reward_neg,
            expert_smoothing = distill_expert_smoothing,
            policy_loss_coef = policy_loss_coef,
            value_loss_coef = value_loss_coef,
            expert_loss_coef = expert_loss_coef,
            entropy_loss_coef = entropy_loss_coef,
            max_grad_norm = max_grad_norm,
            recurrence = recurrence,
            adam_eps = adam_eps,
            clip_eps = clip_eps,
            epochs = epochs,
            batch_size = batch_size,
            preprocess_obss = preprocess_obss,
            render = render,
            pause = pause,
        )

        self.refine_algo = Distill(
            refine_envs,
            model,
            reward_maximizer = refine_reward_maximizer,
            value_loss_model = refine_value_loss_model,
            l_term = refine_l_term,
            r_term = refine_r_term,
            plus_R = refine_plus_R,
            on_policy = True,
            skip_immediate_reward = refine_skip_immediate_reward,
            device = device,
            num_frames_per_proc = refine_frames_per_proc,
            discount = discount,
            lr = lr,
            gae_lambda = gae_lambda,
            policy_loss_coef = policy_loss_coef,
            value_loss_coef = value_loss_coef,
            expert_loss_coef = expert_loss_coef,
            entropy_loss_coef = entropy_loss_coef,
            max_grad_norm = max_grad_norm,
            recurrence = recurrence,
            adam_eps = adam_eps,
            clip_eps = clip_eps,
            epochs = epochs,
            batch_size = batch_size,
            preprocess_obss = preprocess_obss,
            render = render,
            pause = pause,
        )
        
        self.mode = 'distill'
        self.num_frames = 0
        self.switch_frames = switch_frames
    
    def collect_experiences(self):
        if self.mode == 'distill':
            exps, log = self.distill_algo.collect_experiences()
            self.num_frames += log['num_frames']
        
        else:
            exps, log = self.refine_algo.collect_experiences()
        
        return exps, log
    
    def update_parameters(self, exps):
        if self.mode == 'distill':
            log = self.distill_algo.update_parameters(exps)
            if self.num_frames >= self.switch_frames:
                print('Switching To Refine')
                self.mode = 'refine'
        else:
            log = self.refine_algo.update_parameters(exps)
        
        return log
