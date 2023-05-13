import os

import torch

import numpy

from PIL import Image
from ltron.visualization.drawing import write_text

from algos.distill import Distill

class FEAlgo:
    def __init__(self,
        follower_envs,
        explorer_envs,
        model,
        device=None,
        expert_envs=None,
        expert_frames_per_proc=None,
        follower_frames_per_proc=None,
        explorer_frames_per_proc=None,
        explorer_reward_maximizer='ppo',
        discount=0.99,
        explorer_discount=None,
        lr=0.001,
        gae_lambda=0.95,
        expert_matching_reward_pos=0.1,
        expert_matching_reward_neg=-0.1,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        expert_loss_coef=1.0,
        entropy_loss_coef=0.01,
        true_reward_coef=1.0,
        follower_surrogate_reward_coef=1.0,
        explorer_surrogate_reward_coef=1.0,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        follower_epochs=4,
        explorer_epochs=4,
        batch_size=256,
        preprocess_obss=None,
        render = False,
        pause = 0.,
        override_switching_horizon = None,
        uniform_exploration = False,
        winning_target = 0.75,
    ):
        
        self.expert_frames_per_proc = expert_frames_per_proc
        self.override_switching_horizon = override_switching_horizon
        
        self.tmp_batches = 0
        
        if hasattr(model, 'follower'):
            follower_model = model.follower
        else:
            follower_model = model.model.follower
        
        if self.expert_frames_per_proc != 0:
            self.expert_algo = Distill(
                expert_envs,
                model=follower_model,
                reward_maximizer='zero',
                value_loss_model='zero',
                l_term='cross_entropy',
                r_term='zero',
                plus_R=False,
                on_policy=False,
                explorer_model=model,
                device=device,
                num_frames_per_proc=expert_frames_per_proc,
                discount=discount,
                lr=lr,
                gae_lambda=gae_lambda,
                expert_matching_reward_pos=expert_matching_reward_pos,
                expert_matching_reward_neg=expert_matching_reward_neg,
                policy_loss_coef=policy_loss_coef,
                value_loss_coef=value_loss_coef,
                expert_loss_coef=expert_loss_coef,
                entropy_loss_coef=entropy_loss_coef,
                true_reward_coef=true_reward_coef,
                surrogate_reward_coef=follower_surrogate_reward_coef,
                max_grad_norm=max_grad_norm,
                recurrence=recurrence,
                adam_eps=adam_eps,
                clip_eps=clip_eps,
                epochs=follower_epochs,
                batch_size=batch_size,
                preprocess_obss=preprocess_obss,
                log_prefix='expert_',
                render=render,
                pause=pause,
            )
        
        self.follower_algo = Distill(
            follower_envs,
            model=follower_model,
            reward_maximizer='zero',
            value_loss_model='ppo',
            l_term='cross_entropy',
            r_term='zero',
            plus_R=True,
            on_policy=True,
            explorer_model=model,
            device=device,
            num_frames_per_proc=follower_frames_per_proc,
            discount=discount,
            lr=lr,
            gae_lambda=gae_lambda,
            expert_matching_reward_pos=expert_matching_reward_pos,
            expert_matching_reward_neg=expert_matching_reward_neg,
            policy_loss_coef=policy_loss_coef,
            value_loss_coef=value_loss_coef,
            expert_loss_coef=expert_loss_coef,
            entropy_loss_coef=entropy_loss_coef,
            true_reward_coef=true_reward_coef,
            surrogate_reward_coef=follower_surrogate_reward_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=follower_epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            log_prefix='follower_',
            render=render,
            pause=pause,
            override_switching_horizon=override_switching_horizon,
            uniform_exploration=uniform_exploration,
        )
        
        if explorer_discount is None:
            explorer_discount = discount
        self.explorer_algo = Distill(
            explorer_envs,
            model=model,
            reward_maximizer=explorer_reward_maximizer,
            l_term='expert_if_winning',
            r_term='value_shaping',
            plus_R=True,
            on_policy=True,
            value_model=follower_model,
            device=device,
            num_frames_per_proc=explorer_frames_per_proc,
            discount=explorer_discount,
            lr=lr,
            gae_lambda=gae_lambda,
            expert_matching_reward_pos=expert_matching_reward_pos,
            expert_matching_reward_neg=expert_matching_reward_neg,
            policy_loss_coef=policy_loss_coef,
            value_loss_coef=value_loss_coef,
            expert_loss_coef=0.1, #expert_loss_coef,
            entropy_loss_coef=entropy_loss_coef,
            true_reward_coef=true_reward_coef,
            surrogate_reward_coef=explorer_surrogate_reward_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=explorer_epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
            render=render,
            pause=pause,
            winning_target=winning_target,
        )
        
    def collect_experiences(self):
        if self.expert_frames_per_proc:
            expert_exp, expert_log = self.expert_algo.collect_experiences()
        follower_exp, follower_log = self.follower_algo.collect_experiences()
        explorer_exp, explorer_log = self.explorer_algo.collect_experiences()
        
        if not self.override_switching_horizon:
            if len(explorer_log['num_frames_per_episode']):
                avg_frames_per_episode = round(numpy.mean(
                    explorer_log['num_frames_per_episode']))
                self.follower_algo.switching_horizon = avg_frames_per_episode
                print('setting horizon to:', avg_frames_per_episode)
                if self.expert_frames_per_proc:
                    self.expert_algo.switching_horizon = avg_frames_per_episode
        
        combined_exp = {'follower':follower_exp, 'explorer':explorer_exp}
        combined_log = {**follower_log, **explorer_log}
        
        if self.expert_frames_per_proc:
            combined_exp['expert'] = expert_exp
            combined_log.update(expert_log)
        
        return combined_exp, combined_log
    
    def update_parameters(self, exps):
        if self.expert_frames_per_proc:
            expert_log = self.expert_algo.update_parameters(exps['expert'])
        follower_log = self.follower_algo.update_parameters(exps['follower'])
        explorer_log = self.explorer_algo.update_parameters(exps['explorer'])
        
        combined_log = {**follower_log, **explorer_log}
        if self.expert_frames_per_proc:
            combined_log.update(expert_log)
        
        return combined_log
    
    def cleanup(self):
        self.follower_algo.cleanup()
        self.explorer_algo.cleanup()
