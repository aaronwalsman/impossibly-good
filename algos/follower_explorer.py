import os

import torch

import numpy

from PIL import Image
from ltron.visualization.drawing import write_text

#from algos.follower import FollowerAlgo
#from torch_ac.algos.ppo import PPOAlgo
from algos.distill import Distill

#def make_reshaper(preprocess_obss, follower, verbose=False, reshape_coef=10.):
#    def reshape(pre_obs, obs, action, reward, done, device):
#        reward = torch.tensor(reward, dtype=torch.float, device=device)
#        with torch.no_grad():
#            preprocessed_pre_obs = preprocess_obss(pre_obs, device=device)
#            preprocessed_post_obs = preprocess_obss(obs, device=device)
#            _, pre_value = follower(preprocessed_pre_obs)
#            _, post_value = follower(preprocessed_post_obs)
#        
#        shift = (post_value - pre_value)*reshape_coef
#        
#        done = torch.tensor(done, dtype=torch.bool, device=device)
#        
#        if verbose:
#            print(pre_value.item())
#            print(post_value.item())
#        
#        # this is bad.
#        # if the condition is (shift > 0) then the last frame gets negative
#        # return because the post-observation is just as good as the pre
#        # if the condition is (shift >= 0) then the agent can sit there
#        # doing useless pick-up/drop actions that yield the same state
#        # and get positive reward hits all day long
#        # actually, this may be workable again now that we do nothing when done
#        # but doesn't seem necessary yet either
#        #neg_bias = 5.
#        #shift = ((shift >= 0).float() * (neg_bias + 1.) - neg_bias) * 0.1
#        
#        #breakpoint()
#        
#        reward = reward + (shift * ~done)
#        return reward
#    
#    return reshape

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
        extra_fancy = False,
        override_switching_horizon = None,
        uniform_exploration = False,
        fancy_target = 0.75,
    ):
        
        self.expert_frames_per_proc = expert_frames_per_proc
        self.override_switching_horizon = override_switching_horizon
        
        self.tmp_batches = 0
        
        #self.follower_algo = FollowerAlgo(
        #    follower_envs,
        #    fe_model.model.follower,
        #    #fe_model.explorer,
        #    fe_model,
        #    device=device,
        #    num_frames_per_proc=follower_frames_per_proc,
        #    discount=discount,
        #    lr=lr,
        #    gae_lambda=gae_lambda,
        #    value_loss_coef=value_loss_coef,
        #    max_grad_norm=max_grad_norm,
        #    adam_eps=adam_eps,
        #    clip_eps=clip_eps,
        #    epochs=follower_epochs,
        #    batch_size=batch_size,
        #    preprocess_obss=preprocess_obss,
        #)
        
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
            extra_fancy=extra_fancy,
            override_switching_horizon=override_switching_horizon,
            uniform_exploration=uniform_exploration,
        )
        
        if explorer_discount is None:
            explorer_discount = discount
        self.explorer_algo = Distill(
            explorer_envs,
            model=model,
            reward_maximizer=explorer_reward_maximizer,
            l_term='fancy_value',
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
            extra_fancy=extra_fancy,
            fancy_target=fancy_target,
        )
            
        
        #reshape = make_reshaper(preprocess_obss, fe_model.model.follower)
        #if explorer_rl_algo == 'ppo':
        #    self.explorer_algo = PPOAlgo(
        #        explorer_envs,
        #        #fe_model.explorer,
        #        fe_model,
        #        device=device,
        #        num_frames_per_proc=explorer_frames_per_proc,
        #        discount=discount,
        #        lr=lr,
        #        gae_lambda=gae_lambda,
        #        entropy_coef=entropy_coef,
        #        value_loss_coef=value_loss_coef,
        #        max_grad_norm=max_grad_norm,
        #        recurrence=1,
        #        adam_eps=adam_eps,
        #        clip_eps=clip_eps,
        #        epochs=explorer_epochs,
        #        batch_size=batch_size,
        #        preprocess_obss=preprocess_obss,
        #        reshape_reward=reshape,
        #    )
        #else:
        #    raise ValueError('Unsupported rl algo: %s'%explorer_rl_algo)
        
    def collect_experiences(self):
        if self.expert_frames_per_proc:
            expert_exp, expert_log = self.expert_algo.collect_experiences()
        follower_exp, follower_log = self.follower_algo.collect_experiences()
        explorer_exp, explorer_log = self.explorer_algo.collect_experiences()
        
        '''
        # START TMP ============================================================
        if self.tmp_batches % 10 == 0:
            out_dir = './tmp_follower_dump_%04i'%self.tmp_batches
            os.makedirs(out_dir)
            n_exp = follower_exp.obs.image.shape[0]
            print('SAVING FOLLOWER DATA')
            for i in range(n_exp):
                image = follower_exp.obs.image[i,0]
                image = (image * 255).cpu().numpy().astype(numpy.uint8)
                text = '\n'.join((
                    'Act: %s'%follower_exp.action[i].item(),
                    'UE: %s'%follower_exp.use_explorer[i].item(),
                    'Exp: %s'%follower_exp.obs.expert[i].item(),
                    'Mask: %s'%follower_exp.mask[i].item(),
                ))
                image = write_text(image, text, size=10, color=255)
                image = Image.fromarray(image)
                image.save(os.path.join(
                    out_dir, 'img_%04i_%06i.png'%(self.tmp_batches, i)))
            
            out_dir = './tmp_explorer_dump_%04i'%self.tmp_batches
            os.makedirs(out_dir)
            n_exp = explorer_exp.obs.image.shape[0]
            print('SAVING EXPLORER DATA')
            for i in range(n_exp):
                image = explorer_exp.obs.image[i,0]
                image = (image * 255).cpu().numpy().astype(numpy.uint8)
                text = '\n'.join((
                    'Act: %s'%explorer_exp.action[i].item(),
                    'Exp: %s'%explorer_exp.obs.expert[i].item(),
                ))
                image = write_text(image, text, size=10, color=255)
                image = Image.fromarray(image)
                image.save(os.path.join(
                    out_dir, 'img_%04i_%06i.png'%(self.tmp_batches, i)))
        self.tmp_batches += 1
        # END TMP ==============================================================
        '''
        
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
