import torch
from torch.nn import Module, ModuleList

from algos.follower import FollowerAlgo
from torch_ac.algos.ppo import PPOAlgo

def make_reshaper(preprocess_obss, follower, verbose=False):
    def reshape(pre_obs, obs, action, reward, done, device):
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        with torch.no_grad():
            preprocessed_pre_obs = preprocess_obss(pre_obs, device=device)
            preprocessed_post_obs = preprocess_obss(obs, device=device)
            _, pre_value = follower(preprocessed_pre_obs)
            _, post_value = follower(preprocessed_post_obs)
        
        shift = (post_value - pre_value)*10
        
        done = torch.tensor(done, dtype=torch.bool, device=device)
        
        if verbose:
            print(pre_value.item())
            print(post_value.item())
        
        # this is bad.
        # if the condition is (shit > 0) then the last frame gets negative
        # return because the post-observation is just as good as the pre
        # if the condition is (shift >= 0) then the agent can sit there
        # doing useless pick-up/drop actions that yield the same state
        # and get positive reward hits all day long
        #neg_bias = 5.
        #shift = ((shift >= 0).float() * (neg_bias + 1.) - neg_bias) * 0.1
        
        #breakpoint()
        
        reward = reward + (shift * ~done)
        return reward
    
    return reshape

class FEAlgo:
    def __init__(self,
        follower_envs,
        explorer_envs,
        fe_model,
        device=None,
        follower_frames_per_proc=None,
        explorer_frames_per_proc=None,
        explorer_rl_algo='ppo',
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        adam_eps=1e-8,
        clip_eps=0.2,
        follower_epochs=4,
        explorer_epochs=4,
        batch_size=256,
        preprocess_obss=None,
    ):
        
        self.follower_algo = FollowerAlgo(
            follower_envs,
            fe_model.follower,
            fe_model.explorer,
            device=device,
            num_frames_per_proc=follower_frames_per_proc,
            discount=discount,
            lr=lr,
            gae_lambda=gae_lambda,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            adam_eps=adam_eps,
            clip_eps=clip_eps,
            epochs=follower_epochs,
            batch_size=batch_size,
            preprocess_obss=preprocess_obss,
        )
        
        reshape = make_reshaper(preprocess_obss, fe_model.follower)
        if explorer_rl_algo == 'ppo':
            self.explorer_algo = PPOAlgo(
                explorer_envs,
                fe_model.explorer,
                device=device,
                num_frames_per_proc=explorer_frames_per_proc,
                discount=discount,
                lr=lr,
                gae_lambda=gae_lambda,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef,
                max_grad_norm=max_grad_norm,
                recurrence=1,
                adam_eps=adam_eps,
                clip_eps=clip_eps,
                epochs=explorer_epochs,
                batch_size=batch_size,
                preprocess_obss=preprocess_obss,
                reshape_reward=reshape,
            )
        else:
            raise ValueError('Unsupported rl algo: %s'%explorer_rl_algo)
        
    def collect_experiences(self):
        follower_exp, follower_log = self.follower_algo.collect_experiences()
        explorer_exp, explorer_log = self.explorer_algo.collect_experiences()
        
        return (
            {'follower':follower_exp, 'explorer':explorer_exp},
            {**follower_log, **explorer_log},
        )
    
    def update_parameters(self, exps):
        follower_log = self.follower_algo.update_parameters(exps['follower'])
        explorer_log = self.explorer_algo.update_parameters(exps['explorer'])
        
        return {**follower_log, **explorer_log}
        
