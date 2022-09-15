import numpy

import torch

class BCAlgo:
    def __init__(self,
        envs,
        acmodel,
        device=None,
        num_frames_per_proc=None,
        lr=0.001,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
    ):
        #discount = 0.99
        #gae_lambda = 
        #super().__init__(
        #    envs,
        #    acmodel,
        #    device,
        #    num_frames_per_proc,
        #    discount,
        #    lr,
        #    gae_lambda,
        #    entropy_coeff,
        #    max_grad_norm,
        #    recurrence,
        #    preprocess_obss,
        #    reshape_reward,
        #)
        self.collected_data = None
    
    def collect_experiences(self):
        if self.collected_data is not None:
            return self.collected_data
        
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obs(self.obs, device=self.device)
            
