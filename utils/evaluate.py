import torch

from torch_ac.utils.penv import ParallelEnv

import utils

#from envs.env_wrappers import DeferredWrapper

class Evaluator:
    def __init__(
        self,
        env_name,
        num_procs,
        model,
        device,
        preprocessor,
    ):
        seed = 1234567890
        utils.seed(seed)
        
        self.model = model
        self.device = device
        self.preprocessor = preprocessor
        self.num_procs = num_procs
        
        envs = []
        for i in range(num_procs):
            #if i == 0:
            env = utils.make_env(env_name, seed + 10000*i)
            #else:
            #    env = DeferredWrapper(env_name, seed + 10000*i)
            envs.append(env)
        
        self.env = ParallelEnv(envs)
        
        self.logs = []
    
    def cleanup(self):
        self.env.cleanup()
    
    def evaluate(self, num_episodes, argmax=False):
        log = {
            'num_frames_per_episode':[],
            'return_per_episode':[],
            'argmax':argmax,
        }
        obss = self.env.reset()
        
        self.model.eval()
        
        done_count = 0
        episode_return = torch.zeros(self.num_procs, device=self.device)
        episode_frames = torch.zeros(self.num_procs, device=self.device)
        
        if self.model.use_memory:
            memory = torch.zeros(
                self.num_procs, self.model.memory_size, device=self.device)
        
        while done_count < num_episodes:
            with torch.no_grad():
                preprocessed_obss = self.preprocessor(obss, device=self.device)
                if self.model.use_memory:
                    if hasattr(self.model, 'advisor_model'):
                        dist, _, _, memory, *_ = self.model(
                            preprocessed_obss, memory=memory)
                    else:
                        # TODO: real memory
                        dist, _, memory, *_ = self.model(
                            preprocessed_obss, memory=memory)
                else:
                    dist, *_ = self.model(preprocessed_obss)
                
            if argmax:
                actions = dist.probs.max(1, keepdim=True)[1].cpu().numpy()
            else:
                actions = dist.sample().cpu().numpy()
        
            obss, rewards, dones, _ = self.env.step(actions)
            
            masks = 1 - torch.tensor(
                dones, dtype=torch.float, device=self.device)
            
            if self.model.recurrent:
                #pass
                memory *= masks.unsqueeze(1)
            
            episode_return += torch.tensor(
                rewards, device=self.device, dtype=torch.float)
            episode_frames += torch.ones(self.num_procs, device=self.device)
            
            for i, done in enumerate(dones):
                if done:
                    done_count += 1
                    log['return_per_episode'].append(episode_return[i].item())
                    log['num_frames_per_episode'].append(
                        episode_frames[i].item())
            
            episode_return *= masks
            episode_frames *= masks
        
        return_stats = utils.synthesize(log['return_per_episode'])
        log['return_stats'] = return_stats
        frame_stats = utils.synthesize(log['num_frames_per_episode'])
        log['frame_stats'] = frame_stats
        
        print('R:μσmM {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            *return_stats.values()))
        print('F:μσmM {:.4f} {:.4f} {} {}'.format(*frame_stats.values()))
        
        self.model.train()
        
        return log
