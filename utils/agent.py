import torch
from torch.distributions import Categorical

import utils
from .other import device
from model import (
    ImpossiblyGoodACPolicy,
    ImpossiblyGoodFollowerExplorerPolicy,
    ImpossiblyGoodFollowerExplorerSwitcherPolicy,
    VanillaACPolicy,
)

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(
        self,
        obs_space,
        action_space,
        model_dir,
        argmax=False,
        num_envs=1,
        use_memory=False,
        use_text=False,
        use_follower=False,
    ):
        self.argmax = argmax
        self.num_envs = num_envs
        self.use_follower = use_follower

        #if self.acmodel.recurrent:
        #    self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        
        try:
            obs_space_mod, self.preprocess_obss = utils.get_obss_preprocessor(
                obs_space, image_dtype=torch.long)
            self.acmodel = ImpossiblyGoodACPolicy(obs_space_mod, action_space)
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
            self.arch = 'ig'
        except:
            try:
                obs_space_mod, self.preprocess_obss = (
                    utils.get_obss_preprocessor(
                        obs_space, image_dtype=torch.long))
                self.acmodel = ImpossiblyGoodFollowerExplorerPolicy(
                    obs_space_mod, action_space)
                self.acmodel.load_state_dict(utils.get_model_state(model_dir))
                self.arch = 'fe'
            except:
                try:
                    obs_space_mod, self.preprocess_obss = (
                        utils.get_obss_preprocessor(
                            obs_space, image_dtype=torch.long))
                    self.acmodel = ImpossiblyGoodFollowerExplorerSwitcherPolicy(
                        obs_space_mod, action_space)
                    self.acmodel.load_state_dict(
                        utils.get_model_state(model_dir))
                    self.arch = 'fe'
                
                except:
                    obs_space_mod, self.preprocess_obss = (
                        utils.get_obss_preprocessor(
                            obs_space, image_dtype=torch.float))
                    self.acmodel = VanillaACPolicy(obs_space_mod, action_space)
                    self.acmodel.load_state_dict(
                        utils.get_model_state(model_dir))
                    self.arch = 'vanilla'
        
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            #if self.acmodel.recurrent:
            #    dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            #else:
            #if self.arch == 'fe':
            #    f_dist, f_value, e_dist, e_value = self.acmodel(
            #        preprocessed_obss)
            #    if self.fe_rollout_mode == 'follower':
            #        dist = f_dist
            #    elif self.fe_rollout_mode == 'explorer':
            #        dist = e_dist
            #    #elif self.fe_rollout_mode == 'max_value':
            #    #    use_follower = f_value > e_value
            #    #    use_follower = use_follower.reshape(-1,1)
            #    #    dist = Categorical(logits=
            #    #        f_dist.logits*use_follower +
            #    #        e_dist.logits*~use_follower
            #    #    )
            #    else:
            #        raise ValueError(
            #            'Unknown rollout mode: %s'%self.fe_rollout_mode)
            #elif self.arch == 'ig':
            #    dist, _ = self.acmodel(preprocessed_obss)
            #elif self.arch == 'vanilla':
            #    dist, *_ = self.acmodel(preprocessed_obss, memory=None)
            if self.use_follower:
                rollout_model = self.acmodel.model.follower
            else:
                rollout_model = self.acmodel
            
            if self.arch == 'vanilla':
                dist, *_ = self.acmodel(preprocessed_obss, memory=None)
            else:
                dist, *_ = rollout_model(preprocessed_obss)
        
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            #self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
