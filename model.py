import torch
from torch.nn import Module, Sequential, Embedding, Linear, ReLU, Tanh
from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
NUM_OBJECTS = max(OBJECT_TO_IDX.values())+1
NUM_COLORS = max(COLOR_TO_IDX.values())+1

from torch_ac.model import ACModel

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ImpossiblyGoodACModel(Module, ACModel):
    recurrent = False
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        embedding_channels=16
        hidden_channels=256
        
        # feature embeddings
        self.image_object_embedding = Embedding(NUM_OBJECTS, 16)
        self.image_color_embedding = Embedding(NUM_COLORS, 16)
        self.observed_color_embedding = Embedding(NUM_COLORS, 16)
        
        # backbone
        h, w = obs_space['image'][:2]
        self.backbone = Sequential(
            ReLU(),
            Linear(h*w*embedding_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
        
        # actor
        self.actor = Sequential(
            Linear(hidden_channels, hidden_channels),
            Tanh(),
            Linear(hidden_channels, action_space.n),
        )
        
        # critic
        self.critic = Sequential(
            Linear(hidden_channels, hidden_channels),
            Tanh(),
            Linear(hidden_channels, 1),
        )
        
        # initialize
        self.apply(init_params)
    
    def forward(self, obs):
        image_x = obs.image.transpose(1,3).transpose(2,3)
        observed_color_x = obs.observed_color
        
        # featurize objects
        image_object_x = self.image_object_embedding(image_x[:,0])
        
        # featurize colors
        image_color_x = self.image_color_embedding(image_x[:,1])
        
        # combine objects and colors
        image_x = image_object_x + image_color_x
        
        # reshape
        b, h, w, c = image_x.shape
        image_x = image_x.reshape(b, h*w, c)
        
        # featurize observed colors
        observed_color_x = self.observed_color_embedding(observed_color_x)
        observed_color_x = observed_color_x.reshape(b, 1, c)
        
        # combined image features and the observed_color_feature
        x = image_x + observed_color_x
        x = x.reshape(b, h*w*c)
        
        # backbone
        x = self.backbone(x)
        
        # actor
        a = self.actor(x)
        distribution = Categorical(logits=log_softmax(a, dim=-1))
        
        # critic
        value = self.critic(x).view(b)
        
        # return
        return distribution, value
