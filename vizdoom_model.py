import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ltron_torch.models.padding import make_padding_mask

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ImpossiblyGoodDecoderHead(nn.Module):
    def __init__(self, out_channels, in_channels=512, hidden_channels=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x):
        return self.head(x)

class ImpossiblyGoodImageMemoryEncoder(nn.Module):
    def __init__(self, h, w, input_dimensions=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dimensions, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        tmp = torch.zeros(1, input_dimensions, h, w)
        tmp = self.conv1(tmp)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        conv_output_dims = int(numpy.prod(tmp.shape))
        
        self.fc1 = nn.Linear(conv_output_dims, 512)
        
        self.image_to_memory_head = ImpossiblyGoodDecoderHead(
            16, hidden_channels=256)
        self.q_head = ImpossiblyGoodDecoderHead(
            16, hidden_channels=256)
        self.v_head = ImpossiblyGoodDecoderHead(
            512, hidden_channels=256, in_channels=16)
    
    def forward(self, x, step, memory=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        b, *_ = x.shape
        x = x.reshape(b, -1)
        x = F.relu(self.fc1(x))
        
        k = memory.view(b, 75, 16)
        q = self.q_head(x).view(b, 1, 16)
        qk = torch.sum(q*k, dim=-1)
        mask = make_padding_mask(step+1, (b, 75), 1, 0)
        qk.masked_fill_(mask, float('-inf'))
        a = torch.softmax(qk, dim=-1)
        #print(q)
        #print(qk)
        #print(a)
        #breakpoint()
        v = k * a.view(b, 75, 1)
        v = torch.sum(v, dim=1)
        v = self.v_head(v)
        
        m = torch.zeros_like(k)
        m[range(b), step] = self.image_to_memory_head(x)
        memory = memory + m.view(b, 1200)
        
        x = x + v
        
        return x, memory

class ImpossiblyGoodImageEncoder(nn.Module):
    def __init__(self, h, w, input_dimensions=1, use_memory=False):
        super().__init__()
        self.use_memory = use_memory
        self.conv1 = nn.Conv2d(input_dimensions, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        tmp = torch.zeros(1, input_dimensions, h, w)
        tmp = self.conv1(tmp)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        conv_output_dims = int(numpy.prod(tmp.shape))
        
        self.fc1 = nn.Linear(conv_output_dims, 512)
        
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(512, 512)
    
    def forward(self, x, memory=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        b, *_ = x.shape
        x = x.reshape(b, -1)
        x = F.relu(self.fc1(x))
        
        if self.use_memory:
            h = (memory[:,:512], memory[:,512:])
            h = self.memory_rnn(x, h)
            x = h[0]
            memory = torch.cat(h, dim=1)
            return x, memory
        
        else:
            return x

class ImpossiblyGoodVizdoomACPolicy(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()
        
        self.mode = 'lstm' #'attention' # 'lstm'
        
        self.use_memory = use_memory
        self.recurrent = use_memory
        if self.use_memory:
            if self.mode == 'attention':
                self.memory_size = 1200
            elif self.mode == 'lstm':
                self.memory_size = 512*2
            else:
                raise ValueError
        
        c, h, w = obs_space['image']
        
        if self.mode == 'lstm':
            self.encoder = ImpossiblyGoodImageEncoder(
                h, w, input_dimensions=c, use_memory=use_memory)
        elif self.mode == 'attention':
            self.encoder = ImpossiblyGoodImageMemoryEncoder(
                h, w, input_dimensions=c)
        else:
            raise ValueError
        
        #self.actor = nn.Linear(512, action_space.n)
        #self.critic = nn.Linear(512, 1)
        self.actor = ImpossiblyGoodDecoderHead(action_space.n)
        self.critic = ImpossiblyGoodDecoderHead(1)
    
    def forward(self, obs, memory=None, return_x=False):
        if self.use_memory:
            if self.mode == 'lstm':
                x, memory = self.encoder(obs.image, memory=memory)
            elif self.mode == 'attention':
                x, memory = self.encoder(obs.image, obs.step, memory=memory)
        else:
            x = self.encoder(obs.image)
        
        value = self.critic(x).reshape(-1)
        a = self.actor(x)
        dist = Categorical(logits=F.log_softmax(a, dim=1))
        
        if self.use_memory:
            if return_x:
                return dist, value, memory, x
            else:
                return dist, value, memory
        else:
            if return_x:
                return dist, value, x
            else:
                return dist, value

class ImpossiblyGoodVizdoomAdvisorPolicy(ImpossiblyGoodVizdoomACPolicy):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__(obs_space, action_space, use_memory=use_memory)
        self.advisor_model = True
        self.aux_head = ImpossiblyGoodDecoderHead(action_space.n)
    
    def forward(self, obs, memory=None):
        parts = super().forward(obs, memory=memory, return_x=True)
        if self.use_memory:
            dist, value, memory, x = parts
        else:
            dist, value, x = parts
        
        aux = self.aux_head(x)
        aux_dist = Categorical(logits=F.log_softmax(aux, dim=1))
        
        if self.use_memory:
            return dist, value, aux_dist, memory
        
        else:
            return dist, value, aux_dist

class ImpossiblyGoodVizdoomOldFollowerExplorerPolicy(nn.Module):
    def __init__(self, observation_space, action_space, use_memory=False):
        raise Exception('DEPRECATED!')
        super().__init__()
        
        self.follower = ImpossiblyGoodVizdoomACPolicy(
            observation_space, action_space, use_memory=use_memory)
        
        self.explorer = ImpossiblyGoodVizdoomACPolicy(
            observation_space, action_space, use_memory=use_memory)
        
        self.use_memory = self.explorer.use_memory
        self.recurrent = self.explorer.recurrent
        if self.use_memory:
            self.memory_size = self.explorer.memory_size
    
    def forward(self, *args, **kwargs):
        return self.explorer(*args, **kwargs)

class ImpossiblyGoodVizdoomFollowerExplorerPolicy(
    ImpossiblyGoodVizdoomACPolicy
):
    def __init__(self, observation_space, action_space, use_memory=False):
        super().__init__(observation_space, action_space, use_memory=use_memory)
        #self.follower_actor = nn.Linear(512, action_space.n)
        #self.follower_critic = nn.Linear(512, 1)
        self.follower_actor = ImpossiblyGoodDecoderHead(action_space.n)
        self.follower_critic = ImpossiblyGoodDecoderHead(1)
    
    def get_follower(self):
        return FollowerWrapper(self)
    
    follower = property(get_follower)
    
    def get_explorer(self):
        return self
    
    explorer = property(get_explorer)
    
class FollowerWrapper(nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
        self.use_memory = other.use_memory
        self.recurrent = other.recurrent
        if self.use_memory:
            self.memory_size = other.memory_size
    
    def forward(self, obs, memory=None):
        if self.use_memory:
            if self.other.mode == 'attention':
                x, memory = self.other.encoder(
                    obs.image, obs.step, memory=memory)
            else:
                x, memory = self.other.encoder(obs.image, memory=memory)
        else:
            x = self.other.encoder(obs.image)
        
        value = self.other.follower_critic(x).reshape(-1)
        a = self.other.follower_actor(x)
        dist = Categorical(logits=F.log_softmax(a, dim=1))
        
        if self.other.use_memory:
            return dist, value, memory
        else:
            return dist, value
