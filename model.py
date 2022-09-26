import torch
from torch.nn import (
    Module,
    Sequential,
    Embedding,
    Linear,
    Conv2d,
    MaxPool2d,
    GRU,
    LSTMCell,
    ReLU,
    Tanh,
    Flatten,
)
from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
NUM_OBJECTS = max(OBJECT_TO_IDX.values())+1
NUM_COLORS = max(COLOR_TO_IDX.values())+1

from torch_ac.model import ACModel, RecurrentACModel

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ImpossiblyGoodEmbeddingEncoder(Module):
    def __init__(self, h, w, embedding_channels=16):
        super().__init__()
        
        # store the output shape
        self.out_channels = h*w*embedding_channels
        
        # image object
        self.image_object_embedding = Embedding(
            NUM_OBJECTS, embedding_channels)
        
        # image color
        self.image_color_embedding = Embedding(
            NUM_COLORS, embedding_channels)
        
        # observed color
        self.observed_color_embedding = Embedding(
            NUM_COLORS, embedding_channels)
        
    def forward(self, obs):
        # extract inputs
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
        x = x.reshape(b, self.out_channels)
        
        # return
        return x

class ImpossiblyGoodBackbone(Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.backbone = Sequential(
            ReLU(),
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImpossiblyGoodDecoderHead(Module):
    def __init__(self, n, hidden_channels=256):
        super().__init__()
        self.seq = Sequential(
            Linear(hidden_channels, hidden_channels),
            Tanh(),
            Linear(hidden_channels, n),
        )
    
    def forward(self, x):
        return self.seq(x)

class ImpossiblyGoodDistributionWrapper(Module):
    def forward(self, x):
        return Categorical(logits=log_softmax(x, dim=-1))

class ImpossiblyGoodActorDecoder(Module):
    def __init__(self, num_actions, hidden_channels=256):
        super().__init__()
        self.seq = Sequential(
            ImpossiblyGoodDecoderHead(
                num_actions, hidden_channels=hidden_channels),
            ImpossiblyGoodDistributionWrapper(),
        )
    
    def forward(self, x):
        return self.seq(x)

class ImpossiblyGoodCriticDecoder(Module):
    def __init__(self, hidden_channels=256):
        super().__init__()
        self.seq = Sequential(
            ImpossiblyGoodDecoderHead(1, hidden_channels=hidden_channels),
            Flatten(0,-1),
        )
    
    def forward(self, x):
        return self.seq(x)

class ImpossiblyGoodACModel(Module):
    recurrent=False
    def __init__(self,
        h, w,
        num_actions,
        embedding_channels=16,
        hidden_channels=256
    ):
        super().__init__()
        self.encoder = ImpossiblyGoodEmbeddingEncoder(
            h, w, embedding_channels=embedding_channels)
        self.backbone = ImpossiblyGoodBackbone(
            self.encoder.out_channels, hidden_channels=hidden_channels)
        self.actor_decoder = ImpossiblyGoodActorDecoder(
            num_actions, hidden_channels=hidden_channels)
        self.critic_decoder = ImpossiblyGoodCriticDecoder(
            hidden_channels=hidden_channels)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        dist = self.actor_decoder(x)
        value = self.critic_decoder(x)
        
        return dist, value

class ImpossiblyGoodFollowerExplorerModel(Module):
    def __init__(self,
        h,
        w,
        num_actions,
        embedding_channels=16,
        hidden_channels=16,
    ):
        super().__init__()
        
        # follower
        self.follower = ImpossiblyGoodACModel(
            h,
            w,
            num_actions,
            embedding_channels=embedding_channels,
            hidden_channels=hidden_channels,
        )
        
        # explorer
        self.explorer = ImpossiblyGoodACModel(
            h,
            w,
            num_actions,
            embedding_channels=embedding_channels,
            hidden_channels=hidden_channels,
        )
    
    def forward(self, obs):
        return self.explorer(obs)

class ImpossiblyGoodFollowerExplorerSwitcherModel(Module):
    def __init__(self,
        h,
        w,
        num_actions,
        embedding_channels=16,
        hidden_channels=256,
    ):
        super().__init__()
        
        # follower
        self.follower = ImpossiblyGoodACModel(
            h,
            w,
            num_actions,
            embedding_channels=embedding_channels,
            hidden_channels=hidden_channels,
        )
        
        # encoder/backbone
        self.encoder = ImpossiblyGoodEmbeddingEncoder(
            h, w, embedding_channels=embedding_channels)
        self.backbone = ImpossiblyGoodBackbone(
            self.encoder.out_channels, hidden_channels=hidden_channels)
        
        # explorer
        self.explorer_decoder = ImpossiblyGoodActorDecoder(
            num_actions, hidden_channels=hidden_channels)
        
        # switcher
        self.switcher_decoder = ImpossiblyGoodActorDecoder(
            2, hidden_channels=hidden_channels)
        
        # critic
        self.critic_decoder = ImpossiblyGoodCriticDecoder(
            hidden_channels=hidden_channels)
    
    def forward(self, obs, return_switch=False):
        # decode the follower
        with torch.no_grad():
            follower_dist, follower_value = self.follower(obs)
        
        # encdoer/backbone
        x = self.encoder(obs)
        x = self.backbone(x)
        
        # compute explorer distribution
        explorer_dist = self.explorer_decoder(x)
        
        # compute switcher distribution
        switcher_dist = self.switcher_decoder(x)
        
        # compute value
        value = self.critic_decoder(x)
        
        combined_prob = (
            follower_dist.probs * switcher_dist.probs[:,[0]] +
            explorer_dist.probs * switcher_dist.probs[:,[1]]
        )
        combined_dist = Categorical(probs=combined_prob)
        
        if return_switch:
            return combined_dist, value, switcher_dist.probs
        else:
            return combined_dist, value

class ImpossiblyGoodACPolicy(Module, ACModel):
    recurrent = False
    use_memory = False
    def __init__(self,
        obs_space, act_space, embedding_channels=16, hidden_channels=256):
        super().__init__()
        
        h, w = obs_space['image'][:2]
        num_actions = act_space.n
        self.model = ImpossiblyGoodACModel(
            h, w, num_actions, embedding_channels, hidden_channels)
        
        # initialize
        self.apply(init_params)
    
    def forward(self, obs):
        return self.model(obs)

#class ImpossiblyGoodFollowerExplorerPolicy(Module, ACModel):
#    recurrent = False
#    def __init__(self, obs_space, act_space):
#        super().__init__()
#        
#        h, w = obs_space['image'][:2]
#        num_actions = act_space.n
#        self.follower = ImpossiblyGoodACModel(
#            h, w, num_actions, embedding_channels, hidden_channels)
#        self.explorer = ImpossiblyGoodACModel(
#            h, w, num_actions, embedding_channels, hidden_channels)
#    
#    def forward(self, obs, memory=None):
#        f_dist, f_value = self.follower(obs)
#        e_dist, e_value = self.explorer(obs)
#        
#        return f_dist, f_value, e_dist, e_value

class ImpossiblyGoodFollowerExplorerPolicy(Module, RecurrentACModel):
    recurrent = False
    use_memory = False
    def __init__(self,
        obs_space,
        act_space,
        embedding_channels=16,
        hidden_channels=256
    ):
        super().__init__()
        
        h, w = obs_space['image'][:2]
        num_actions = act_space.n
        self.model = ImpossiblyGoodFollowerExplorerModel(
            h, w, num_actions, embedding_channels, hidden_channels)
    
    def forward(self, obs, memory=None):
        return self.model(obs)

#class ImpossiblyGoodFollowerExplorerSwitcherPolicy(Module, RecurrentACModel):
#    recurrent = False
#    def __init__(self, obs_space, act_space):
#        super().__init__()
#        
#        h, w = obs_space['image']
#        self.follower = ImpossiblyGoodACModel(obs_space, act_space)
#        self.explorer = ImpossiblyGoodACModel(obs_space, act_space)
#        self.switcher = ImpossiblyGoodSwitcherModel(
#            obs_space, self.follower, self.explorer)

class ImpossiblyGoodFollowerExplorerSwitcherPolicy(Module, RecurrentACModel):
    recurrent = False
    use_memory = False
    def __init__(self,
        obs_space,
        act_space,
        embedding_channels=16,
        hidden_channels=256
    ):
        super().__init__()
        
        h, w = obs_space['image'][:2]
        num_actions = act_space.n
        self.model = ImpossiblyGoodFollowerExplorerSwitcherModel(
            h, w, num_actions, embedding_channels, hidden_channels)
    
    def forward(self, obs, memory=None, return_switch=False):
        return self.model(obs, return_switch=return_switch)

class VanillaACPolicy(Module, RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = Sequential(
            Conv2d(3, 16, (2, 2)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(16, 32, (2, 2)),
            ReLU(),
            Conv2d(32, 64, (2, 2)),
            ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = Sequential(
            Linear(self.embedding_size, 64),
            Tanh(),
            Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = Sequential(
            Linear(self.embedding_size, 64),
            Tanh(),
            Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size
    
    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

