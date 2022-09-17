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

class ImpossiblyGoodFollowerExplorerModel(Module, ACModel):
    recurrent = False
    def __init__(self, obs_space, act_space):
        super().__init__()
        
        self.follower = ImpossiblyGoodACModel(obs_space, act_space)
        self.explorer = ImpossiblyGoodACModel(obs_space, act_space)
    
    def forward(self, obs):
        #print('FOllOWER')
        f_dist, f_value = self.follower(obs)
        #print('EXPLORER')
        e_dist, e_value = self.explorer(obs)
        
        return f_dist, f_value, e_dist, e_value

class ImpossiblyGoodACModel(Module, ACModel):
    recurrent = False
    def __init__(self, obs_space, act_space):
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
            Linear(hidden_channels, act_space.n),
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
        
        #print(value)
        
        # return
        return distribution, value

class VanillaACModel(Module, RecurrentACModel):
    recurrent = False
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
