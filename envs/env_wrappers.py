import random
import math

import numpy

import gym

from PIL import Image

#import utils

class EnvSwitcher(gym.Wrapper):
    def __init__(self, enva, envb, pa=0.5):
        super().__init__(enva)
        self.enva = enva
        self.envb = envb
        self.pa = pa
    
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        a = self.enva.reset(seed=seed)
        b = self.envb.reset(seed=seed)
        
        r = random.random()
        if r < self.pa:
            self.env = self.enva
            return a
        else:
            self.env = self.envb
            return b
    
    #def step(self, action):
    #    return self.env.step(action)

class WaypointerVertex:
    def __init__(self, p, target=None, use=False, bad_door=False):
        self.p = numpy.array(p)
        self.target = target
        self.use = use
        self.use_pushed = False
        self.bad_door = bad_door
    
    def __lt__(self, b):
        return False

class Waypointer(gym.Wrapper):
    def __init__(self, env, vertices, angle_tolerance=math.radians(15)):
        super().__init__(env)
        
        max_ticks = env.game.get_episode_timeout()
        frame_skip = env.frame_skip
        start_time = env.game.get_episode_start_time()
        self.max_steps = math.ceil((max_ticks - start_time) / frame_skip)
        
        self.observation_space = gym.spaces.Dict({
            **env.observation_space,
            'expert':env.action_space,
            'step':gym.spaces.Discrete(self.max_steps)
        })
        
        self.vertices = vertices
        self.angle_tolerance = angle_tolerance
    
    def reset(self, seed=None):
        self.step_count = 0
        self.recent_actions = []
        self.recent_positions = []
        self.requested_push = False
        for vertex in self.vertices:
            vertex.use_pushed = False
        o = self.env.reset(seed=seed)
        if isinstance(o, tuple):
            o = o[0]
        return self.observation(o)
    
    def step(self, action):
        self.step_count += 1
        #o, r, t, i = self.env.step(action)
        orti = self.env.step(action)
        o, r, t, i = orti[0], orti[1], orti[2], orti[-1]
        
        # compute expert action
        o = self.observation(o, action)
        
        # reshape rewards
        EARLY_STOPPING_REWARD = -0.05
        STEP_REWARD = -.001 #-0.00025 #-0.001
        TIMEOUT_REWARD = 0
        FOUND_EXIT_REWARD = 2 #1
        EXPLORATION_BONUS = 0.0005 #0.000125 #0.0005
        DEAD_REWARD = -2 # -1
        
        # early stopping
        current_position = numpy.array(o['gamevariables'][:2])
        hits_taken = o['gamevariables'][3]
        if hits_taken:
            print('HIT!')
        self.recent_actions.append(action)
        early_termination_sequences = [
            [1,2,1],
            [2,1,2],
            [3,3],
            'travel',
        ]
        for seq in early_termination_sequences:
            if seq == 'travel':
                # make sure moving actually does something
                if (action == 0 and
                    len(self.recent_positions) >= 1 and
                    numpy.linalg.norm(
                        self.recent_positions[-1] - current_position) < 5. and
                    not (self.min_v.use or self.min_v.bad_door)
                ):
                    #print('DIDN\'T TRAVEL ENOUGH')
                    if r < -50 or hits_taken:
                        r = DEAD_REWARD
                    else:
                        r = EARLY_STOPPING_REWARD
                    t = True
                    break
            elif self.recent_actions[-len(seq):] == seq:
                if seq == [3,3] and (self.min_v.use or self.min_v.bad_door):
                    continue
                
                if r < -50 or hits_taken:
                    r = DEAD_REWARD
                else:
                    r = EARLY_STOPPING_REWARD
                t = True
                break
        else:
            # if dead r=-1
            if r < -50 or hits_taken: # if dead, r = -1
                #print('DED')
                r = DEAD_REWARD
                t = True
            elif t:
                # if the exit was not found, reward = 0
                if self.step_count > self.max_steps-4:
                    #print('TIMEOUT')
                    r = TIMEOUT_REWARD
                # if the reward was found, r = 1
                else:
                    #print('SAFETY!')
                    #r = 1. - 0.9 * (self.step_count / self.max_steps)
                    r = FOUND_EXIT_REWARD
            else: # if nothing happens, reward = STEP_REWARD
                r = STEP_REWARD
            
                # exploration bonus
                if self.recent_positions:
                    offsets = (
                        current_position.reshape(1,2) - self.recent_positions)
                    distances = numpy.linalg.norm(offsets, axis=1)
                    min_distance = numpy.min(distances)
                    r += min_distance * EXPLORATION_BONUS
                
                #print('NOTHING HAPPENED')
        
        self.recent_positions.append(current_position)
        
        return o, r, t, i
    
    def observation(self, o, a=None):
        # compute the expert action
        p = numpy.array([o['gamevariables'][0], o['gamevariables'][1]])
        angle = math.radians(o['gamevariables'][2])
        
        # get the closest vertex
        min_d, self.min_v = min([
            (numpy.linalg.norm(v.p - p), v)
            for v in self.vertices
        ])
        
        # get the closest vertices' target
        target_vertex = self.min_v.target
        target_p = target_vertex.p
        
        to_target = target_p - p
        to_target = to_target / numpy.linalg.norm(to_target)
        
        current_direction = numpy.array([math.cos(angle), math.sin(angle)])
        
        # if we told the agent to 'use' last time and it just did, record it
        if self.requested_push and a == 3:
            self.min_v.use_pushed = True
        
        self.requested_push = False
        # if you are pointing in the right direction...
        if numpy.dot(to_target, current_direction) > 0.9:
            # if this vertex needs 'use' and it hasn't been done already...
            if self.min_v.use and not self.min_v.use_pushed:
                '''
                # if 'use' was just pushed then walk forward and mark 'use' done
                if a == 3:
                    o['expert'] = 0
                    self.min_v.use_pushed = True
                
                # if 'use' was not just pushed, push 'use'
                else:
                    o['expert'] = 3
                '''
                o['expert'] = 3
                self.requested_push = True
            else:
                o['expert'] = 0
        
        else:
            if (to_target[0] * current_direction[1] -
                current_direction[0] * to_target[1]) < 0.:
                o['expert'] = 1
            else:
                o['expert'] = 2
        
        o['step'] = self.step_count
        
        return o

#class DeferredWrapper:
#    def __init__(self, *args, **kwargs):
#        self.args = args
#        self.kwargs = kwargs
#        self.initialized = False
#    
#    def reset(self, *args, **kwargs):
#        if not self.initialized:
#            self.env = utils.make_env(*self.args, **self.kwargs)
#            self.initialized = True
#        
#        return self.env.reset(*args, **kwargs)
#    
#    def __getattr__(self, attr):
#        if attr == 'env':
#            raise AttributeError
#        return getattr(self.env, attr)

class ProcessFrame(gym.Wrapper):
    def __init__(self, env, h, w):
        super().__init__(env)
        self.h = h
        self.w = w
        
        #max_ticks = env.game.get_episode_timeout()
        #frame_skip = env.frame_skip
        #start_time = env.game.get_episode_start_time()
        #self.max_steps = math.ceil((max_ticks - start_time) / frame_skip)
        
        observation_space = {
            'image' : gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,h,w), dtype=numpy.float32),
            'step' : gym.spaces.Discrete(self.max_steps),
        }
        if 'expert' in env.observation_space:
            observation_space['expert'] = env.observation_space['expert']
        self.observation_space = gym.spaces.Dict(observation_space)
    
    def step(self, action):
        o, r, t, i = self.env.step(action)
        #current_position = numpy.array(o['gamevariables'][:2])
        
        '''
        if t:
            if self.step_count == self.max_steps:
                r += 0
            else:
                r += 1
        '''
        
        
        '''
        # change reward structure
        if r < -50: # if dead, reward = 0
            r = -1
        elif t:
            # if the exit was not found, reward = 0
            if self.step_count == self.max_steps:
                r = 0
            else:
                #r = 1. - 0.9 * (self.step_count / self.max_steps)
                r = 1
        else: # if nothing happens, reward = 0
            r = 0
        
        r -= 0.001
        '''
        #elif t: # if exit not found
        #    r = 1. - 0.9 * (self.step_count / self.max_steps)
        #else:
        #    r = 0
        
        '''
        # exploration bonus
        if self.recent_positions:
            offsets = current_position.reshape(1,2) - self.recent_positions
            distances = numpy.linalg.norm(offsets, axis=1)
            min_distance = numpy.min(distances)
            r += min_distance * 0.0005
        self.recent_positions.append(current_position)
        
        self.step_count += 1
        
        # early termination
        self.recent_actions.append(action)
        early_termination_sequences = [
            [1,2,1],
            [2,1,2],
            [3,3],
        ]
        for seq in early_termination_sequences:
            if self.recent_actions[-len(seq):] == seq:
                r = -0.05 # = 0
                t = True
                break
        '''
        
        '''
        # this causes an early termination if you follow directions right up to
        # the door because the door gets in your way sometimes
        if (action == 0 and
            len(self.recent_positions) >= 2 and
            numpy.linalg.norm(
                self.recent_positions[-1] - self.recent_positions[-2]) < 5.
        ):
            r = -0.05
            t = True
        '''
        
        #    print('Travelled:', numpy.linalg.norm(
        #        self.recent_positions[-1] - self.recent_positions[-2]))
        
        #if (self.recent_actions[-2:] == [2,1] and action == 2 or
        #    self.recent_actions[-2:] == [1,2] and action == 1
        #):
        #    r = 0
        #    t = True
        
        
        #print(o['gamevariables'])
        return self.observation(o), r, t, i
    
    def reset(self, seed=None):
        self.step_count = 0
        o = self.env.reset(seed=seed)
        return self.observation(o)
    
    def observation(self, obs):
        if 'rgb' in obs:
            rgb = Image.fromarray(obs['rgb'])
        elif 'screen' in obs:
            rgb = Image.fromarray(obs['screen'])
        rgb = rgb.resize((self.h, self.w), Image.BILINEAR).convert('L')
        rgb = numpy.array(rgb).astype(numpy.float32) / 255.
        rgb = rgb.reshape((1, self.h, self.w))
        new_obs = {'image':rgb, 'step':self.step_count}
        if 'expert' in obs:
            new_obs['expert'] = obs['expert']
        new_obs['step'] = obs['step']
        
        return new_obs
