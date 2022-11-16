import time

import envs.vizdoom as vizdoom
vizdoom.register_vizdoom_envs()

from envs.env_wrappers import EnvSwitcher, ProcessFrame

import gym
from vizdoom import gym_wrapper

if __name__  == '__main__':
    env = ProcessFrame(gym.make("ImpossiblyGoodMonsterRoom-v0"), 84, 84)
    #env = gym.make('VizdoomCorridor-v0')

    # Rendering random rollouts for ten episodes
    for _ in range(10):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            #a = env.action_space.sample()
            #print('a:', a)
            if False:
                while True:
                    try:
                        a = int(input())
                    except TypeError:
                        print('Int please!')
                    finally:
                        break
            else:
                a = obs['expert']
                time.sleep(0.5)
            obs, rew, terminated, info = env.step(a)
            #print('r:', rew)
            done = terminated
            #breakpoint()
            
