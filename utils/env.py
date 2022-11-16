import gym
import gym_minigrid

from envs.env_wrappers import ProcessFrame

def make_env(env_key, seed=None):
    env = gym.make(env_key)
    if 'vizdoom' in env_key.lower():
        env = ProcessFrame(env, 84, 84)
    env.reset(seed=seed)
    return env
