from PIL import Image

import gym

from envs.zoo import register_impossibly_good_envs
register_impossibly_good_envs()

env_names = [
    #'ImpossiblyGood-ExampleTwo-7x7-v0',
    #'ImpossiblyGood-ExampleTwo-9x9-v0',
    #'ImpossiblyGood-ExampleThree-7x7-v0',
    #'ImpossiblyGood-ExampleThreeEasy-9x9-v0',
    #'ImpossiblyGood-ExampleThreeMed-9x9-v0',
    #'ImpossiblyGood-ExampleFour-9x9-v0',
    'ImpossiblyGood-ExampleOne-7x7-v0',
    'ImpossiblyGood-ExampleFive-9x9-v0',
]

for env_name in env_names:
    env = gym.make(env_name)
    env.reset()
    while env.agent_dir != 3:
        env.step(env.Actions.left)
    img = env.render(mode='rgb_array')
    Image.fromarray(img).save('%s.png'%env_name)
