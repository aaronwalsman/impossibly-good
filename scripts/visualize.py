import time
import argparse
import numpy

import utils
from utils import device

from envs.zoo import register_impossibly_good_envs
register_impossibly_good_envs()

from envs.vizdoom import register_vizdoom_envs
register_vizdoom_envs()

#from algos.follower_explorer import make_reshaper

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument('--slow', type=float, default=0.)
parser.add_argument('--breakpoint', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument('--use-follower', action='store_true')
#parser.add_argument("--fe-rollout-mode", type=str, default='max_value',
#                    help='follower/explorer/max_value, default=max_value')
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
#parser.add_argument("--text", action="store_true", default=False,
#                    help="add a GRU to the model")
parser.add_argument('--checkpoint-index', type=int, default=None)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

is_vizdoom = 'vizdoom' in args.env.lower()

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    argmax=args.argmax,
    use_follower=args.use_follower,
    verbose=args.verbose,
    use_memory=args.memory,
    vizdoom=is_vizdoom,
    checkpoint_index=args.checkpoint_index,
)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

#if agent.arch == 'fe':
#    reshaper = make_reshaper(
#        agent.preprocess_obss, agent.acmodel.model.follower, verbose=True)

override_action = None

for episode in range(args.episodes):
    obs = env.reset()
    done = True
    
    #if args.memory:
    #    memory = torch.zeros(
    #        (1, agent.acmodel.memory_size), device=device)
    
    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))
        
        #pre_obs = obs
        #if args.memory:
        #    action = agent.get_action(obs, memory)
        #else:
        #    action = agent.get_action(obs)
        
        if args.verbose:
            print('============================')
            print('Step: %i/%i'%(obs['step'], env.max_steps))
        
        if args.memory:
            m = agent.memories
        action = agent.get_action(obs) # memory is handled by the agent
        
        if args.verbose:
            if 'fe' in agent.arch:
                proc_obs = agent.preprocess_obss([obs], device=device)
                if is_vizdoom:
                    follower_model = agent.acmodel.follower
                else:
                    follower_model = agent.acmodel.model.follower
                
                if args.memory:
                    follower_dist, follower_value, *_ = follower_model(
                        proc_obs, memory=m)
                else:
                    follower_dist, follower_value = follower_model(proc_obs)
                print('Follower Distribution:')
                probs = follower_dist.probs.detach().cpu().numpy()[0]
                for i, p in enumerate(probs):
                    print('    p(%i): %.04f'%(i,p))
                
                print('Follower Value: %.04f'%follower_value.item())
                
                if is_vizdoom:
                    if args.memory:
                        _, explorer_value, *_ = agent.acmodel(
                            proc_obs, memory=m)
                    else:
                        _, explorer_value, *_ = agent.acmodel(
                            proc_obs)
                    print('Explorer Value: %.04f'%explorer_value.item())
                else:
                    _, explorer_value, switch = agent.acmodel(
                        proc_obs, return_switch=True)
                    print('Explorer Value: %.04f'%explorer_value.item())
                    print('P(follower): %.04f'%switch[0,0].item())
            #if agent.arch == 'fe':
            #    reshaped_reward = reshaper(
            #        [pre_obs], [obs], [action], [reward], [done],
            #        device=device)
            #    print('Reshaped Reward: %.04f'%reshaped_reward.item())
            if is_vizdoom:
                print('Action: %i'%action)
                print('Expert: %i'%obs['expert'])
            else:
                print('Action: %s'%env.Actions(action), int(action))
                print('Expert: %s'%env.Actions(obs['expert']),
                    int(obs['expert'])
                )
        if args.breakpoint:
            command = input()
            if command:
                if command == 'breakpoint':
                    breakpoint()
                try:
                    if is_vizdoom:
                        override_action = int(command)
                        if override_action > 3:
                            raise ValueError('invalid action')
                    else:
                        override_action = getattr(env.Actions, command)
                    print('OVERRIDE ACTION: %s'%override_action)
                except (AttributeError, ValueError):
                    print('INVALID OVERRIDE ACTION: %s'%command)
                    override_action = None
            else:
                override_action = None
        
        if override_action is not None:
            action = override_action
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)
        
        if args.verbose:
            print('Reward: %.04f'%reward)
        
        if args.slow:
            time.sleep(args.slow)

        if done:
            break
        if hasattr(env, 'window') and env.window.closed:
            break
    
    if hasattr(env, 'window') and env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
