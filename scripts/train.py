import argparse
import time
import datetime
import sys

import torch

import torch_ac

import tensorboardX

import utils
from utils import device
from model import (
    ImpossiblyGoodACPolicy,
    ImpossiblyGoodFollowerExplorerPolicy,
    ImpossiblyGoodFollowerExplorerSwitcherPolicy,
    VanillaACPolicy,
)
#from algos.teacher_distill import TeacherDistillAlgo
from algos.follower_explorer import FEAlgo
from algos.distill import Distill
from envs.zoo import register_impossibly_good_envs
register_impossibly_good_envs()

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo | bc | opbc | fe | fes (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--arch", type=str, default='ig')
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--follower-frames-per-proc", type=int, default=128)
parser.add_argument("--explorer-frames-per-proc", type=int, default=128)
parser.add_argument("--explorer-reward-maximizer", type=str, default='ppo',
                    help="rl algorithm to use for the explorer policy")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--expert-matching-reward-pos", type=float, default=0.1)
parser.add_argument("--expert-matching-reward-neg", type=float, default=-0.1)
parser.add_argument("--policy-loss-coef", type=float, default=1.0)
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--expert-loss-coef", type=float, default=1.0)
parser.add_argument("--entropy-loss-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--adam-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--reward-shaping", type=str, default='none')
#parser.add_argument("--recurrence", type=int, default=1,
#                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
#parser.add_argument("--text", action="store_true", default=False,
#                    help="add a GRU to the model to handle text input")

if __name__ == '__main__':
    args = parser.parse_args()

    #args.mem = args.recurrence > 1
    args.mem = False
    args.recurrence = 1

    # set the run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)
    
    # load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # set seed for all randomness sources
    utils.seed(args.seed)

    # set device
    txt_logger.info(f"Device: {device}\n")

    # load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    if args.algo in ('fe', 'fes'):
        explorer_envs = []
        for i in range(args.procs):
            explorer_envs.append(
                utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # load observations preprocessor
    if args.arch == 'vanilla':
        obs_space, preprocess_obss = utils.get_obss_preprocessor(
            envs[0].observation_space, image_dtype=torch.float)
    else:
        obs_space, preprocess_obss = utils.get_obss_preprocessor(
            envs[0].observation_space, image_dtype=torch.long)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # load model
    if args.algo == 'fe':
        acmodel = ImpossiblyGoodFollowerExplorerPolicy(
            obs_space, envs[0].action_space)
    elif args.algo == 'fes':
        acmodel = ImpossiblyGoodFollowerExplorerSwitcherPolicy(
            obs_space, envs[0].action_space)
    else:
        if args.arch == 'ig':
            acmodel = ImpossiblyGoodACPolicy(obs_space, envs[0].action_space)
        elif args.arch == 'vanilla':
            acmodel = VanillaACPolicy(obs_space, envs[0].action_space)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))
    
    # setup reward shaping
    neg_bias = 6
    shaping_scale = 0.1
    def expert_reshaping(pre_obs, post_obs, action, reward, done, device):
        a = torch.tensor(action, dtype=torch.long, device=device)
        e = preprocess_obss(pre_obs, device=device).expert
        matching = ((a == e).float() * (neg_bias+1) - neg_bias) * shaping_scale
        r = torch.tensor(reward, dtype=torch.float, device=device)
        reshaped_reward = r + matching
        return reshaped_reward
    
    if args.reward_shaping == 'none':
        reward_shaping_fn = None
    elif args.reward_shaping == 'expert':
        reward_shaping_fn = expert_reshaping
    else:
        raise ValueError('Unknown reward shaping: %s'%args.reward_shaping)
    
    # Load algo
    
    #if args.algo == "old_a2c":
    #    algo = torch_ac.A2CAlgo(
    #        envs,
    #        acmodel,
    #        device,
    #        args.frames_per_proc,
    #        args.discount,
    #        args.lr,
    #        args.gae_lambda,
    #        args.entropy_loss_coef,
    #        args.value_loss_coef,
    #        args.max_grad_norm,
    #        args.recurrence,
    #        args.optim_alpha,
    #        args.adam_eps,
    #        preprocess_obss,
    #    )
    #elif args.algo == "old_ppo":
    #    algo = torch_ac.PPOAlgo(
    #        envs,
    #        acmodel,
    #        device,
    #        args.frames_per_proc,
    #        args.discount,
    #        args.lr,
    #        args.gae_lambda,
    #        args.entropy_loss_coef,
    #        args.value_loss_coef,
    #        args.max_grad_norm,
    #        args.recurrence,
    #        args.adam_eps,
    #        args.clip_eps,
    #        args.epochs,
    #        args.batch_size,
    #        preprocess_obss,
    #        reshape_reward=reward_shaping_fn,
    #    )
    default_settings = {
        'device':device,
        'num_frames_per_proc' : args.frames_per_proc,
        'discount' : args.discount,
        'lr' : args.lr,
        'gae_lambda' : args.gae_lambda,
        'expert_matching_reward_pos' : args.expert_matching_reward_pos,
        'expert_matching_reward_neg' : args.expert_matching_reward_neg,
        'policy_loss_coef' : args.policy_loss_coef,
        'value_loss_coef' : args.value_loss_coef,
        'expert_loss_coef' : args.expert_loss_coef,
        'entropy_loss_coef' : args.entropy_loss_coef,
        'max_grad_norm' : args.max_grad_norm,
        'recurrence' : args.recurrence,
        'adam_eps' : args.adam_eps,
        'clip_eps' : args.clip_eps,
        'epochs' : args.epochs,
        'batch_size' : args.batch_size,
        'preprocess_obss' : preprocess_obss,
    }
    if args.algo in ('fe', 'fes'):
        algo = FEAlgo(
            envs,
            explorer_envs,
            acmodel,
            device=device,
            follower_frames_per_proc=args.follower_frames_per_proc,
            explorer_frames_per_proc=args.explorer_frames_per_proc,
            explorer_reward_maximizer=args.explorer_reward_maximizer,
            discount=args.discount,
            lr=args.lr,
            gae_lambda=args.gae_lambda,
            expert_matching_reward=args.expert_matching_reward,
            policy_loss_coef=args.policy_loss_coef,
            value_loss_coef=args.value_loss_coef,
            expert_loss_coef=args.expert_loss_coef,
            entropy_loss_coef=args.entropy_loss_coef,
            max_grad_norm=args.max_grad_norm,
            recurrence=args.recurrence,
            adam_eps=args.adam_eps,
            clip_eps=args.clip_eps,
            follower_epochs=args.epochs,
            explorer_epochs=args.epochs,
            batch_size=args.batch_size,
            preprocess_obss=preprocess_obss,
        )
    elif args.algo == 'ppo':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            plus_R=True,
            **default_settings,
        )
    elif args.algo == 'a2c':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='a2c',
            plus_R=True,
            **default_settings,
        )
    elif args.algo == 'teacher_distill':
        # online behavior cloning
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='zero',
            l_term='cross_entropy',
            r_term='zero',
            plus_R=False,
            on_policy=False,
            **default_settings,
        )
    elif args.algo == 'on_policy_distill':
        # dagger lite
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='zero',
            value_loss_model='zero',
            l_term='cross_entropy',
            r_term='zero',
            plus_R=False,
            on_policy=True,
            value_model=None,
            explorer_model=None,
            **default_settings,
        )
    elif args.algo == 'on_policy_distill_plus_r':
        # dagger lite + reward
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='cross_entropy',
            r_term='zero',
            plus_R=True,
            on_policy=True,
            value_model=None,
            explorer_model=None,
            **default_settings,
        )
    elif args.algo == 'entropy_regularized':
        raise Exception('no log_p')
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='zero',
            r_term='log_p',
            plus_R=False,
            on_policy=True,
            skip_immediate_reward=False,
            **default_settings,
        )
    elif args.algo == 'entropy_regularized_plus_r':
        raise Exception('no log_p')
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='zero',
            r_term='log_p',
            plus_R=True,
            on_policy=True,
            skip_immediate_reward=False,
            **default_settings,
        )
    elif args.algo == 'expert_matching_reward':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='zero',
            r_term='expert_matching_reward',
            plus_R=False,
            on_policy=True,
            skip_immediate_reward=False,
            **default_settings,
        )
    elif args.algo == 'expert_matching_reward_plus_r':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='zero',
            r_term='expert_matching_reward',
            plus_R=True,
            on_policy=True,
            skip_immediate_reward=False,
            **default_settings,
        )
    elif args.algo == 'n_distill':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='cross_entropy',
            r_term='cross_entropy',
            plus_R=False,
            on_policy=True,
            skip_immediate_reward=True,
            **default_settings,
        )
    elif args.algo == 'n_distill_plus_r':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='cross_entropy',
            r_term='cross_entropy',
            plus_R=True,
            on_policy=True,
            skip_immediate_reward=True,
            **default_settings,
        )
            
    elif args.algo == 'exp_entropy_regularized':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='reverse_cross_entropy',
            r_term='log_p', #'expert_matching_reward', # was 'log_p'
            plus_R=False,
            on_policy=True,
            skip_immediate_reward=True,
            **default_settings,
        )
    
    elif args.algo == 'exp_entropy_regularized_plus_r':
        algo = Distill(
            envs,
            acmodel,
            reward_maximizer='ppo',
            value_loss_model='ppo',
            l_term='reverse_cross_entropy',
            r_term='log_p', #'expert_matching_reward', # was 'log_p',
            plus_R=True,
            on_policy=True,
            skip_immediate_reward=True,
            **default_settings,
        )
    
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        if args.algo in ('fe', 'fes'):
            algo.follower_algo.optimizer.load_state_dict(
                status['optimizer_state']['follower']
            )
            algo.explorer_algo.optimizer.load_state_dict(
                status['optimizer_state']['explorer']
            )
        else:
            algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        if args.algo in ('fe', 'fes'):
            num_frames += logs["follower_num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(
                logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(
                logs["num_frames_per_episode"])
            
            if args.algo in ('fe', 'fes'):
                follower_return_per_episode = utils.synthesize(
                    logs['follower_return_per_episode'])
                follower_num_frames_per_episode = utils.synthesize(
                    logs['follower_num_frames_per_episode'])
            
            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += [
                "num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += [
                "entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [
                logs["entropy"], logs["value"], logs["policy_loss"],
                logs["value_loss"], logs["grad_norm"]
            ]
            
            if args.algo in ('fe', 'fes'):
                header += [
                    "follower_return_" + key
                    for key in follower_return_per_episode.keys()]
                data += follower_return_per_episode.values()
                header += [
                    "follower_num_frames_" + key
                    for key in follower_num_frames_per_episode.keys()]
                data += follower_num_frames_per_episode.values()
                header += [
                    "follower_entropy",
                    "follower_value",
                    "follower_policy_loss",
                    "follower_value_loss",
                    "follower_grad_norm",
                ]
                data += [
                    logs['follower_entropy'],
                    logs['follower_value'],
                    logs['follower_policy_loss'],
                    logs['follower_value_loss'],
                    logs['follower_grad_norm'],
                ]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | f_rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | f_F:μσmM {:.1f} {:.1f} {} {} | f_H {:.3f} | f_V {:.3f} | f_pL {:.3f} | f_vL {:.3f} | f_∇ {:.3f}".format(*data))
            
            else:
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            if args.algo in ('fe', 'fes'):
                optimizer_state = {
                    'follower' : algo.follower_algo.optimizer.state_dict(),
                    'explorer' : algo.explorer_algo.optimizer.state_dict(),
                }
            else:
                optimizer_state = algo.optimizer.state_dict()
            status = {
                "num_frames": num_frames,
                "update": update,
                "model_state": acmodel.state_dict(),
                "optimizer_state": optimizer_state,
            }
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
