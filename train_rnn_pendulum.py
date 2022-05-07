import logging
import os
import time

import gym
import numpy as np
import torch

from src.algo.pg import pathlength
from src.algo.pg import PGAgent
from src.envs.pendulum import InvPendulumLinEnv
from src.envs.pendulum import InvPendulumLinEnvPartial
from src.envs.pendulum import InvPendulumLinEnvPartialNorm


def train(
    exp_name,
    n_iter,
    gamma,
    min_timesteps_per_batch,
    step_num,
    max_path_length,
    learning_rate,
    reward_to_go,
    animate,
    logdir,
    normalize_advantages,
    nn_baseline,
    seed,
    n_layers,
    size,
    states_size,
    rnn_bias,
    rnn_test_nostates,
    factor
):
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        handlers=[
            logging.FileHandler(logdir + "train.log"),
            logging.StreamHandler()
        ]
    )

    # create environment
    if 'partial' in exp_name and 'norm' in exp_name:
        env = InvPendulumLinEnvPartialNorm(factor=factor)
    elif 'partial' in exp_name:
        env = InvPendulumLinEnvPartial(factor=factor)
    else:
        env = InvPendulumLinEnv(factor=factor)

    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # ========================================================================================#
    # Initialize Agent
    # ========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'states_size': states_size,
        'learning_rate': learning_rate,
        'rnn_bias': rnn_bias
    }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
        'step_num': step_num,
        'rnn_test_nostates': rnn_test_nostates
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = PGAgent(
        computation_graph_args,
        sample_trajectory_args,
        estimate_return_args)

    # create logging header
    msg = 'Iteration'
    msg += '\tTotalSeconds'
    msg += '\tSecondsThisIter'
    msg += '\tMeanReward'
    msg += '\tStdDevReward'
    msg += '\tMinReward'
    msg += '\tMaxReward'
    msg += '\tMeanEpLength'
    msg += '\tStdDevEpLength'
    msg += '\tMinEpLength'
    msg += '\tMaxEpLength'
    msg += '\tNumEp'
    msg += '\tTimeStepsThisIter'
    msg += '\tTotalTimeSteps'
    msg += '\tActorLoss'
    if agent.nn_baseline:
        msg += '\tCriticLoss'
    logging.info(msg)

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0
    start = time.time()
    past = start

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = torch.cat([p['observation'] for p in paths], dim=0)
        # ac_na = torch.cat([p['action'] for p in paths], dim=0)
        # dn_n = torch.cat([p['termination'] for p in paths], dim=0)
        lp_n = torch.cat([p['log_probs'] for p in paths], dim=0)
        re_n = [path["reward"] for path in paths]

        q_n, adv_n = agent.estimate_return(ob_no, re_n)

        out = agent.update_parameters(ob_no, q_n, adv_n, lp_n)
        if agent.nn_baseline:
            a_loss, c_loss = out
        else:
            a_loss = out

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        msg = f'{itr:d}'
        msg += f'\t{time.time() - start:.4f}'
        msg += f'\t{time.time() - past:.4f}'
        msg += f'\t{np.mean(returns):.4f}'
        msg += f'\t{np.std(returns):.4f}'
        msg += f'\t{np.min(returns):.1f}'
        msg += f'\t{np.max(returns):.1f}'
        msg += f'\t{np.mean(ep_lengths):.1f}'
        msg += f'\t{np.std(ep_lengths):.1f}'
        msg += f'\t{int(np.min(ep_lengths)):d}'
        msg += f'\t{int(np.max(ep_lengths)):d}'
        msg += f'\t{len(ep_lengths):d}'
        msg += f'\t{timesteps_this_batch:d}'
        msg += f'\t{total_timesteps:d}'
        msg += f'\t{a_loss:.2f}'
        if agent.nn_baseline:
            msg += f'\t{c_loss:.2f}'

        # print("Iteration", itr)
        # print("Total Time", time.time() - start)
        # print("Time This Iter", time.time() - past)
        # print("AverageReturn", np.mean(returns))
        # print("StdReturn", np.std(returns))
        # print("MaxReturn", np.max(returns))
        # print("MinReturn", np.min(returns))
        # print("EpLenMean", np.mean(ep_lengths))
        # print("EpLenStd", np.std(ep_lengths))
        # print("TimestepsThisBatch", timesteps_this_batch)
        # print("TimestepsSoFar", total_timesteps)
        # print('ActorLoss', a_loss)
        # if agent.nn_baseline:
        #     print('CriticLoss', c_loss)
        past = time.time()
        logging.info(msg)

        if itr % 10 == 0:
            agent.save(logdir)

    agent.save(logdir)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,
                        default='pendulum_partial_norm')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.98)
    parser.add_argument('--n_iter', '-n', type=int, default=500)
    parser.add_argument('--batch_size', '-b', type=int, default=6000)
    parser.add_argument('--step_num', '-k', type=int, default=20)
    parser.add_argument('--ep_len', '-ep', type=float, default=200)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--factor', type=float, default=1e-1)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna',
                        action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=16)
    parser.add_argument('--states_size', '-ss', type=int, default=16)
    parser.add_argument('--rnn_bias', '-rb', action='store_true')
    parser.add_argument('--rnn_test_nostates', '-rtns', action='store_true')
    args = parser.parse_args()
    args.env_name = 'rnn'

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    train(
        exp_name=args.exp_name,
        n_iter=args.n_iter,
        gamma=args.discount,
        min_timesteps_per_batch=args.batch_size,
        step_num=args.step_num,
        max_path_length=max_path_length,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        logdir=os.path.join(logdir, '%d' % args.seed) + '/',
        normalize_advantages=not (args.dont_normalize_advantages),
        nn_baseline=args.nn_baseline,
        seed=args.seed,
        n_layers=args.n_layers,
        size=args.size,
        states_size=args.states_size,
        rnn_bias=args.rnn_bias,
        rnn_test_nostates=args.rnn_test_nostates,
        factor=args.factor
    )


if __name__ == '__main__':
    main()
