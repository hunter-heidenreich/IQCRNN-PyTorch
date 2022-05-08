import logging
import os
import time

import gym
import numpy as np
import scipy
import torch
import wandb as wb

from src.algo.pg import pathlength
from src.algo.pg import PGAgent
from src.envs import get_env
# from src.projector.method import Projector
from src.projector.method2 import Projector
from src.projector.Q1Init import Q1_init


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
        factor,
        tilde,
        init_trunc_norm,
        verbose=True
):
    # configure logger
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        handlers=[
            logging.FileHandler(logdir + "train.log"),
            # logging.StreamHandler()
        ]
    )

    # create environment
    env = get_env(exp_name, factor)

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
        estimate_return_args,
        tilde=tilde,
        init_trunc_norm=init_trunc_norm,
    )
    wb.watch(agent.rnn, log='gradients', log_freq=1)
    # if agent.nn_baseline:
    #     wb.watch(agent.mlp, log='gradients', log_freq=1)

    if tilde:
        AG, BG, CG = env.get_system_params()

        Q1initializer = Q1_init(AG, BG, CG=CG)
        Q1 = Q1initializer.compute()
        Q10 = scipy.linalg.block_diag(Q1, np.eye(states_size - env.nx))

        projector = Projector(
            computation_graph_args['states_size'],
            computation_graph_args['size'],
            AG, BG, CG=CG,
        )

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0
    start = time.time()
    past = start

    best_mean = -np.inf
    for itr in range(n_iter):
        if itr % 100 == 0:
            agent.save(logdir + f'e{itr:03d}_')

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

        if tilde:
            totdiff, xidiff, udiff, vdiff = projector.updateRNN(agent.rnn)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        num_paths = len(paths)
        now = time.time()
        metrics = {
            'iteration': itr,
            'seconds_total': now - start,
            'seconds_iter': past - start,
            'paths': num_paths,
            'timesteps_total': total_timesteps,
            'timesteps_iter': timesteps_this_batch,
            'loss_actor': a_loss,
        }

        if agent.nn_baseline:
            metrics['loss_critic'] = c_loss

        if tilde:
            metrics['diff_total'] = totdiff
            metrics['diff_xi'] = xidiff
            metrics['diff_u'] = udiff
            metrics['diff_v'] = vdiff


        for key, vals in [
            ('reward', returns),
            ('eplen', ep_lengths)
        ]:
            metrics[key + '_mean'] = np.mean(vals)
            metrics[key + '_std'] = np.std(vals)
            metrics[key + '_se'] = metrics[key + '_std'] / np.sqrt(metrics['paths'])
            metrics[key + '_mean_u'] = metrics[key + '_mean'] + 1.96 * metrics[key + '_se']
            metrics[key + '_mean_l'] = metrics[key + '_mean'] - 1.96 * metrics[key + '_se']
            metrics[key + '_max'] = np.max(vals)
            metrics[key + '_min'] = np.min(vals)

        if itr == 0:
            msg0 = '\t'.join(sorted(list(metrics.keys())))
            logging.info(msg0)

        msg = []
        for k, v in sorted(metrics.items(), key=lambda kv: kv[0]):
            if verbose:
                print(f'{k}:\t{v}')
            msg.append(str(v))
        msg = '\t'.join(msg)

        logging.info(msg)
        wb.log(metrics)

        if metrics['reward_mean'] >= best_mean:
            best_mean = metrics['reward_mean']
            agent.save(logdir + 'best_')
            if verbose:
                print('Higher mean. Saved weights.')

        # if itr % 10 == 0:
        #     agent.save(logdir)

        past = time.time()

    # agent.save(logdir)
    agent.save(logdir + f'e{n_iter:03d}_')
