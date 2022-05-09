import os
import time

import wandb as wb

from src.train import train


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
    parser.add_argument('--init_trunc_norm', '-itn', action='store_true')
    args = parser.parse_args()
    args.env_name = 'rnn'
    args.tilde = False

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    args.logdir = logdir

    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    disp_name = f'Pendulum(proj={args.tilde}, init={"trunc. normal" if args.init_trunc_normal else "uniform"})'
    wb.init(config=vars(args), project='IQCRNN', name=disp_name)

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
        factor=args.factor,
        tilde=False,
        init_trunc_norm=args.init_trunc_norm,
        verbose=True
    )


if __name__ == '__main__':
    main()
