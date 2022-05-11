#!/usr/bin/env bash

python train_rnn_pendulum.py -rtg --nn_baseline -itn
python train_rnn_pendulum_tilde.py -rtg --nn_baseline -itn

python train_rnn_pendulum.py -rtg --nn_baseline
python train_rnn_pendulum_tilde.py -rtg --nn_baseline
