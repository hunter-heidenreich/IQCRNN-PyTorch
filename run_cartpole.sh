#!/usr/bin/env bash

python train_rnn_cartpole.py -rtg --nn_baseline -itn
python train_rnn_cartpole_tilde.py -rtg --nn_baseline -itn

#python train_rnn_cartpole.py -rtg --nn_baseline
#python train_rnn_cartpole_tilde.py -rtg --nn_baseline
