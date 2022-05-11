#!/usr/bin/env bash

python train_rnn_pendubot.py -rtg --nn_baseline -itn
python train_rnn_pendubot_tilde.py -rtg --nn_baseline -itn

python train_rnn_pendubot.py -rtg --nn_baseline
python train_rnn_pendubot_tilde.py -rtg --nn_baseline
