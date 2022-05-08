#!/usr/bin/env bash

python train_rnn_powergrid.py -rtg --nn_baseline -itn
python train_rnn_powergrid_tilde.py -rtg --nn_baseline -itn

python train_rnn_powergrid.py -rtg --nn_baseline
python train_rnn_powergrid_tilde.py -rtg --nn_baseline
