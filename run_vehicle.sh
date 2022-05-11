#!/usr/bin/env bash

python train_rnn_vehicle.py -rtg --nn_baseline -itn
python train_rnn_vehicle_tilde.py -rtg --nn_baseline -itn

python train_rnn_vehicle.py -rtg --nn_baseline
python train_rnn_vehicle_tilde.py -rtg --nn_baseline
