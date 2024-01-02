#!/bin/bash

# Copyright (c) 2022-2023 authors
# Author    : Varun A. Kelkar
# Email     : vak2@illinois.edu 

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

lr=$1 # 1e-04
num_bits=$2 # 8
# python train_vanilla.py --data_type MNISTDataset --results_dir ../results/MNIST/vanilla --input_shape 1 28 28 --num_conv_layers 12 4 --lr 1e-05
python train_vanilla.py --data_type MNISTDataset --data_args "{'power_of_two': True}" --results_dir ../results/MNIST/vanilla --input_shape 3 32 32 --model_type Glow --model_args "{'num_flow':3, 'num_block':3, 'filter_size':512}" --lr $lr --batch_size 256 --actnorm 1 --num_bits $num_bits