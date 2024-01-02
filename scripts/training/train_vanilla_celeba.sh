#!/bin/bash

# Copyright (c) 2022-2023 authors
# Author    : Varun A. Kelkar
# Email     : vak2@illinois.edu 

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Following hyperparameter configurations are for example only. The optimal parameter configurations are provided in `paper_configs.txt`.

cd ..
cd ..
cd src

lr=1e-04
num_bits=8
gpus=1
batch_size=128

results_dir=../results/celeba/vanilla
echo $results_dir

# python train_vanilla.py --data_type MNISTDataset --results_dir ../results/MNIST/vanilla --input_shape 1 28 28 --num_conv_layers 12 4 --lr 1e-05
python train_vanilla.py --gpus $gpus --num_kimgs 20000 --data_type CelebAHQDataset --results_dir $results_dir --input_shape 3 64 64 --model_type Glow --model_args "{'num_flow':32, 'num_block':4, 'filter_size':512}" --lr $lr --batch_size $batch_size --actnorm 1 --num_bits $num_bits

