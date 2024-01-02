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

# python train_ambient.py --data_type MNISTDataset --results_dir ../results/MNIST/ambient --lr 1e-05 --reg_parameter 0.1 --num_iters_post 1000 --num_iters_main 10000

blur_sigma=1.5
other=$2
python train_ambient_mnist.py --data_type MNISTDataset --data_args "{'power_of_two': True}" --results_dir ../results/MNIST/ambient --input_shape 3 32 32 --main_model_type Glow --main_model_args "{'num_flow':3, 'num_block':3, 'filter_size':512}" --post_model_type CondConvINN --post_model_args "{'num_conv_layers':[4, 12], 'num_fc_layers':[4]}" --reg_parameter 0.1 --lr_post 1e-03 --lr_main 1e-03 --batch_size 32 --main_actnorm 1 --num_bits 0 --num_iters_post 1000 --num_iters_main 10000 --num_iters 50000 --degradation_type GaussianBlurNoise --degradation_args "{'kernel_sigma':$blur_sigma, 'mean':0., 'std':0.2}" $other