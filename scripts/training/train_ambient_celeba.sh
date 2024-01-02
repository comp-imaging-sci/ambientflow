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

# export NCCL_IB_DISABLE=1

num_gpus=1
num_bits=0 # 0
lr=1e-04
lr_post=2e-05
batch_size=92
num_z=2
reg_parameter=2.3
importance_weighting=1
noise_std=0.2
threshold_weight=0.133
sparsity_weight=0.0097
other=${1} # --port 12357

post_model_args="{'num_conv_layers':[2, 4, 4], 'num_fc_layers':[4], 'cond_layer_thicknesses':[32, 64, 64, 256]}"

# Gaussian noise degradation
degradation_type=GaussianNoise
degradation_args="{'mean': 0.0, 'std': 0.2}"

# Blur plus noise degradation
degradation_type=GaussianBlurNoise
degradation_args="{'kernel_sigma': 1.5, 'mean': 0.0, 'std': 0.2}"

results_dir=../results/celeba/ambient/${degradation_type}/
mkdir -p $results_dir

echo "Training for 4000 kimgs without sparsity constraint ..."

python -u train_ambient.py --num_gpus $num_gpus --data_type CelebAHQDataset --data_args "{'on_the_fly': False, 'augmentation': 'vf'}" --results_dir $results_dir --input_shape 3 64 64 --main_model_type Glow --main_model_args "{'num_flow':32, 'num_block':4, 'filter_size':512}" --post_model_type CondConvINN2 --post_model_args "$post_model_args" --sparsifier_type GradientTransform --sparsifier_args "{}" --reg_parameter $reg_parameter --sparsity_weight 0 --threshold_weight 0 --num_z $num_z --lr $lr --lr_post $lr_post --batch_size $batch_size --main_actnorm 1 --post_actnorm 1 --num_bits $num_bits --num_kimgs_post 0 --num_kimgs_main 0 --num_kimgs 4000 --degradation_type $degradation_type --degradation_args "$degradation_args" --schedule_sparsity_weight 0 --benchmark_cudnn 1 --importance_weighting $importance_weighting $other

echo "Resuming training with sparsity constraint ..."

resume_from=../results/celeba/ambient/noisy/000-*-spparam0.0-thw0.0-*/main-network_004000.pt

python -u train_ambient.py --num_gpus $num_gpus --data_type CelebAHQDataset --data_args "{'on_the_fly': False, 'augmentation': 'vf'}" --results_dir $results_dir --input_shape 3 64 64 --main_model_type Glow --main_model_args "{'num_flow':32, 'num_block':4, 'filter_size':512}" --post_model_type CondConvINN2 --post_model_args "$post_model_args" --sparsifier_type GradientTransform --sparsifier_args "{}" --reg_parameter $reg_parameter --sparsity_weight $sparsity_weight --threshold_weight $threshold_weight --num_z $num_z --lr $lr --lr_post $lr_post --batch_size $batch_size --main_actnorm 1 --post_actnorm 1 --num_bits $num_bits --num_kimgs_post 0 --num_kimgs_main 0 --num_kimgs 20000 --degradation_type $degradation_type --degradation_args "$degradation_args" --schedule_sparsity_weight 0 --benchmark_cudnn 1 --importance_weighting $importance_weighting $other --resume_from $resume_from

python -c "import time; print('\a'); time.sleep(0.5); print('\a'); time.sleep(0.5); print('\a')"

