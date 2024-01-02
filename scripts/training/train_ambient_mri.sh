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
lr_post=1e-04
batch_size=4
num_z=2
reg_parameter=5 # weight of likelihood term for second stage of training. 
importance_weighting=1
noise_std=0.1
subsampling=4
threshold_weight=0.05 # second stage of training
sparsity_weight=0.03 # second stage of training
cutoff_dim=0 # 0.75
cutoff_mul=1
other=${1} # other arguments, eg. --port 12357

post_model_args="{'num_flow': 4, 'num_block': 4, 'num_cond_chans': 32, 'filter_size': 32, 'final_cond_dim': 512, 'first_split_ratio': 0.25, 'squeezer': 'HaarSqueeze', 'sparsity_weight': ${sparsity_weight}, 'cutoff_dim': ${cutoff_dim}, 'cutoff_mul': ${cutoff_mul}}"

# Fully sampled
# degradation_args="{'random': False, 'mask_file':None, 'mean':0., 'std':$noise_std}"

# Downsampled
degradation_args="{'mask_file':'../masks_mri/cartesian_${subsampling}fold_128.npy', 'mean':0., 'std':$noise_std}"

results_dir=../results/mri/ambient/${subsampling}/
mkdir -p $results_dir

echo "Training for 2000 kimgs without sparsity constraint ..."

python -u train_ambient.py --num_gpus $num_gpus --data_type FastMRIT2Dataset --data_args "{'on_the_fly': False}" --results_dir $results_dir --input_shape 1 128 128 --main_model_type Glow --main_model_args "{'num_flow':16, 'num_block':4, 'filter_size':512}" --post_model_type CondGlow3 --post_model_args "$post_model_args" --sparsifier_type GradientTransform --sparsifier_args "{}" --reg_parameter 1.5 --sparsity_weight 0 --threshold_weight 0 --cutoff_dim $cutoff_dim --num_z $num_z --lr $lr --lr_post $lr_post --batch_size $batch_size --main_actnorm 1 --post_actnorm 1 --num_bits $num_bits --num_kimgs_post 0 --num_kimgs_main 0 --num_kimgs 2000 --degradation_type FourierSamplerGaussianNoise --degradation_args "$degradation_args" --schedule_sparsity_weight 0 --benchmark_cudnn 1 --importance_weighting $importance_weighting $other 

resume_from=../results/celeba/ambient/${subsampling}/000-*-spparam0.0-thw0.0-*/main-network_002000.pt

python -u train_ambient.py --num_gpus $num_gpus --data_type FastMRIT2Dataset --data_args "{'on_the_fly': False}" --results_dir $results_dir --input_shape 1 128 128 --main_model_type Glow --main_model_args "{'num_flow':16, 'num_block':4, 'filter_size':512}" --post_model_type CondGlow3 --post_model_args "$post_model_args" --sparsifier_type GradientTransform --sparsifier_args "{}" --reg_parameter $reg_parameter --sparsity_weight $sparsity_weight --threshold_weight $threshold_weight --cutoff_dim $cutoff_dim --num_z $num_z --lr $lr --lr_post $lr_post --batch_size $batch_size --main_actnorm 1 --post_actnorm 1 --num_bits $num_bits --num_kimgs_post 0 --num_kimgs_main 0 --num_kimgs 20000 --degradation_type FourierSamplerGaussianNoise --degradation_args "$degradation_args" --schedule_sparsity_weight 0 --benchmark_cudnn 1 --importance_weighting $importance_weighting $other --resume_from $resume_from

python -c "import time; print('\a'); time.sleep(0.5); print('\a'); time.sleep(0.5); print('\a')"

