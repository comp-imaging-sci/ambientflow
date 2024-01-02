#!/bin/bash

# Copyright (c) 2022-2023 authors
# Author    : Varun A. Kelkar
# Email     : vak2@illinois.edu 

path_to_reals=/shared/aristotle/SOMS/varun/ambientflow/data/FastMRIT2Dataset-real-image-data
# path_to_reals=/shared/aristotle/SOMS/varun/ambientflow/data/CelebAHQDataset-real-image-data

num_images=50
data_type=FastMRIT2Dataset
degradation_type=FourierSamplerGaussianNoise

# degradation_args="{'mask_file':None, 'mean':0., 'std':0.1}"
degradation_args="{'mask_file':'../../masks_mri/cartesian_4fold_128.npy', 'mean':0., 'std':0.1}"

# 0x 0.1 noise
# network_path=/shared/anastasio1/SOMS/varun/ambientflow/results/mri/ambient/condglow/001-anastasio6-1gpu-degFourierSamplerGaussianNoise-Glow-CondGlow3-reg5.0-spparam0.03-thw0.05-bit0-lr0.0001-nz2-impw1/main-network_019906.pt

# 4x 0.1 noise
network_path=/shared/radon/SOMS/varun/ambientflow/results/mri/ambient/condglow/003-anastasio6-1gpu-degFourierSamplerGaussianNoise-Glow-CondGlow3-reg5.0-spparam0.03-thw0.05-bit0-lr0.0001-nz2-impw1/main-network_019301.pt

# results_dir=../../results/mri/ambient/log-odds/FourierSamplerGaussianNoise-0x-0.1
results_dir=../../results/mri/ambient/log-odds/FourierSamplerGaussianNoise-4x-0.1

mkdir -p $results_dir

python log_odds.py --data_type $data_type --data_args "{'power_of_two': True}" --input_shape 1 128 128 --degradation_type $degradation_type --degradation_args "$degradation_args" --network_path $network_path --results_dir $results_dir --num_images $num_images
