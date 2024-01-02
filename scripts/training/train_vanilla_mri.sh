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
num_bits=0 # 8
gpus=1
batch_size=18

results_dir=../results/mri/vanilla
echo $results_dir

python train_vanilla.py --gpus $gpus --num_kimgs 20000 --data_type FastMRIT2Dataset --results_dir $results_dir --input_shape 1 256 256 --model_type Glow --model_args "{'num_flow':16, 'num_block':6, 'filter_size':512}" --lr $lr --batch_size $batch_size --actnorm 1 --num_bits $num_bits
