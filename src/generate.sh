#!/bin/bash

# Copyright (c) 2022-2023 authors
# Author    : Varun A. Kelkar
# Email     : vak2@illinois.edu 

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

iter=$1
temperature=$2
batch_size=$3 
gen_dir="<Path to dir containing the network>"

python generate.py --gen_dir $gen_dir --iter $iter --temperature $temperature --ambient 0 --num_images 5000 --batch_size $batch_size