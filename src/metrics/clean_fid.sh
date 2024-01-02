#!/bin/bash

# Copyright (c) 2022-2023 authors
# Author    : Varun A. Kelkar
# Email     : vak2@illinois.edu 

path_to_reals="<path to folder containing real image pngs>"
path_to_fakes="<path to folder containing fake image pngs>"

num_images=5000

results_dir="<path to results dir>"

mkdir -p $results_dir

python clean_fid.py --path_to_reals $path_to_reals --path_to_fakes $path_to_fakes --num_images $num_images
