""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 
"""

from cleanfid import fid
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_reals", type=str)
parser.add_argument("--path_to_fakes", type=str)
parser.add_argument("--num_images", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=200)
args = parser.parse_args()

score = fid.compute_fid(args.path_to_fakes, args.path_to_reals, batch_size=args.batch_size, num_images=args.num_images)
print(score)