""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from skimage.restoration import wiener
from scipy.ndimage import gaussian_filter
import torch
import dataset_tool
import degradations
import ast
import os
import argparse
import imageio as io
import sys
sys.path.append('../src')

parser = argparse.ArgumentParser()
# recon args
parser.add_argument("--num_images", type=int, default=50, help="Index of the image to be reconstructed")
parser.add_argument("--reg_parameter", type=float, default=0.2, help="Wiener regularization parameter")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--dataset", action='store_true')

# data args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

args = parser.parse_args()

torch.manual_seed(0)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

# data
noisy_dataset = getattr(dataset_tool, args.data_type)(train=True, ambient=True,  degradation=degradation, **args.data_args)
clean_dataset = getattr(dataset_tool, args.data_type)(train=True, ambient=False, **args.data_args)
kernel = np.zeros(args.input_shape[1:])
kernel[args.input_shape[1]//2, args.input_shape[2]//2] = 1
kernel = gaussian_filter(kernel, args.degradation_args['kernel_sigma'])
kernel = kernel / kernel.max()
lw = int(4 * args.degradation_args['kernel_sigma'] + 0.5)
kernel = kernel[ 
    args.input_shape[1]//2 - lw : args.input_shape[1]//2 + lw + 1,
    args.input_shape[1]//2 - lw : args.input_shape[1]//2 + lw + 1 ]

for idx in range(args.num_images):
    print(idx)
    xgt  ,_ = clean_dataset[idx]; xgt = xgt.reshape(1,*xgt.shape)
    ymeas,_ = noisy_dataset[idx]; ymeas = ymeas.reshape(1, *ymeas.shape)
    ymeas = degradation.rev(ymeas, mode='real')
    # print(xgt.min(), xgt.max(), ymeas.min(), ymeas.max())

    ymeas = np.squeeze(ymeas.numpy())    
    ymeas = np.swapaxes(ymeas.T, 0,1)
    xgt  = np.squeeze(xgt.numpy())
    xgt = np.swapaxes(xgt.T, 0,1)

    xest = np.stack([
        wiener(ymeas[...,0], kernel, balance=args.reg_parameter),
        wiener(ymeas[...,1], kernel, balance=args.reg_parameter),
        wiener(ymeas[...,2], kernel, balance=args.reg_parameter),
    ], axis=-1)

    mse_error = np.linalg.norm(xest - xgt)**2
    print(mse_error)
    
    # ymeas = np.clip(ymeas,-0.5,0.5)
    if args.dataset:
        os.makedirs(os.path.join(args.results_dir, 'dataset'), exist_ok=True)

    if args.dataset:
        io.imsave( os.path.join(args.results_dir, 'dataset', f'xest_{idx}_reg{args.reg_parameter}.png' ), xest)
    else:
        np.save( os.path.join(args.results_dir,   f'ymeas_{idx}_reg{args.reg_parameter}.npy'), ymeas)
        np.save( os.path.join(args.results_dir,   f'xest_{idx}_reg{args.reg_parameter}.npy' ), xest)
        np.save( os.path.join(args.results_dir,   f'xgt_{idx}_reg{args.reg_parameter}.npy'  ), xgt)
        io.imsave( os.path.join(args.results_dir, f'ymeas_{idx}_reg{args.reg_parameter}.png' ), ymeas)
        io.imsave( os.path.join(args.results_dir, f'xest_{idx}_reg{args.reg_parameter}.png' ), xest)
        io.imsave( os.path.join(args.results_dir, f'xgt_{idx}_reg{args.reg_parameter}.png'  ), xgt)

