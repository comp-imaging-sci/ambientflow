""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""

import numpy as np
import bm3d
import torch
import dataset_tool
import degradations
import ast
import os
import argparse
import imageio as io
import glob
import numpy.linalg as la
import sys
sys.path.append('../src')

parser = argparse.ArgumentParser()
# recon args
parser.add_argument("--num_images", type=int, default=50, help="Index of the image to be reconstructed")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--dataset", action='store_true')

# data args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})
parser.add_argument("--tune", action='store_true', help="Tune the regularization parameter")

args = parser.parse_args()

torch.manual_seed(0)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

# data
fnames = glob.glob(f"/shared/aristotle/SOMS/varun/ambientflow/data/CelebAHQDataset-GaussianNoise-0.2-image-data/*.png")
fnames_gt = glob.glob(f"/shared/aristotle/SOMS/varun/ambientflow/data/CelebAHQDataset-real-image-data/*.png")

for idx in range(args.num_images):
    print(idx)
    ymeas = io.imread(fnames[5000+idx]) / 255
    xgt = io.imread(fnames_gt[5000+idx]) / 255
    # ymeas = np.swapaxes(ymeas,0,1).T
    # ymeas = ymeas.reshape(1,3,64,64)
    # ymeas,_ = noisy_dataset[idx]; ymeas = ymeas.reshape(1, *ymeas.shape)
    # ymeas = degradation.rev(ymeas, mode='real', use_device=False)
    # print(xgt.min(), xgt.max(), ymeas.min(), ymeas.max())

    # ymeas = np.squeeze(ymeas.numpy())    
    # ymeas = np.swapaxes(ymeas.T, 0,1)
    # xgt  = np.squeeze(xgt.numpy())
    # xgt = np.swapaxes(xgt.T, 0,1)

    # if args.tune:
    #     xests = []
    #     for i in range(5):
    #         reg_parameter = args.reg_parameter * ( args.reg_parameter2 / args.reg_parameter )**(i/4)
    xest = bm3d.bm3d(ymeas, sigma_psd=args.reg_parameter, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    print("RMSE error : ", la.norm(xest - xgt)/np.sqrt(np.prod(xgt.shape)))

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

