""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np
import dataset_tool
import degradations
import ast
import os
import argparse
import imageio as io
import cv2
import sys
sys.path.append('../src')


parser = argparse.ArgumentParser()
# recon args
parser.add_argument("--idx", type=int, default=0, help="Index of the image to be reconstructed")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--project_on_mask", action='store_true', help="Only applicable for inpainting")

# data args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

args = parser.parse_args()
print(args, flush=True)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# data
# noisy_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=True,  degradation=degradation, **args.data_args)
clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, input_shape=args.input_shape, **args.data_args)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

xgt  ,_ = clean_dataset[args.idx]; xgt = xgt.reshape(1,*xgt.shape)
xgt = torch.flip(xgt, dims=[-1])
# ymeas,_ = noisy_dataset[args.idx]; ymeas = ymeas.reshape(1, *ymeas.shape); ymeas = ymeas.to(device)
ymeas = degradation(xgt)
ymeas = np.squeeze(ymeas.cpu().detach().numpy())
ymeas = np.swapaxes(ymeas.T, 0,1)
ymeas = (ymeas + 0.5) * 255
xgt  = np.squeeze(xgt.cpu().detach().numpy())
xgt = np.swapaxes(xgt.T, 0,1)
xgt = (xgt + 0.5) * 255

print(xgt.min(), xgt.max(), abs(ymeas).min(), abs(ymeas).max())

mask = (1 - degradation.mask[0])*255
mask = mask.astype(np.uint8)
ymeas = np.clip(ymeas,0,255).astype(np.uint8)

xest = cv2.inpaint(ymeas, mask, 3, cv2.INPAINT_NS)
xest = xest / 255 - 0.5
xgt = xgt / 255 - 0.5
ymeas = ymeas / 255 - 0.5

np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}.npy'), ymeas)
np.save( os.path.join(args.results_dir,   f'xest_{args.idx}.npy' ), xest)
np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}.npy'  ), xgt)
io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}.png' ), ymeas)
io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}.png' ), (np.clip((xest+0.5)*255, 0, 255)))
io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}.png'  ), xgt)
