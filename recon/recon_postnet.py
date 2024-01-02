""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.optim
import numpy as np
import dataset_tool
from models2 import *
import degradations
import ast
import os
import argparse
import imageio as io
import sys
sys.path.append('../src')

parser = argparse.ArgumentParser()
# recon args
parser.add_argument("--idx", type=int, default=0, help="Index of the image to be reconstructed")
parser.add_argument("--temperature", type=float, default=1, help="Temperature of the images sampled from the posterior network")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--batch_size", type=int, default=50, help="Number of samples to draw from the posterior")

# model args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--model_path", type=str, default='', help="Path to posterior model pkl file")

# data args
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

args = parser.parse_args()
print(args, flush=True)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

# data
# noisy_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=True,  degradation=degradation, **args.data_args)
clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, **args.data_args)

xgt  ,_ = clean_dataset[args.idx]; xgt = xgt.reshape(1,*xgt.shape)
# ymeas,_ = noisy_dataset[args.idx]; ymeas = ymeas.reshape(1, *ymeas.shape); ymeas = ymeas.to(device)
ymeas = degradation(xgt).to(device)
xrev = degradation.rev(ymeas)
xgt = xgt.to(device)
print(xgt.min(), xgt.max(), abs(ymeas).min(), abs(ymeas).max())

# model
post_model = load_post_model(args.model_path).to(device)
post_model.eval()

xests = post_model.sample(args.batch_size, xrev, temp=args.temperature)
x_mse = np.mean(xests, axis=0)



ymeas = np.squeeze(ymeas[0].cpu().detach().numpy())
xmse = np.squeeze(x_mse)
xgt  = np.squeeze(xgt.cpu().detach().numpy())
xvar = np.squeeze( np.std(xests, axis=0)**2 )

recon_error = np.sqrt( np.mean( (xmse - xgt)**2 ) )
print(f"MMSE Recon. error : {recon_error}", flush=True)
with open(os.path.join(args.results_dir, 'recon_error.txt'), 'w') as fid:
    print(f"MMSE Recon. error : {recon_error}", file=fid)    

np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}_temp{args.temperature}.npy'), ymeas)
np.save( os.path.join(args.results_dir,   f'xests_{args.idx}_temp{args.temperature}.npy' ), xests)
np.save( os.path.join(args.results_dir,   f'xmse_{args.idx}_temp{args.temperature}.npy' ), xmse)
np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}_temp{args.temperature}.npy'  ), xgt)
np.save( os.path.join(args.results_dir,   f'xvar_{args.idx}_temp{args.temperature}.npy' ), xvar)

ymeas = np.swapaxes(ymeas.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}_temp{args.temperature}.png' ), ymeas)
for i,xest in enumerate(xests):
    xest = np.squeeze(xest)
    io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}_samp{i}_temp{args.temperature}.png'), xest ) 
xmse = np.swapaxes(xmse.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xmse_{args.idx}_temp{args.temperature}.png'  ), xmse)
xgt = np.swapaxes(xgt.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}_temp{args.temperature}.png'  ), xgt)
xvar = np.swapaxes(xvar.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xvar_{args.idx}_temp{args.temperature}.png'  ), xvar)