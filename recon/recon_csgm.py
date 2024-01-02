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
parser.add_argument("--lamda", type=float, default=0, help="MAP regularization parameter")
parser.add_argument("--tv", type=float, default=0, help="TV regularization parameter")
parser.add_argument("--step", type=float, default=1e-03, help="Step size")
parser.add_argument("--num_iter", type=int, default=10000, help="Number of iterations for recon")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--project_on_mask", action='store_true', help="Only applicable for inpainting")

# model args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--model_path", type=str, default='', help="Path to model pkl file")

# data args
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

args = parser.parse_args()
print(args, flush=True)

def total_variation(img):
    x = img[:,:,1:,:] - img[:,:,:-1,:]
    y = img[:,:,:,1:] - img[:,:,:,:-1]
    tvnorm = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return tvnorm

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# data
# noisy_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=True,  degradation=degradation, **args.data_args)
clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, input_shape=args.input_shape, **args.data_args)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

xgt  ,_ = clean_dataset[args.idx]; xgt = xgt.reshape(1,*xgt.shape)
# ymeas,_ = noisy_dataset[args.idx]; ymeas = ymeas.reshape(1, *ymeas.shape); ymeas = ymeas.to(device)
ymeas = degradation(xgt).to(device)
xgt = xgt.to(device)
print(xgt.min(), xgt.max(), abs(ymeas).min(), abs(ymeas).max())

# model
if '/ambient/' in args.model_path:
    model = load_model(args.model_path, ambient=True).to(device)
else: model = load_model(args.model_path).to(device)
model.eval()

# z = torch.zeros([1, np.prod(args.input_shape)], requires_grad=True).to(device)
z = torch.zeros( [1, np.prod(args.input_shape)], requires_grad=True, device=device)
# z = torch.nn.Parameter(z).to(device)
# model.initialize_actnorm(ymeas)
optimizer = torch.optim.Adam([z], lr=args.step)

for i in range(args.num_iter):

    optimizer.zero_grad()
    x,_ = model.reverse(z)
    map_reg = args.lamda * torch.sum(z**2)
    tv_reg = args.tv * total_variation(x)
    loss = 0.5*torch.norm(ymeas - degradation.fwd_noiseless(x, use_device=True))**2 + tv_reg + map_reg
    loss.backward()
    optimizer.step()

    if (i < 10) or (i < 50 and i % 10 == 0) or (i % 50 == 0) or (i == args.num_iter-1):
        if args.project_on_mask:
            x[:, degradation.mask.astype(bool)] = xgt[:, degradation.mask.astype(bool)]
        recon_error = torch.mean( (x - xgt)**2 )
        print(f"Idx : {i:05}, loss : {loss.cpu().detach().numpy()}, tv reg : {tv_reg}, recon. error : {recon_error}", flush=True)

ymeas = np.squeeze(ymeas.cpu().detach().numpy())    
ymeas = np.swapaxes(ymeas.T, 0,1)
xest = np.squeeze(x.cpu().detach().numpy())
xest = np.swapaxes(xest.T, 0,1)
xgt  = np.squeeze(xgt.cpu().detach().numpy())
xgt = np.swapaxes(xgt.T, 0,1)
np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.npy'), ymeas)
np.save( os.path.join(args.results_dir,   f'xest_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.npy' ), xest)
np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.npy'  ), xgt)
io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.png' ), ymeas)
io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.png' ), (np.clip((xest+0.5)*255, 0, 255)))
io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}_lam{args.lamda}_tv{args.tv}_niter{args.num_iter}_step{args.step}.png'  ), xgt)
