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
import models
import degradations
import ast
import os
import argparse
import imageio as io
import torch.distributions as dists
import sys
sys.path.append('../src')

parser = argparse.ArgumentParser()
# recon args
parser.add_argument("--idx", type=int, default=0, help="Index of the image to be reconstructed")
parser.add_argument("--reg_parameter", type=float, default=0, help="TV regularization parameter")
parser.add_argument("--step", type=float, default=1e-03, help="Step size")
parser.add_argument("--num_iter", type=int, default=10000, help="Number of iterations for recon")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--dataset", action='store_true')
parser.add_argument("--tune", action='store_true')
parser.add_argument("--reg_parameter2", type=float, default=1e-01, help="In case tune is true, use this to try multiple reg parameters")

# data args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

args = parser.parse_args()

def total_variation(img):
    x = img[:,:,1:,:] - img[:,:,:-1,:]
    y = img[:,:,:,1:] - img[:,:,:,:-1]
    tvnorm = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return tvnorm

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# forward model
args.data_args['input_shape'] = args.input_shape
fwd = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits, device=device)

# data
if args.dataset:
    clean_dataset = getattr(dataset_tool, args.data_type)(train=True, ambient=False, **args.data_args)
else:
    clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, **args.data_args)

xgt  ,_ = clean_dataset[args.idx]; xgt = xgt.reshape(1,*xgt.shape)
ymeas = fwd(xgt).to(device)
xgt = xgt.to(device)
# print(xgt.min(), xgt.max(), ymeas.numpy().min(), ymeas.cpu().numpy().max())

def run_iters():
    x = torch.zeros([1, *args.input_shape], requires_grad=True, device=device)
    optimizer = torch.optim.Adamax([x], lr=args.step)

    for i in range(args.num_iter):

        optimizer.zero_grad()
        tv_reg = args.reg_parameter * total_variation(x)
        loss = 0.5*torch.norm(ymeas - fwd.fwd_noiseless(x, use_device=True))**2 + tv_reg
        loss.backward()
        optimizer.step()

        if (i < 10) or (i < 50 and i % 10 == 0) or (i % 50 == 0):
            recon_error = torch.mean( (x - xgt)**2 )
            print(f"Idx : {i:05}, loss : {loss.cpu().detach().numpy()}, tv reg : {tv_reg}, recon. error : {recon_error}")
        
    ymeas_np = np.squeeze(ymeas.cpu().detach().numpy())    
    ymeas_np = np.swapaxes(ymeas_np.T, 0,1)
    xest = np.squeeze(x.cpu().detach().numpy())
    xest = np.swapaxes(xest.T, 0,1)
    xgt_np  = np.squeeze(xgt.cpu().detach().numpy())
    xgt_np = np.swapaxes(xgt_np.T, 0,1)

    if (not args.dataset) or (args.dataset and args.idx < 20):
        np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy'), ymeas_np)
        np.save( os.path.join(args.results_dir,   f'xest_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy' ), xest)
        np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy'  ), xgt_np)
        io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png' ), ymeas_np)
        io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png' ), xest)
        io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png'  ), xgt_np)

    return xgt_np, xest

if not args.tune:
    _, xest = run_iters()
    if args.dataset:
        os.makedirs(os.path.join(args.results_dir, 'dataset'), exist_ok=True)
        io.imsave( os.path.join(args.results_dir, 'dataset', f'xest_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png' ), xest)
else:
    xests = []
    reg_parameter_init = args.reg_parameter
    mses = []
    for i in range(5):
        args.reg_parameter = reg_parameter_init * ( args.reg_parameter2 / reg_parameter_init )**(i/4)
        xgt_np, xest = run_iters()
        xests.append(xest)
        mse = np.linalg.norm(xgt_np - xest)
        mses.append(mse)
    
    min_idx = np.where(np.array(mses) == min(mses))[0][0]
    xest = xests[min_idx]
    if args.dataset:
        os.makedirs(os.path.join(args.results_dir, 'dataset'), exist_ok=True)
        io.imsave( os.path.join(args.results_dir, 'dataset', f'xest_{args.idx}_reg{reg_parameter_init * ( args.reg_parameter2 / reg_parameter_init )**(min_idx/4)}_niter{args.num_iter}_step{args.step}.png' ), xest)






