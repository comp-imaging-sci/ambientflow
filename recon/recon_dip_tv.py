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

# model args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--model_type", type=str, default='ConvINN')
parser.add_argument("--model_args", type=ast.literal_eval, default={'num_conv_layers': [4,12], 'num_fc_layers': [4]})

# data args
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
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=args.num_bits)

# model
model = getattr(models, args.model_type)(args.input_shape, **args.model_args, device=device).to(device)

# data
noisy_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=True,  degradation=degradation, **args.data_args)
clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, **args.data_args)

xgt  ,_ = clean_dataset[args.idx]; xgt = xgt.reshape(1,*xgt.shape); xgt = xgt.to(device)
ymeas,_ = noisy_dataset[args.idx]; ymeas = ymeas.reshape(1, *ymeas.shape); ymeas = ymeas.to(device)
print(xgt.min(), xgt.max(), ymeas.min(), ymeas.max())

z = torch.randn([1, *args.input_shape]).to(device)
# model.initialize_actnorm(ymeas)
optimizer = torch.optim.Adam(list(model.trainable_parameters), lr=args.step)

for i in range(args.num_iter):

    optimizer.zero_grad()
    x = model(z)
    tv_reg = args.reg_parameter * total_variation(x)
    loss = 0.5*torch.norm(ymeas - x)**2 + tv_reg
    loss.backward()
    optimizer.step()

    if (i < 10) or (i < 50 and i % 10 == 0) or (i % 50 == 0):
        recon_error = torch.mean( (x - xgt)**2 )
        print(f"Idx : {i:05}, loss : {loss.cpu().detach().numpy()}, tv reg : {tv_reg}, recon. error : {recon_error}")

ymeas = np.squeeze(ymeas.cpu().detach().numpy())    
ymeas = np.swapaxes(ymeas.T, 0,1)
xest = np.squeeze(x.cpu().detach().numpy())
xest = np.swapaxes(xest.T, 0,1)
xgt  = np.squeeze(xgt.cpu().detach().numpy())
xgt = np.swapaxes(xgt.T, 0,1)
np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy'), ymeas)
np.save( os.path.join(args.results_dir,   f'xest_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy' ), xest)
np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.npy'  ), xgt)
io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png' ), ymeas)
io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png' ), (np.clip((xest+0.5)*255, 0, 255)))
io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}_reg{args.reg_parameter}_niter{args.num_iter}_step{args.step}.png'  ), xgt)
