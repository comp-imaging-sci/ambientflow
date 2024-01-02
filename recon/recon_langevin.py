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
parser.add_argument("--T", type=int, default=200, help="Number of iterations T for each level of noise in Langevin annealing (except the last level, which may have a different number of iterations)")
parser.add_argument("--L", type=int, default=10, help='number of noise levels for annealing Langevin')
parser.add_argument("--num_iter", type=int, default=2000, help="Number of iterations")
parser.add_argument("--sigma_init", type=float, default=0.4, help="Initial noise level for annealing Langevin")
parser.add_argument("--sigma_final", type=float, default=0.1, help="Final noise level for annealing Langevin")
parser.add_argument("--num_bits", type=int, default=0)
parser.add_argument("--results_dir", type=str, default='', help="Results dir")
parser.add_argument("--anneal", action='store_true', help="Whether or not to anneal Langevin iterations")
parser.add_argument("--batch_size", type=int, default=50, help="Number of samples to draw from the posterior")
parser.add_argument("--step", type=float, default=0.001, help="Basic step size")
parser.add_argument("--momentum", type=float, default=0, help="Momentum between [0,1]")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature of the latent distribution")
parser.add_argument("--project_on_mask", action='store_true', help="Project on a mask, only applicable for inpainting.")

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
y_noiseless = degradation.fwd_noiseless(xgt).to(device)
ybatch = torch.cat([ymeas]*args.batch_size)
xgt = xgt.to(device)
print(xgt.min(), xgt.max(), abs(ymeas).min(), abs(ymeas).max())

# model
if '/ambient/' in args.model_path:
    model = load_model(args.model_path, ambient=True).to(device)
else: model = load_model(args.model_path).to(device)
model.eval()

# setup langevin params
T = args.T; L = args.L
sigma1 = args.sigma_init
sigmaT = args.sigma_final
factor = np.power(sigmaT / sigma1, 1/(L-1))
if args.anneal:
    sigma_func = lambda i: max( sigmaT, sigma1 * np.power(factor, i//T) )
    # sigma_func = lambda i: sigma1 * np.power(factor, i//T)
    lr_func = lambda t : (sigma1 * np.power(factor, t//T))**2 / (sigmaT **2)
else: 
    lr_func = lambda i : 1
    # sigma_func = lambda i : args.degradation_args['std']
    sigma_func = lambda i : args.sigma_final

zprior_weight = 0.5 / (args.temperature**2)
mloss_weight = 0.5 * np.prod(ymeas.shape) / (sigma1**2)
z = args.temperature * torch.randn( [args.batch_size, np.prod(args.input_shape)])
z = torch.tensor(z, requires_grad=True, device=device)

print("Training ...")
print("Measurement MSE : ", torch.mean(torch.abs(ymeas - y_noiseless)**2), flush=True)
optimizer = torch.optim.SGD([z], lr=args.step, momentum=args.momentum)
for i in range(args.num_iter):

    sigma = sigma_func(i)
    step_size = lr_func(i) * args.step
    mloss_weight = 0.5 * np.prod(ymeas.shape) / (sigma**2)

    optimizer.zero_grad()
    x,_ = model.reverse(z)
    z_loss_batch = torch.sum(z**2, dim=1)
    y_loss_batch = torch.sum( torch.abs( ybatch - degradation.fwd_noiseless(x, use_device=True))**2, dim=[1,2,3])
    loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    loss = torch.sum(loss_batch) #/ np.prod(ymeas.shape)
    loss.backward()

    # gradient step
    optimizer.param_groups[0]["lr"] = step_size
    optimizer.step()

    # randomization step
    grad_noise_weight = np.sqrt(2*step_size/(1-args.momentum))
    randomizing_noise = grad_noise_weight * torch.randn(z.shape).to(device)
    with torch.no_grad():
        z += randomizing_noise

    if (i < 10) or (i < 50 and i % 10 == 0) or (i % 50 == 0) or (i == args.num_iter-1):
        x,_ = model.reverse(z)
        x_mse = torch.mean(x, dim=0, keepdim=True)
        if args.project_on_mask:
            x[:, degradation.mask.astype(bool)] = xgt[:, degradation.mask.astype(bool)]
            x_mse[:, degradation.mask.astype(bool)] = xgt[:, degradation.mask.astype(bool)]
        recon_error = torch.mean( (x_mse - xgt)**2 )
        meas_error = torch.mean( torch.mean(y_loss_batch) / np.prod(ymeas.shape) )
        print(f"Idx : {i:05}, loss : {loss.cpu().detach().numpy()}, recon. error : {recon_error}, meas. error : {meas_error}", flush=True)

with open(os.path.join(args.results_dir, 'recon_error.txt'), 'w') as fid:
    print(f"MMSE Recon. error : {recon_error}", file=fid)

ymeas = np.squeeze(ymeas[0].cpu().detach().numpy())
xests = x.cpu().detach().numpy()
xmse = np.squeeze(x_mse.cpu().detach().numpy())
xgt  = np.squeeze(xgt.cpu().detach().numpy())
xvar = np.squeeze( np.std(xests, axis=0)**2 )
np.save( os.path.join(args.results_dir,   f'ymeas_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.npy'), ymeas)
np.save( os.path.join(args.results_dir,   f'xests_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.npy' ), xests)
np.save( os.path.join(args.results_dir,   f'xmse_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.npy' ), xmse)
np.save( os.path.join(args.results_dir,   f'xgt_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.npy'  ), xgt)
np.save( os.path.join(args.results_dir,   f'xvar_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.npy' ), xvar)

ymeas = np.swapaxes(ymeas.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'ymeas_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.png' ), ymeas)
for i,xest in enumerate(xests):
    xest = np.swapaxes(xest.T, 0,1)
    xest = np.squeeze(xest)
    io.imsave( os.path.join(args.results_dir, f'xest_{args.idx}_samp{i}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.png'), xest ) 
xmse = np.swapaxes(xmse.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xmse_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.png'  ), xmse)
xgt = np.swapaxes(xgt.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xgt_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.png'  ), xgt)
xvar = np.swapaxes(xvar.T, 0,1)
io.imsave( os.path.join(args.results_dir, f'xvar_{args.idx}_niter{args.num_iter}_step{args.step}_mom{args.momentum}_temp{args.temperature}.png'  ), xvar)