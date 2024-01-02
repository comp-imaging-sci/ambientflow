""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 
"""

import sys
sys.path.append("../")
import argparse
import ast
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'size'   : 17}

matplotlib.rc('font', **font)

import models2
import dataset_tool
import degradations

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default='')

# Model args
parser.add_argument("--network_path", type=str, default='')
parser.add_argument("--num_images", type=int, default=500)
parser.add_argument("--ambient", type=int, default=1)

# degradation args
parser.add_argument("--degradation_type", type=str, default='GaussianNoise')
parser.add_argument("--degradation_args", type=ast.literal_eval, default={'mean':0., 'std':0.3})

# Data args
parser.add_argument("--input_shape", type=int, nargs='+', default=[3, 28, 28])
parser.add_argument("--data_type", type=str, default='MNISTDataset')
parser.add_argument("--data_args", type=ast.literal_eval, default={'power_of_two': False})
args = parser.parse_args()

# data
clean_dataset = getattr(dataset_tool, args.data_type)(train=False, ambient=False, input_shape=args.input_shape, **args.data_args)

# forward model
args.data_args['input_shape'] = args.input_shape
degradation = getattr(degradations, args.degradation_type)(**args.degradation_args, input_shape=args.input_shape, num_bits=0)

# models
main_model, post_model = models2.load_model(args.network_path, ambient=bool(args.ambient), load_post_model=True)
main_model = main_model.to('cuda')
post_model = post_model.to('cuda')

colors = ['b', 'orange', 'g', 'r', 'purple']
for j in range(3):
    logpxgys = []
    logpxs = []
    logpygxs = []

    for i in range(args.num_images):
        print(i)
        xgt  ,_ = clean_dataset[3*j]; xgt = xgt.reshape(1,*xgt.shape)
        ymeas = degradation(xgt).to('cuda')
        xgt = xgt.to('cuda')

        post_loss, x_posts = post_model.get_loss(ymeas, degradation, num_z=1, reg_parameter=0, num_bits=0, tiled=True, importance_weighting=1)
        logpxgys.append(post_loss.detach().cpu().numpy().squeeze())
        x_posts = x_posts[0]

        nll = main_model.get_loss(x_posts, importance_weighting=True) # importance_weighting=True only gives separate nlls instead of sum/average in this case
        logpxs.append(nll.detach().cpu().numpy().squeeze())

        l = - degradation.log_prob(ymeas, x_posts) / np.log(2) / np.prod(xgt.shape)
        logpygxs.append(l.detach().cpu().numpy().squeeze())

    logpxgys = np.array(logpxgys)
    logpxs = np.array(logpxs)
    logpygxs = np.array(logpygxs)

    pphi = logpxgys
    ptheta = -logpxs - logpygxs
    m, b = np.polyfit(pphi, ptheta, 1)

    plt.scatter(pphi, ptheta, c=colors[j], alpha=0.2)
    plt.plot( [pphi.min(), pphi.max()], [m*pphi.min()+b, m*pphi.max()+b], color=colors[j], label=f"Slope : {m:.1f}")
plt.legend()
plt.xlabel(r"$\log p_\phi ~(\mathbf{f}~ |~ \mathbf{g})$ (bits/dim)", math_fontfamily='cm')
plt.ylabel(r"$\log p_\theta ~(\mathbf{f}) ~+~ \log~ q_{\mathbf{g} | \mathbf{f}}~ (\mathbf{g}~ | ~\mathbf{f})$ (bits/dim)", math_fontfamily='cm')
plt.savefig(os.path.join(args.results_dir, 'log-odds.png'), bbox_inches='tight');plt.close()
plt.savefig(os.path.join(args.results_dir, 'log-odds.svg'), bbox_inches='tight');plt.close()



# print(f"Mean NLL : {mean_nll}, Median NLL : {medn_nll}, Std. NLL : {stdv_nll}")
# with open( os.path.join( args.results_dir, 'nll.txt' ), 'w') as fid:
#     print(f"Mean NLL : {mean_nll}, Median NLL : {medn_nll}, Std. NLL : {stdv_nll}", file=fid)





