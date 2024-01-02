""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import numpy as np
import os
import glob
import imageio as io
import json
import sys
import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        return self.terminal.flush()

def scatterplot_saver(data, path):
    plt.scatter(data[:,0], data[:,1], alpha=0.3, edgecolor='none')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.savefig(path)
    plt.close()

def log_normal(var, std=1.):
    K = np.prod(var.shape[1:])
    var = torch.reshape(var, [var.shape[0], -1])
    return -K/2 * np.log(2*np.pi) - K * np.log(std) - 0.5 * torch.norm(var, dim=-1)**2 / std**2

def setup_saver(results_dir, identifier):
    folder_no = len(glob.glob(f'{results_dir}/*'))
    folder = f'{results_dir}/{folder_no:03}-{identifier}'
    os.makedirs(folder, exist_ok=True)
    return folder

def save_images(images, path, imrange=[0,1]):

    images = np.squeeze(images)
    if len(images.shape) == 4:
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)
        if images.shape[-1] == 2:
            images = images[...,0] + 1.j*images[...,1]
    # images = images.detach().numpy()
    N = int(np.sqrt(len(images)))
    imrows = [ np.concatenate(images[N*i:N*i+N], axis=1) for i in range(N) ]
    im = np.concatenate(imrows)
    im = np.clip(im, *imrange)
    im = (im - imrange[0]) / (imrange[1] - imrange[0]) * 255
    im = abs(im)
    im = im.astype(np.uint8)
    io.imsave(path, im)
    return im

def get_tv_loss(data_batch):
    data_shape = data_batch.shape 
    if len(data_shape) == 5:
        data_batch = data_batch.reshape( -1, *data_shape[2:] )

def total_variation(img):
    if len(img.shape) == 5:
        img = img.reshape( -1, *img.shape[2:] )
    x = img[:,:,1:,:] - img[:,:,:-1,:]
    y = img[:,:,:,1:] - img[:,:,:,:-1]
    tvnorm = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return tvnorm