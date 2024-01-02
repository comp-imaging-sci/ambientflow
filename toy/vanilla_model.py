""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from time import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
# from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm

torch.manual_seed(1234)
np.random.seed(42)

# from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
# from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

def get_main_model(num_layers=16, block_type='AllInOneBlock'):

    # Define the main flow
    mainflow = Ff.SequenceINN(2)
    for k in range(num_layers):
        mainflow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
    mainflow.block_type = block_type

    return mainflow
    
def get_main_model2(num_layers=8, block_type='GLOWCouplingBlock'):

    nodes = [Ff.InputNode(2, name='input')]

    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1],
                        Fm.GLOWCouplingBlock,
                        {'subnet_constructor':subnet_fc, 'clamp':2.0},
                        name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                        Fm.PermuteRandom,
                        {'seed':k},
                        name=F'permute_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        # TODO: Define the posterior flow
        # This will be a conditional INN, use the conditional INN paper arch for this.
    model = Ff.ReversibleGraphNet(nodes, verbose=False)
    model.block_type = block_type
    return model



def log_normal(var):
    K = np.prod(var.shape[1:])
    var = torch.reshape(var, [var.shape[0], -1])
    return -K/2 * np.log(2*np.pi) - 0.5 * torch.norm(var, dim=-1)**2

def get_loss(model):

    def loss_fn(data_batch):
        zbatch, jac = model([data_batch], jac=True)
        return - torch.mean(log_normal(zbatch) + jac) 

    return loss_fn

def sample_from(model):

    def sampler(num_samples):
        z = np.random.randn(num_samples, 2)
        z = torch.Tensor(z).to(device)
        x,_ = model(z, rev=True)
        x = x.detach().cpu().numpy()
        return x
    
    return sampler

def scatterplot_saver(data, path):
    plt.scatter(data[:,0], data[:,1], alpha=0.3, edgecolor='none')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.savefig(path)
    plt.close()



if __name__ == '__main__':
    import data
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import dill

    N = 5000000
    batch_size = 16384
    num_iters = 50000
    num_layers = 24

    dataset, _ = data.generate('all', N)
    print(dataset.shape)

    model = get_main_model(num_layers)

    model = model.to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=5e-04)

    loss_fn = get_loss(model)

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset), batch_size=batch_size, shuffle=True, drop_last=True)
    dloader = iter(train_dataloader)

    # Saving
    folder_no = len(glob.glob('../results/toy/toy_vanilla2/*'))
    folder = f'../results/toy/toy_vanilla2/{folder_no:03}-{model.block_type}-{num_layers}'
    os.makedirs(folder, exist_ok=True)
    scatterplot_saver(dataset[:1000], f'{folder}/reals.png')

    sampler = sample_from(model)

    print("Training vanilla model")
    idx = 0
    while True:
        try:    x = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            x = next(dloader)[0]

        x = x.to(device)
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            print(f"Idx. : {idx}, Curr. loss : {loss}")
        if idx % 100 == 0:
            x_samp = sampler(1000)
            scatterplot_saver(x_samp, f'{folder}/fakes_{idx:06}.png')
            torch.save(model, f'{folder}/network_{idx:06}.pth', pickle_module=dill)


        idx += 1
        if idx >= num_iters:
            break
