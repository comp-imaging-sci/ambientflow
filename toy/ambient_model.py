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

def get_posterior_model(num_layers=8, block_type='AllInOneBlock'):

    cond = Ff.ConditionNode(2)
    nodes = [Ff.InputNode(2, name='Input')]

    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1], getattr(Fm, block_type), 
            {'subnet_constructor': subnet_fc, 
             'permute_soft'      : True},
             conditions=cond))

    model = Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
    model.block_type = block_type

    return model

def log_normal(var, std=1.):
    K = np.prod(var.shape[1:])
    var = torch.reshape(var, [var.shape[0], -1])
    return -K/2 * np.log(2*np.pi) - K * np.log(std) - 0.5 * torch.norm(var, dim=-1)**2 / std**2

def get_model_loss(model, importance_weighting=0):

    def loss_fn(data_batch):
        zbatch, jac = model([data_batch], jac=True)
        return - torch.mean(log_normal(zbatch) + jac) 

    def loss_fn_impw(data_batch):
        zbatch, jac = model([data_batch], jac=True)
        return - log_normal(zbatch) + jac

    if importance_weighting:
        return loss_fn_impw
    else:    return loss_fn

def get_posterior_loss(model, latent_bs=512, noise_std=0.1, num_z_per_data=1, reg_parameter=1., importance_weighting=0, **kwargs):

    if kwargs['fwd_model'] == 'simple_slant_squish':
        R = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2)],
            [1/np.sqrt(2),  1/np.sqrt(2)]
        ])
        A = torch.Tensor(R.T @ np.diag([np.sqrt(3), 1/np.sqrt(3)]) @ R).to(device)
        Ainv = torch.linalg.inv(A)
        fwd = lambda x : (A @ x.T).T
        fwdinv = lambda y : ( Ainv @ y.T).T
    elif kwargs['fwd_model'] == None:
        fwd = lambda x : x
        fwdinv = lambda y : y

    def loss_fn(data_batch):

        x_posts = []
        jac_rev = torch.zeros(size=(batch_size,)).to(device)
        data_fidelity_prob = torch.zeros(size=(batch_size,)).to(device)
        for i in range(num_z_per_data):
            z = np.random.randn(latent_bs, 2)
            x_post,jac_r = model(torch.Tensor(z).to(device), [fwdinv(data_batch)], rev=True)
            x_posts.append(x_post)
            jac_rev += jac_r/ num_z_per_data
            data_fidelity_prob += log_normal( data_batch - fwd(x_post) , std=noise_std) / num_z_per_data

        return - torch.mean(reg_parameter*data_fidelity_prob + jac_rev), x_posts

    def loss_fn_impw(data_batch):
        x_posts = []
        jac_rev = []
        data_fidelity_prob = []
        for i in range(num_z_per_data):
            z = np.random.randn(latent_bs, 2)
            x_post,jac_r = model(torch.Tensor(z).to(device), [fwdinv(data_batch)], rev=True)
            x_posts.append(x_post)
            # jac_rev += jac_r/ num_z_per_data
            jac_rev.append(jac_r)
            data_fidelity_prob.append( log_normal( data_batch - fwd(x_post) , std=noise_std) )
        jac_rev = torch.stack(jac_rev, dim=0)
        data_fidelity_prob = torch.stack(data_fidelity_prob, dim=0)

        return -(reg_parameter*data_fidelity_prob + jac_rev), x_posts

    if importance_weighting:
        return loss_fn_impw
    else:
        return loss_fn


def sample_from_cond(model, **kwargs):

    if kwargs['fwd_model'] == 'simple_slant_squish':
        R = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2)],
            [1/np.sqrt(2),  1/np.sqrt(2)]
        ])
        A = torch.Tensor(R.T @ np.diag([np.sqrt(3), 1/np.sqrt(3)]) @ R).to(device)
        Ainv = torch.linalg.inv(A)
        fwd = lambda x : (A @ x.T).T
        fwdinv = lambda y : ( Ainv @ y.T).T
    elif kwargs['fwd_model'] == None:
        fwd = lambda x : x
        fwdinv = lambda y : y

    def sampler(num_samples, cond):
        z = np.random.randn(num_samples, 2)
        cond = torch.stack([cond]*num_samples, dim=0).to(device)
        assert len(cond) == num_samples, "Wrong number of conditional or num_samples"
        x,_ = model(torch.Tensor(z).to(device), [fwdinv(cond)], rev=True)
        x = x.cpu().detach().numpy()
        return x
    
    return sampler


def sample_from(model):

    def sampler(num_samples, temp=1):
        z = np.random.randn(num_samples, 2) * temp
        x,_ = model(torch.Tensor(z).to(device), rev=True)
        x = x.cpu().detach().numpy()
        return x
    
    return sampler

def scatterplot_saver(*args, path):
    for dat in args:
        plt.scatter(dat[:,0], dat[:,1], alpha=0.3, edgecolor='none')
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

    N = 50000000
    batch_size = 2000
    num_iters = 100000
    num_iters_pre = 5000
    num_layers = 24
    num_layers_post = 16
    noise_std = 0.4
    reg_parameter = 0.05
    reg_parameter2 = 0.05
    num_z_per_data = 30
    importance_weighting = 0
    # fwd_model = 'simple_slant_squish'
    fwd_model = None

    if fwd_model == 'simple_slant_squish':
        R = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2)],
            [1/np.sqrt(2),  1/np.sqrt(2)]
        ])
        A = torch.Tensor(R.T @ np.diag([np.sqrt(3), 1/np.sqrt(3)]) @ R).to(device)
        Ainv = torch.linalg.inv(A)
        fwd = lambda x : (A @ x.T).T
        fwdinv = lambda y : ( Ainv @ y.T).T
    elif fwd_model == None:
        fwd = lambda x : x
        fwdinv = lambda y : y

    dataset, _ = data.generate('all', N, noise='gaussian', fwd_model=fwd_model, noise_std=noise_std)
    print(dataset.shape)

    main_model = get_main_model(num_layers)
    post_model = get_posterior_model(num_layers_post)

    main_model = main_model.to(device)
    post_model = post_model.to(device)

    main_trainable_parameters = [p for p in main_model.parameters() if p.requires_grad]
    post_trainable_parameters = [p for p in post_model.parameters() if p.requires_grad]
    main_optimizer = torch.optim.Adam(main_trainable_parameters, lr=2e-04)
    post_optimizer = torch.optim.Adam(post_trainable_parameters, lr=2e-04)
    tot_optimizer  = torch.optim.Adam(main_trainable_parameters + post_trainable_parameters, lr=2e-04)

    ## Pretrain the main model with noisy data
    main_loss_fn = get_model_loss(main_model)

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset), batch_size=batch_size, shuffle=True, drop_last=True)
    dloader = iter(train_dataloader)

    # Saving
    folder = '../results/toy/toy_ambient2'
    folder_no = len(glob.glob(f'{folder}/*'))
    folder = f'../results/toy/toy_ambient2/{folder_no:03}-{main_model.block_type}-{num_layers}/pre-main'
    os.makedirs(folder, exist_ok=True)
    scatterplot_saver(dataset[:1000], path=f'{folder}/reals.png')

    sampler = sample_from(main_model)

    # Pretrain the main model with noisy data
    print("Pretraining the main model with noisy data")
    idx = 0
    while True:
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]

        y = y.to(device)
        y = fwdinv(y)
        main_optimizer.zero_grad()
        loss = main_loss_fn(y)
        loss.backward()
        main_optimizer.step()

        if idx % 50 == 0:
            print(f"Idx. : {idx}, Curr. loss : {loss}")
        if idx % 50 == 0:
            x_samp = sampler(1000)
            scatterplot_saver(x_samp, path=f'{folder}/fakes_{idx:06}.png')
            torch.save(main_model, f'{folder}/network_{idx:06}.pth', pickle_module=dill)

        idx += 1
        if idx >= num_iters_pre:
            break


    ## Pretrain the posterior model
    post_loss_fn = get_posterior_loss(post_model, latent_bs=batch_size, reg_parameter=reg_parameter, num_z_per_data=num_z_per_data, fwd_model=fwd_model)

    dloader = iter(train_dataloader)

    # Saving
    folder = f'../results/toy/toy_ambient2/{folder_no:03}-{main_model.block_type}-{num_layers}/pre-post'
    os.makedirs(folder, exist_ok=True)
    y_tests,_ = data.generate('all', 8, noise='gaussian', shuffle=False, seed=1, fwd_model=fwd_model, noise_std=noise_std)
    real_post_samples = [data.generate_posterior_samples(200, yt, noise_std=noise_std) for yt in y_tests]
    scatterplot_saver(dataset[:100], *real_post_samples, path=f'{folder}/reals.png')
    # scatterplot_saver(dataset[:100], path=f'{folder}/reals.png')

    sampler = sample_from_cond(post_model, fwd_model=fwd_model)

    # Pretrain the posterior model
    print("Pretraining the posterior model")
    idx = 0
    while True:
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]

        y = y.to(device)
        post_optimizer.zero_grad()
        loss, _ = post_loss_fn(y)
        loss.backward()
        post_optimizer.step()

        if idx % 50 == 0:
            print(f"Idx. : {idx}, Curr. loss : {loss}")
        if idx % 50 == 0:
            x_samp = [sampler(200, yt) for yt in y_tests]
            scatterplot_saver(dataset[:100], *x_samp, path=f'{folder}/fakes_{idx:06}.png')
            torch.save(post_model, f'{folder}/network_{idx:06}.pth', pickle_module=dill)

        idx += 1
        if idx >= num_iters_pre:
            break

    
    ## Train both models jointly
    dloader = iter(train_dataloader)
    main_loss_fn = get_model_loss(main_model, importance_weighting=importance_weighting)
    post_loss_fn = get_posterior_loss(post_model, latent_bs=batch_size, reg_parameter=reg_parameter2, num_z_per_data=num_z_per_data, importance_weighting=importance_weighting, fwd_model=fwd_model)

    # Saving
    folder = f'../results/toy/toy_ambient2/{folder_no:03}-{main_model.block_type}-{num_layers}/main'
    os.makedirs(folder, exist_ok=True)
    data_gt,_ = data.generate('all', 1000, shuffle=True, seed=1)
    data_gt = data_gt.numpy()
    scatterplot_saver(dataset[:1000], data_gt, path=f'{folder}/reals.png')

    sampler = sample_from(main_model)
    cond_sampler = sample_from_cond(post_model, fwd_model=fwd_model)

    # Train both models jointly
    print("Training both models jointly")
    idx = 0
    while True:
        try:    y = next(dloader)[0]
        except StopIteration: 
            dloader = iter(train_dataloader)
            y = next(dloader)[0]

        y = y.to(device)
        tot_optimizer.zero_grad()
        loss_post, x_posts = post_loss_fn(y)

        if importance_weighting == 0:
            loss_main = torch.zeros(size=()).to(device)
            for x_post in x_posts:
                loss_main += main_loss_fn(x_post) / len(x_posts)
        else:
            loss_main = []
            for x_post in x_posts:
                loss_main.append( main_loss_fn(x_post) )
            loss_main = torch.stack(loss_main, dim=0)
        loss = loss_main + loss_post
        if importance_weighting:
            importance_weights = torch.softmax(loss, dim=0).detach()
            loss = loss * importance_weights
            loss = torch.mean(torch.sum(loss, dim=0))
        loss.backward()
        tot_optimizer.step()

        if idx % 50 == 0:
            print(f"Idx. : {idx}, Curr. loss : {loss}")
        if idx % 50 == 0:

            # save samples from main model
            x_samp = sampler(1000, temp=0.8)
            scatterplot_saver(dataset[:1000], data_gt, x_samp, path=f'{folder}/fakes_{idx:06}.png')

            # save samples from posterior model
            x_post = [cond_sampler(200, yt) for yt in y_tests]
            scatterplot_saver(dataset[:100], *x_post, path=f'{folder}/posts_{idx:06}.png')

            torch.save(post_model, f'{folder}/post-network_{idx:06}.pth', pickle_module=dill)
            torch.save(main_model, f'{folder}/main-network_{idx:06}.pth', pickle_module=dill)

        idx += 1
        if idx >= num_iters:
            break
