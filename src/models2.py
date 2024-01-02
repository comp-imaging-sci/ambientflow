""" 
Copyright (c) authors.
Author  : Varun A. Kelkar
Email   : vak2@illinois.edu

Parts copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from time import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import utils
import model_tools
from model_tools import *
import torch.distributions as dists
import ast 
import os
# from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm

torch.manual_seed(1234)
np.random.seed(42)

# from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
# from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path_to_pkl, ambient=False, load_post_model=False):
    if ambient:
        main = 'main_'
    else: main = ''
    folder_path = os.path.split(path_to_pkl)[0]
    configpath = os.path.join(folder_path, 'config.txt')
    with open(configpath, 'r') as fid:
        configs = fid.read()
    configs = ast.literal_eval(configs)
    main_model = globals()[configs[main+'model_type']](
        configs['input_shape'],
        **configs[main+'model_args']
    )
    main_model.load(path_to_pkl)
    if load_post_model:
        # raise NotImplementedError()
        post_model = globals()[configs['post_model_type']](
            configs['input_shape'],
            configs['cond_shape'],
            **configs['post_model_args']
        )
        suffix = path_to_pkl.split('_')[-1]
        path_to_post = os.path.join(folder_path, f'post-network_{suffix}')
        post_model.load(path_to_post)
        return main_model, post_model
    return main_model

def load_post_model(path_to_pkl):
    folder_path = os.path.split(path_to_pkl)[0]
    configpath = os.path.join(folder_path, 'config.txt')
    with open(configpath, 'r') as fid:
        configs = fid.read()
    configs = ast.literal_eval(configs)
        # raise NotImplementedError()
    post_model = globals()[configs['post_model_type']](
        configs['input_shape'],
        configs['cond_shape'],
        **configs['post_model_args']
    )
    post_model.load(path_to_pkl)
    return post_model


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out, interm=256):
    return nn.Sequential(nn.Conv2d(c_in, interm,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(interm,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 1))


# Loss function for the conditional INNs
def get_conditional_loss_freia(cond_model, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False, cutoff_dim=0, cutoff_mul=1):
    
    batch_size = len(data_batch)
    n_pixel = np.prod(cond_model.input_shape)
    num_bins = 2 ** num_bits

    with torch.no_grad():
        zs = cond_model.latent_dist.sample([batch_size*num_z, np.prod(cond_model.input_shape)])
    # cutoff_dim = int(np.prod(cond_model.input_shape) * cutoff_dim)
    # zs[:,:cutoff_dim] *= cutoff_mul

    if tiled:
        data_batch_tiled = torch.cat([data_batch]*num_z, 0)
        cond = cond_model.cond_net(degradation_op.rev(data_batch_tiled))
        x_posts, jac_rev = cond_model.model(zs, cond, rev=True)
        data_fidelity_prob = degradation_op.log_prob(data_batch_tiled, x_posts)
        x_posts = x_posts.view(num_z, batch_size, *cond_model.input_shape)
        jac_rev = jac_rev.view(num_z, batch_size)
        data_fidelity_prob = data_fidelity_prob.view(num_z, batch_size)

        if importance_weighting: # return an array of losses for diffeernt z and diffeent data samples
            losses = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
            losses = - losses / np.log(2) / n_pixel
            return losses, x_posts
        else:
            jac_rev = torch.mean(jac_rev, axis=0)
            data_fidelity_prob = torch.mean(data_fidelity_prob, axis=0)
    else:
        if importance_weighting: raise("Importance sampling is only impemented with tiling for loss")
        x_posts = []
        cond = cond_model.cond_net(degradation_op.rev(data_batch))
        jac_rev = torch.zeros(size=(batch_size,)).to(cond_model.device)
        data_fidelity_prob = torch.zeros(size=(batch_size,)).to(cond_model.device)
        for i in range(num_z):
            # z = torch.randn(batch_size, np.prod(cond_model.input_shape)).to(cond_model.device)
            z = zs[i*batch_size:(i+1)*batch_size]
            x_post,jac_r = cond_model.model(z, cond, rev=True)
            x_posts.append(x_post)
            jac_rev += jac_r/ num_z
            data_fidelity_prob_single = degradation_op.log_prob(data_batch, x_post)
            data_fidelity_prob += data_fidelity_prob_single / num_z
        x_posts = torch.stack(x_posts, dim=0)

    loss = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
    loss = - torch.mean(loss / np.log(2) / (n_pixel-cutoff_dim) )
    return loss, x_posts

def get_conditional_loss(cond_model, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False):
    
    batch_size = len(data_batch)
    n_pixel = np.prod(cond_model.input_shape)
    num_bins = 2 ** num_bits

    if tiled:
        # z = torch.randn(batch_size*num_z, np.prod(cond_model.input_shape)).to(cond_model.device)
        z = cond_model.latent_dist.sample([batch_size*num_z, cond_model.latent_shape])
        data_batch_tiled = torch.cat([data_batch]*num_z, 0)
        cond = degradation_op.rev(data_batch_tiled)
        x_posts, jac_rev, log_p_rev = cond_model.reverse(z, cond)
        jac_rev += log_p_rev
        data_fidelity_prob = degradation_op.log_prob(data_batch_tiled, x_posts)
        x_posts = torch.reshape(x_posts, [num_z, batch_size, *cond_model.input_shape])
        jac_rev = jac_rev.reshape(num_z, batch_size)
        data_fidelity_prob = data_fidelity_prob.reshape(num_z, batch_size)

        if importance_weighting: # return an array of losses for diffeernt z and diffeent data samples
            losses = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
            losses = - losses / np.log(2) / n_pixel 
            return losses, x_posts
        else:
            jac_rev = torch.mean(jac_rev, axis=0)
            data_fidelity_prob = torch.mean(data_fidelity_prob, axis=0)
    else:
        if importance_weighting: raise("Importance sampling requires tiling for loss")
        x_posts = []
        jac_rev = torch.zeros(size=(batch_size,)).to(cond_model.device)
        data_fidelity_prob = torch.zeros(size=(batch_size,)).to(cond_model.device)
        zs = cond_model.latent_dist.sample([batch_size*num_z, cond_model.latent_shape])
        for i in range(num_z):
            # z = torch.randn(batch_size, np.prod(cond_model.input_shape)).to(cond_model.device)
            z = zs[i*batch_size:(i+1)*batch_size]
            x_post, jac_r, log_p_r = cond_model.reverse(z, degradation_op.rev(data_batch))
            jac_r += log_p_r
            x_posts.append(x_post)
            jac_rev += jac_r/ num_z
            data_fidelity_prob_single = degradation_op.log_prob(data_batch, x_post)
            data_fidelity_prob += data_fidelity_prob_single / num_z
        x_posts = torch.stack(x_posts, dim=0)

    loss = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
    loss = - torch.mean(loss / np.log(2) / n_pixel )
    return loss, x_posts


class ConvINN(object):

    def __init__(self, input_shape=[3,28,28], num_conv_layers=[4,12], num_fc_layers=[12], device=None):

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        # print(input_shape, num_conv_layers, num_fc_layers)
        nodes = [Ff.InputNode(*input_shape, name='input')]
        ndim_x = np.prod(input_shape)

        # Higher resolution convolutional part
        for k in range(num_conv_layers[0]):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_conv, 'clamp':1.2},
                                name=F'conv_high_res_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_high_res_{k}'))

        nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

        # Lower resolution convolutional part
        for k in range(num_conv_layers[1]):
            if k%2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv

            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet, 'clamp':1.2},
                                name=F'conv_low_res_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_low_res_{k}'))

        # Make the outputs into a vector, then split off 1/4 of the outputs for the
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        split_node = Ff.Node(nodes[-1],
                            Fm.Split,
                            {'section_sizes':(ndim_x // 4, 3 * ndim_x // 4), 'dim':0},
                            name='split')
        nodes.append(split_node)

        # Fully connected part
        for k in range(num_fc_layers[0]):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_fc, 'clamp':2.0},
                                name=F'fully_connected_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_{k}'))

        # Concatenate the fully connected part and the skip connection to get a single output
        nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                            Fm.Concat1d, {'dim':0}, name='concat'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        conv_inn = Ff.GraphINN(nodes)

        self.model = conv_inn.to(self.device)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.identifier = "ConvINN-" + '-'.join(['{}']*(len(num_fc_layers) + len(num_conv_layers))).format(*num_conv_layers, *num_fc_layers)

    def sample(self, num_samples, temp=1.):
        z = np.random.randn(num_samples, np.prod(self.input_shape)) * temp
        x,_ = self.model(torch.Tensor(z).to(self.device), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch):

        zbatch, jac = self.model([data_batch.to(self.device)], jac=True)
        return - torch.mean(utils.log_normal(zbatch) + jac) 

    def save(self, *args, **kwargs):
        torch.save(self.model.state_dict(), *args, **kwargs)


class CondNet(nn.Module):
    '''conditioning network'''
    def __init__(self, cond_shape):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(3,  16, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(16, 16, 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(16,  32, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(32, 32, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(2),
                                         Flatten(),
                                         nn.Linear(32 * cond_shape[1]//4 * cond_shape[2]//4, 256))
                                         ])

    def forward(self, c):
        outputs = [c]
        for m in self.resolution_levels:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class CondConvINN(nn.Module):

    def __init__(self, input_shape=[3,28,28], cond_shape=[3,28,28], num_conv_layers=[4,12], num_fc_layers=[12], device=None):

        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        # print(input_shape, num_conv_layers, num_fc_layers)
        nodes = [Ff.InputNode(*input_shape, name='cond_input')]
        ndim_x = np.prod(input_shape)

        # Define the conditions
        # num_conds = int(np.ceil(np.log2(input_shape[-1]))) - 3
        # cond_channels = [np.min( base_cond_channels * 2**i , 256 ) for i in range(num_conds)]
        # conditions = [ 
        #     Ff.ConditionNode( np.min( base_cond_channels * 2**i, 256), 
        #         input_shape[1] // 2**i, input_shape[2] // 2**i) for i in range(num_conds)
        # ]

        conditions = [Ff.ConditionNode(16, input_shape[1], input_shape[2]),
                      Ff.ConditionNode(32, input_shape[1]//2, input_shape[2]//2),
                      Ff.ConditionNode(256)
                      ]

        # Higher resolution convolutional part
        for k in range(num_conv_layers[0]):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_conv, 'clamp':1.2},
                                conditions  = conditions[0],
                                name        = F'cond_conv_high_res_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'cond_permute_high_res_{k}'))

        nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

        # Lower resolution convolutional part
        for k in range(num_conv_layers[1]):
            if k%2 == 0:
                subnet = subnet_conv_1x1
            else:
                subnet = subnet_conv

            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet, 'clamp':1.2},
                                conditions  = conditions[1],
                                name        = F'cond_conv_low_res_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'cond_permute_low_res_{k}'))

        # Make the outputs into a vector, then split off 1/4 of the outputs for the
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        split_node = Ff.Node(nodes[-1],
                            Fm.Split,
                            {'section_sizes':(ndim_x // 4, 3 * ndim_x // 4), 'dim':0},
                            name='split')
        nodes.append(split_node)

        # Fully connected part
        for k in range(num_fc_layers[0]):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_fc, 'clamp':2.0},
                                conditions  = conditions[2],
                                name        = F'cond_fully_connected_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'cond_permute_{k}'))

        # Concatenate the fully connected part and the skip connection to get a single output
        nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                            Fm.Concat1d, {'dim':0}, name='cond_concat'))
        nodes.append(Ff.OutputNode(nodes[-1], name='cond_output'))

        conv_inn = Ff.GraphINN(nodes + conditions)

        self.model = conv_inn.to(self.device)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet(cond_shape=self.cond_shape).to(self.device)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = "CondConvINN-" + '-'.join(['{}']*(len(num_fc_layers) + len(num_conv_layers))).format(*num_conv_layers, *num_fc_layers)

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def forward(self, input, cond):
        z, jac = self.model([input], self.cond_net(cond), jac=True)
        return z, jac

    def sample(self, num_samples, cond, temp=1.):
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        cond = torch.cat([cond]*num_samples)
        x,_ = self.model(z, self.cond_net(cond), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False):
        
        return get_conditional_loss_freia(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting)

    def save(self, *args, **kwargs):
        torch.save(self.model.state_dict(), *args, **kwargs)


class Glow(nn.Module):
    def __init__(
        self, input_shape, num_flow, num_block, 
        filter_size     = 512, 
        affine          = True, 
        conv_lu         = True, 
        device          = None,
    ):
        super().__init__()

        self.device = device if device != None else DEVICE
        self.blocks = nn.ModuleList()
        self.input_shape = input_shape
        self.num_flow = num_flow
        self.num_block = num_block
        n_channel = input_shape[0]
        for i in range(num_block - 1):
            self.blocks.append(InvBlock(n_channel, num_flow, affine=affine, filter_size=filter_size, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(InvBlock(n_channel, num_flow, split=False, filter_size=filter_size, affine=affine))

        self.z_shapes, self.cum_idxs = self.calc_z_shapes()

        self.identifier = f"Glow-flow{num_flow}-block{num_block}"
        self.trainable_parameters = self.parameters()

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def calc_z_shapes(self):
        z_shapes = []
        input_size = self.input_shape[1]
        n_channel = self.input_shape[0]

        for i in range(self.num_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        z_inds = [np.prod(zz) for zz in z_shapes]
        z_inds_c = [np.sum(z_inds[:i+1]) for i in range(len(z_shapes))]
        return z_shapes, z_inds_c

    def initialize_actnorm(self, data_batch):
        with torch.no_grad():
            _,_,_ = self(data_batch.to(self.device))

    def list2array_z(self, z):
        bs = z[0].shape[0]
        assert np.prod( [ zz.shape[0] == bs for zz in z ] ), "Wrong shapes in z"

        if isinstance(z[0], np.ndarray):
            return np.concatenate([zz.reshape(bs, -1) for zz in z], axis=1)
        elif isinstance(z[0], torch.Tensor):
            return torch.cat( [torch.reshape(zz, [bs, -1]) for zz in z], dim=1)
    
    def array2list_z(self, z):
        bs = z.shape[0]
        z_shapes = [(0,)] + self.z_shapes
        z_idxs = [0] + self.cum_idxs
        z_list = []
        for i in range(1,len(z_shapes)):
            z_list.append( z[:,z_idxs[i-1]:z_idxs[i]].reshape(bs, *z_shapes[i]) )
        return z_list

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        z_outs = self.list2array_z(z_outs)
        return z_outs, log_p_sum, logdet

    def reverse(self, z, reconstruct=False):
        z_list = self.array2list_z(z)
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input, None

    def sample(self, num_samples, temp=1.):
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        x,_ = self.reverse(z)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, num_bits=0, importance_weighting=False):

        data_shape = list(data_batch.shape)
        if len(data_shape) == 5:
            data_batch = data_batch.reshape( -1, *data_shape[2:] )
        npixel = np.prod(self.input_shape)

        if num_bits == 0:
            _,log_p,logdet = self(data_batch.to(self.device))
            loss = log_p + logdet
        else:
            num_bins = 2 ** num_bits
            dbatch = data_batch + torch.rand_like(data_batch) / num_bins
            _,log_p,logdet = self(dbatch.to(self.device))
            loss = log_p + logdet - np.log(num_bins) * npixel
        
        if not importance_weighting:
            return - torch.mean( loss / np.log(2) / npixel )
        else: # return per-data-point array of losses (1D, will need to be reshaped)
            return - loss.reshape(data_shape[:-3]) / np.log(2) / npixel

    def save(self, *args, **kwargs):
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict)


class CondNet2(nn.Module):
    '''conditioning network'''
    def __init__(self, cond_shape, cond_layer_thicknesses=[64, 128, 128, 512], avg_pool=4):
        super().__init__()

        ct = cond_layer_thicknesses
        self.cond_shape = cond_shape

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        in_channels = 3 if self.cond_shape[0] == 3 else 4
        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(in_channels,  ct[0], 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(ct[0], ct[0], 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(ct[0], ct[1], 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(ct[1], ct[1], 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(ct[1], ct[2], 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(avg_pool),
                                         Flatten(),
                                         nn.Linear(ct[2] * cond_shape[1] // 16 * cond_shape[2] // 16, ct[3]))])

    def forward(self, c):
        if self.cond_shape[0] != 3:
            outputs = [forward_squeeze(c)]
        else: outputs = [c]
        for m in self.resolution_levels:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class CondConvINN2(nn.Module):
    '''cINN, including the ocnditioning network'''
    def __init__(self, 
        input_shape             = [3,64,64], 
        cond_shape              = [3,64,64], 
        num_conv_layers         = [2,4,4], 
        num_fc_layers           = [4], 
        cond_layer_thicknesses  = [64, 128, 128, 512],
        cond_avg_pool           = 4,
        device                  = None,
        **kwargs):
        """
        `input_shape`               : Shape of the *output* image of the conditional INN
        `cond_shape`                : Shape of the conditioning input
        `num_conv_layers`           : List containing the number of convolutional layers in each block. The len of the list is the number of convolutional blocks.
        `num_fc_layers`             : List containing the number of fully connected layers in each block. The len of the list is the number of fully connected blocks.
        `cond_layer_thicknesses`    : Number of channels in conditioning layer input. The conditioning input will be preprocessed to have these many channels.
        `cond_avg_pool`             : Average pooling ratio to use in the conditioning subnetwork.
        `device`                    : CPU/GPU device to load the model on. Defaults to DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        """

        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.cond_layer_thicknesses = cond_layer_thicknesses
        # nodes = [Ff.InputNode(*input_shape, name='cond_input')]
        # ndim_x = np.prod(input_shape)

        self.model = self.build_inn()
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet2(cond_shape=self.cond_shape, cond_layer_thicknesses=cond_layer_thicknesses, avg_pool=cond_avg_pool)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = "CondConvINN2-" + '-'.join(['{}']*(len(num_fc_layers) + len(num_conv_layers))).format(*num_conv_layers, *num_fc_layers)

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def build_inn(self):

        input_shape = self.input_shape
        cond_shape = self.cond_shape if input_shape[0] >= 2 else [self.cond_shape[0]*4, self.cond_shape[1]//2, self.cond_shape[2]//2]
        ct = self.cond_layer_thicknesses

        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        nodes = [Ff.InputNode(*input_shape)]
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(ct[0], cond_shape[1], cond_shape[2]),
                      Ff.ConditionNode(ct[1], cond_shape[1] // 2, cond_shape[2] // 2),
                      Ff.ConditionNode(ct[2], cond_shape[1] // 4, cond_shape[1] // 4),
                      Ff.ConditionNode(ct[3])]

        split_nodes = []

        nodes.append(Ff.Node(nodes[-1], model_tools.ActNormCoarse, {'inverse': True, 'scaling': 1./6}))

        subnet = sub_conv(32, 3)
        if self.input_shape[0] < 2:
            nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

        for k in range(self.num_conv_layers[0]):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[0]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))

        for k in range(self.num_conv_layers[1]):
            subnet = sub_conv(64, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[1]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 75% of the channels
        section_sizes = [ cond_shape[0], 3*cond_shape[0] ]
        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':section_sizes, 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))

        for k in range(self.num_conv_layers[2]):
            subnet = sub_conv(128, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[2]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 50% ch
        section_sizes = [2*cond_shape[0], 2*cond_shape[0]]
        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':section_sizes, 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

        # fully_connected part
        subnet = sub_fc(512)
        for k in range(self.num_fc_layers[0]):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[3]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def initialize_actnorm(self, data_batch, degradation_op):
        for m in self.model.module_list:
            if isinstance(m, model_tools.ActNormCoarse):
                m.init_on_next_batch = True
        with torch.no_grad():
            data_batch_rev = degradation_op.rev(data_batch)
            x = self.sample(data_batch.shape[0], data_batch_rev)
            # After 1 pass through self.model, init_on_next_batch on all the actnorms automatically is set to False again.

    def forward(self, input, cond):
        z, jac = self.model([input], self.cond_net(cond), jac=True)
        return z, jac

    def sample(self, num_samples, cond, temp=1.):
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        if cond.shape[0] == 1:
            cond = torch.cat([cond]*num_samples)
        else: assert cond.shape[0] == num_samples, "batch size of condition must be equal to 1 or num_samples"
        x,_ = self.model(z, self.cond_net(cond), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False, cutoff_dim=0., cutoff_mul=0.):
        
        return get_conditional_loss_freia(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting)

    def save(self, *args, **kwargs):
        # torch.save(self.model.state_dict(), *args, **kwargs)
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict, strict=False)


class CondNet3(nn.Module):
    '''conditioning network'''
    def __init__(self, cond_shape, 
    num_cond_chans      = 4,
    num_block           = 5,
    num_noncond_blocks  = 0,
    ):
        super().__init__()

        self.cond_shape = cond_shape
        self.num_cond_chans = num_cond_chans
        self.num_noncond_blocks = num_noncond_blocks

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        # Note that at the beginning the input is Haar Squeezed based on the num_noncond_blocks to match the sizes
        self.blocks = nn.ModuleList()
        C = self.cond_shape[0] * (4 ** num_noncond_blocks)
        ncc = num_cond_chans
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(C, ncc, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(ncc, ncc, 3, padding=1),
            )
        )
        for i in range(num_block - num_noncond_blocks - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Conv2d(ncc, ncc, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(ncc, ncc*2, 3, padding=1, stride=2),
                )
            )
            ncc *= 2

    def forward(self, c):
        for i in range(self.num_noncond_blocks):
            c = Squeeze()(c)
        outputs = [c]
        for m in self.blocks:
            outputs.append(m(outputs[-1]))
        return outputs[1:]


class CondLDFlow(nn.Module):
    "Conditional low dimenional injective flow achitecture"
    def __init__(self,
        input_shape             = [3,64,64],
        cond_shape              = [3,64,64],
        num_flow                = 2,
        num_block               = 4,
        num_cond_chans          = 16,
        compression_ratio       = 4,
        num_noncond_blocks      = 1,
        filter_size             = 512,
        gamma                   = 1e-08,
        squeezer                = 'Squeeze',
        device                  = None):
        """
        input_shape             : shape of image output of the posterior network
        cond_shape              : shape of the input to the conditioning network
        num_flow                : number of flows in each invertible block
        num_cond_chans          : number of channels in the first conditioning layer
        compression_ratio       : ratio of latent space size and image size
        num_noncond_blocks      : num of blocks to go without conditioning
        filter_size             : filter size in affine coupling 
        tikhonov_gamma          : reg parameter for tikhonov inversion
        """


        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_flow = num_flow
        self.num_block = num_block
        self.num_cond_chans = num_cond_chans
        self.gamma = gamma
        self.compression_ratio = compression_ratio # must be a power of two
        self.num_inj_flows = num_inj_flows = int(np.log2(compression_ratio))
        self.squeezer = getattr(model_tools, squeezer)

        C = input_shape[0]
        self.blocks = nn.ModuleList([])
        nif = 0
        for i in range(num_noncond_blocks):
            if nif < num_inj_flows:
                self.blocks.append(
                    InvBlock(C, num_flow, split=False, filter_size=filter_size, extract_z=False, squeeze=True, squeezer=self.squeezer).to(self.device)
                )
                self.blocks.append(
                    InjBlock(4*C, 1, gamma=gamma, filter_size=filter_size, extract_z=False, squeeze=False).to(self.device)
                )
                nif += 1
            else:
                self.blocks.append(
                    InvBlock(C, num_flow, split=True, filter_size=filter_size, extract_z=True, squeeze=True, squeezer=self.squeezer).to(self.device)
                )
            C *= 2
            
        ncc = num_cond_chans
        for i in range(num_block - num_noncond_blocks - 1):
            if nif < num_inj_flows:
                self.blocks.append(
                    InvCondBlock(C, num_flow, split=False, filter_size=filter_size, num_cond_layers=ncc, extract_z=False, squeeze=True, squeezer=self.squeezer).to(self.device)
                )
                self.blocks.append(
                    InjBlock(4*C, 1, gamma=gamma, filter_size=filter_size, extract_z=False, squeeze=False).to(self.device)
                )
                nif += 1
            else:
                self.blocks.append(
                    InvCondBlock(C, num_flow, split=True, filter_size=filter_size, num_cond_layers=ncc, extract_z=True, squeeze=True, squeezer=self.squeezer).to(self.device)
                )
            C *= 2
            ncc *= 2

        self.blocks.append(
            InvCondBlock(C, num_flow, split=False, filter_size=filter_size, num_cond_layers=ncc, extract_z=True, squeeze=True, squeezer=self.squeezer).to(self.device)
        )
                
        self.cond_net = CondNet3(
            cond_shape              = self.cond_shape, 
            num_cond_chans          = self.num_cond_chans,
            num_block               = self.num_block,
            num_noncond_blocks      = num_noncond_blocks,
        ).to(self.device)

        self.trainable_parameters = self.parameters()

        self.z_shapes, self.cum_idxs = self.calc_z_shapes()
        self.latent_shape = np.sum([np.prod(zsh) for zsh in self.z_shapes])
        self.identifier = f"CondLDFlow-nf{num_flow}-nb{num_block}-nc{num_cond_chans}-cr{compression_ratio}-nn{num_noncond_blocks}"

        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def calc_z_shapes(self):
        z_shapes = []
        input_size = self.input_shape[1]
        n_channel = self.input_shape[0]

        for i in range(self.num_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        z_shapes = z_shapes[self.num_inj_flows:]

        z_inds = [np.prod(zz) for zz in z_shapes]
        z_inds_c = [np.sum(z_inds[:i+1]) for i in range(len(z_shapes))]

        return z_shapes, z_inds_c

    def initialize_actnorm(self, data_batch, cond_batch):
        with torch.no_grad():
            _,_,_ = self(data_batch.to(self.device), cond_batch.to(self.device))

    def list2array_z(self, z):
        bs = z[0].shape[0]
        assert np.prod( [ zz.shape[0] == bs for zz in z ] ), "Wrong shapes in z"

        if isinstance(z[0], np.ndarray):
            return np.concatenate([zz.reshape(bs, -1) for zz in z], axis=1)
        elif isinstance(z[0], torch.Tensor):
            return torch.cat( [torch.reshape(zz, [bs, -1]) for zz in z], dim=1)

    def array2list_z(self, z):
        bs = z.shape[0]
        z_shapes = [(0,)] + self.z_shapes
        z_idxs = [0] + self.cum_idxs
        z_list = []
        for i in range(1,len(z_shapes)):
            z_list.append( z[:,z_idxs[i-1]:z_idxs[i]].reshape(bs, *z_shapes[i]) )
        return z_list

    def forward(self, input, cond):

        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        conditions = self.cond_net(cond)
        conds = iter(conditions)

        for i,block in enumerate(self.blocks):
            if isinstance(block, InvCondBlock):
                cond = next(conds)
                out, det, log_p, z_new = block(out, cond)
            else:
                out, det, log_p, z_new = block(out)
            
            logdet += det
            log_p_sum += log_p
            if z_new != None:
                z_outs.append(z_new)

        z_outs = self.list2array_z(z_outs)
        return z_outs, log_p_sum, logdet

    def reverse(self, z, cond):

        log_p_sum = 0
        logdet = 0
        z_list = self.array2list_z(z)
        conditions = self.cond_net(cond)
        out = z_list[-1]
        conds = iter(conditions[::-1])
        zs = iter(z_list[::-1])

        for i, block in enumerate(self.blocks[::-1]):
            try: z = next(zs)
            except StopIteration: z = out

            if isinstance(block, InvCondBlock):
                cond = next(conds)
                out, det, log_p = block.reverse(out, cond, z, jac=True)
            else:
                out, det, log_p = block.reverse(out, z, jac=True)

            logdet += det
            log_p_sum += log_p

        return out, log_p_sum, logdet

    def sample(self, num_samples, cond, temp=1.):
        if len(cond.shape) == 3:    cond = cond.unsqueeze(0)
        if cond.shape[0] == 1:      cond = torch.cat([cond]*num_samples, dim=0)
        assert cond.shape[0] == num_samples, "Num. conditionals should be 1 or equal to num_samples"

        z = self.latent_dist.sample([num_samples, self.latent_shape]) * temp
        x,_,_ = self.reverse(z, cond)
        return x.cpu().detach().numpy()
        
    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False):

        return get_conditional_loss(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting)

    def save(self, *args, **kwargs):
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict)

class CondNet4(nn.Module):
    '''conditioning network'''
    def __init__(self, cond_shape, 
    num_cond_chans      = 4,
    num_block           = 5,
    linear_layer_dim    = 512,
    squeezer            = Squeeze(),
    squeeze_first       = True,
    ):
        super().__init__()

        self.cond_shape = cond_shape
        self.num_cond_chans = num_cond_chans
        self.squeezer = squeezer
        self.squeeze_first = squeeze_first

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        # Note that at the beginning the input is Squeezed to match the sizes
        self.blocks = nn.ModuleList()
        if squeeze_first:
            C = 4*self.cond_shape[0]
            H = self.cond_shape[1] // 2
        else:
            C = self.cond_shape[0]
            H = self.cond_shape[1]
        ncc = num_cond_chans
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(C, ncc, 3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(ncc, ncc, 3, padding=1),
            )
        )
        for i in range(num_block - 2):
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Conv2d(ncc, ncc, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(ncc, ncc*2, 3, padding=1, stride=2),
                )
            )
            ncc *= 2

        self.blocks.append(
            nn.Sequential(
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(
                    ncc * (H // 2**(num_block - 2)) * (H // 2**(num_block - 2)),
                    linear_layer_dim
                )
            )
        )

    def forward(self, c):
        if self.squeeze_first:
            c = self.squeezer(c)
        outputs = [c]
        for m in self.blocks:
            outputs.append(m(outputs[-1]))
        return outputs[1:]


class CondGlow(nn.Module):
    "Conditional flow architecture"
    def __init__(self,
        input_shape             = [3,64,64],
        cond_shape              = [3,64,64],
        num_flow                = 2,
        num_block               = 4,
        num_cond_chans          = 16,
        filter_size             = 32,
        final_cond_dim          = 512,
        first_split_ratio       = 0.25,
        squeezer                = 'Squeeze',
        device                  = None):
        """
        input_shape             : shape of image output of the posterior network
        cond_shape              : shape of the input to the conditioning network
        num_flow                : number of flows in each invertible block
        num_cond_chans          : number of channels in the first conditioning layer
        compression_ratio       : ratio of latent space size and image size
        num_noncond_blocks      : num of blocks to go without conditioning
        filter_size             : filter size in affine coupling 
        """


        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_flow = num_flow
        self.num_block = num_block
        self.num_cond_chans = num_cond_chans
        self.filter_size = filter_size
        self.final_cond_dim = final_cond_dim
        self.first_split_ratio = first_split_ratio
        self.squeezer = getattr(model_tools, squeezer)(cond_shape[0])

        self.model = self.build_inn()
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet4(cond_shape=self.cond_shape, num_cond_chans=self.num_cond_chans, num_block=self.num_block, linear_layer_dim=self.final_cond_dim, squeezer=self.squeezer)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = f"CondGlow-nf{num_flow}-nb{num_block}-nc{num_cond_chans}-fil{filter_size}-spl{first_split_ratio}"

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def build_inn(self):

        C = self.input_shape[0]
        filter_size = self.filter_size

        def subnet_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def subnet_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        def add_block(nodes, num_flow, subnet, condition, clamp=1.0):
            for k in range(num_flow):
                nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                        {'subnet_constructor':subnet, 'clamp':clamp},
                                        conditions=condition))
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
            return nodes

        def split(nodes, split_nodes, section_sizes):
            nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':section_sizes, 'dim':0}))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
            return nodes, split_nodes

        nodes = [Ff.InputNode(*self.input_shape)]
        conditions = []
        ncc = self.num_cond_chans
        chans = C
        cs = self.cond_shape[1] // 2
        for i in range(self.num_block - 1):
            conditions.append(Ff.ConditionNode(ncc, cs, cs))
            ncc *= 2; cs //= 2
        conditions.append(Ff.ConditionNode(self.final_cond_dim))

        split_nodes = []

        # First block with Wavelet Downsampling
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
        subnet = subnet_conv(filter_size, 3)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[0])
        section_sizes = [ int(4*chans*self.first_split_ratio), int(4*chans*(1-self.first_split_ratio)) ]
        print(section_sizes)
        nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
        chans = int(4*chans*self.first_split_ratio)
        filter_size *= 2

        for bl in range(1,self.num_block-1):
            # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
            subnet = subnet_conv(filter_size, 3)
            nodes = add_block(nodes, self.num_flow, subnet, conditions[bl], clamp=0.6)
            section_sizes = [ 2*chans, 2*chans ]
            nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
            chans *= 2
            filter_size *= 2
            
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        subnet = subnet_fc(filter_size)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[-1], clamp=0.6)

        # concat the split nodes
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def forward(self, input, cond):
        z, jac = self.model([input], self.cond_net(cond), jac=True)
        return z, jac

    def sample(self, num_samples, cond, temp=1.):
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        cond = torch.cat([cond]*num_samples)
        x,_ = self.model(z, self.cond_net(cond), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False):
        
        return get_conditional_loss_freia(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting)

    def save(self, *args, **kwargs):
        # torch.save(self.model.state_dict(), *args, **kwargs)
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict, strict=False)


class CondGlow2(nn.Module):
    "Conditional flow architecture"
    def __init__(self,
        input_shape             = [3,64,64],
        cond_shape              = [3,64,64],
        num_flow                = 2,
        num_block               = 4,
        num_cond_chans          = 16,
        filter_size             = 32,
        final_cond_dim          = 512,
        first_split_ratio       = 0.25,
        squeezer                = 'Squeeze',
        sparsity_weight         = 0.1,
        sparsifier              = None,
        cutoff_dim              = 0,
        cutoff_mul              = 1,
        permutation             = 'random',
        device                  = None):
        """
        input_shape             : shape of image output of the posterior network
        cond_shape              : shape of the input to the conditioning network
        num_flow                : number of flows in each invertible block
        num_cond_chans          : number of channels in the first conditioning layer
        compression_ratio       : ratio of latent space size and image size
        num_noncond_blocks      : num of blocks to go without conditioning
        filter_size             : filter size in affine coupling 
        """


        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_flow = num_flow
        self.num_block = num_block
        self.num_cond_chans = num_cond_chans
        self.filter_size = filter_size
        self.final_cond_dim = final_cond_dim
        self.first_split_ratio = first_split_ratio
        self.squeezer = getattr(model_tools, squeezer)(cond_shape[0])
        self.sparsity_weight = sparsity_weight
        self.sparsifier = sparsifier
        self.cutoff_dim = cutoff_dim
        self.cutoff_mul = cutoff_mul
        self.permutation = permutation

        self.model = self.build_inn()
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet4(cond_shape=self.cond_shape, num_cond_chans=self.num_cond_chans, num_block=self.num_block, linear_layer_dim=self.final_cond_dim, squeezer=self.squeezer, squeeze_first=False)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = f"CondGlow-nf{num_flow}-nb{num_block}-nc{num_cond_chans}-fil{filter_size}-spl{first_split_ratio}"

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def build_inn(self):

        C = self.input_shape[0]
        filter_size = self.filter_size

        def subnet_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def subnet_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        def add_block(nodes, num_flow, subnet, condition, clamp=1.0, permutation='invconv'):
            for k in range(num_flow):
                nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                        {'subnet_constructor':subnet, 'clamp':clamp},
                                        conditions=condition))
                # nodes.append(Ff.Node(nodes[-1], model_tools.ActNormCoarse, {'inverse': True, 'scaling': 1.}))
                # nodes.append(Ff.Node(nodes[-1], model_tools.ActNorm2, {'inverse': True, 'scaling': 1.}))
                if permutation=='random':
                    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
                elif permutation=='invconv':
                    nodes.append(Ff.Node(nodes[-1], model_tools.Inv1x1Conv, {}))
            return nodes

        def split(nodes, split_nodes, section_sizes):
            nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':section_sizes, 'dim':0}))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
            return nodes, split_nodes

        nodes = [Ff.InputNode(*self.input_shape)]
        conditions = []
        ncc = self.num_cond_chans
        chans = C
        cs = self.cond_shape[1]
        for i in range(self.num_block - 1):
            conditions.append(Ff.ConditionNode(ncc, cs, cs))
            ncc *= 2; cs //= 2
        conditions.append(Ff.ConditionNode(self.final_cond_dim))

        split_nodes = []


        nodes.append(Ff.Node(nodes[-1], model_tools.ActNormCoarse, {'inverse': True, 'scaling': 1./6}))

        subnet = subnet_conv(filter_size, 3)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[0], permutation=self.permutation)
        # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
        section_sizes = [ int(4*chans*self.first_split_ratio), int(4*chans*(1-self.first_split_ratio)) ]
        nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
        chans = int(4*chans*self.first_split_ratio)
        filter_size *= 2

        for bl in range(1,self.num_block-1):
            subnet = subnet_conv(filter_size, 3)
            nodes = add_block(nodes, self.num_flow, subnet, conditions[bl], clamp=0.6, permutation=self.permutation)
            # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
            section_sizes = [ 2*chans, 2*chans ]
            nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
            chans *= 2
            filter_size *= 2
            
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        subnet = subnet_fc(filter_size)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[-1], clamp=0.6, permutation='random')

        # concat the split nodes
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def change_sparsity_weight(self, new_weight):
        self.model.module_list[0].threshold = new_weight

    # def initialize_actnorm(self, data_batch, degradation_op):
    #     with torch.no_grad():
    #         data_batch_rev = degradation_op.rev(data_batch)
    #         x = self.sample(data_batch.shape[0], data_batch_rev)
    #         std = np.std(x)
    #         self.model.module_list[0].final_scale *= 1/std/6
    #         self.model.module_list[0].final_scale.to(self.device)

    def initialize_actnorm(self, data_batch, degradation_op):
        for m in self.model.module_list:
            if isinstance(m, model_tools.ActNormCoarse):
                m.init_on_next_batch = True
        with torch.no_grad():
            data_batch_rev = degradation_op.rev(data_batch)
            x = self.sample(data_batch.shape[0], data_batch_rev)
            # After 1 pass through self.model, init_on_next_batch on all the actnorms automatically is set to False again.

    def forward(self, input, cond, rev=False):

        if not rev:
            z, jac = self.model([input], self.cond_net(cond), jac=True)
            return z, jac

        else:
            x_posts, jac_rev = self.model(input, self.cond_net(cond), rev=True)
            return x_posts, jac_rev

    def sample(self, num_samples, cond, temp=1., cutoff_dim=None, cutoff_mul=None):
        if cutoff_dim == None: cutoff_dim = self.cutoff_dim
        if cutoff_mul == None: cutoff_mul = self.cutoff_mul
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        cutoff_dim = int(np.prod(self.input_shape) * cutoff_dim)
        z[:, :cutoff_dim] *= cutoff_mul
        if cond.shape[0] == 1:
            cond = torch.cat([cond]*num_samples)
        else: assert cond.shape[0] == num_samples, "batch size of condition must be equal to 1 or num_samples"
        x,_ = self.model(z, self.cond_net(cond), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False, cutoff_dim=None, cutoff_mul=None):
        if cutoff_dim == None: cutoff_dim = self.cutoff_dim
        if cutoff_mul == None: cutoff_mul = self.cutoff_mul 
        return get_conditional_loss_freia(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting, cutoff_dim=cutoff_dim, cutoff_mul=cutoff_mul)

    def save(self, *args, **kwargs):
        # torch.save(self.model.state_dict(), *args, **kwargs)
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict, strict=False)


class CondGlow3(nn.Module):
    "Conditional flow architecture"
    def __init__(self,
        input_shape             = [3,64,64],
        cond_shape              = [3,64,64],
        num_flow                = 2,
        num_block               = 4,
        num_cond_chans          = 16,
        filter_size             = 32,
        final_cond_dim          = 512,
        first_split_ratio       = 0.25,
        squeezer                = 'Squeeze',
        sparsity_weight         = 0.1,
        sparsifier              = None,
        cutoff_dim              = 0,
        cutoff_mul              = 1,
        device                  = None):
        """
        input_shape             : shape of image output of the posterior network
        cond_shape              : shape of the input to the conditioning network
        num_flow                : number of flows in each invertible block
        num_cond_chans          : number of channels in the first conditioning layer
        compression_ratio       : ratio of latent space size and image size
        num_noncond_blocks      : num of blocks to go without conditioning
        filter_size             : filter size in affine coupling 
        """


        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_flow = num_flow
        self.num_block = num_block
        self.num_cond_chans = num_cond_chans
        self.filter_size = filter_size
        self.final_cond_dim = final_cond_dim
        self.first_split_ratio = first_split_ratio
        self.squeezer = getattr(model_tools, squeezer)(cond_shape[0])
        self.sparsity_weight = sparsity_weight
        self.sparsifier = sparsifier
        self.cutoff_dim = cutoff_dim
        self.cutoff_mul = cutoff_mul

        self.model = self.build_inn()
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet4(cond_shape=self.cond_shape, num_cond_chans=self.num_cond_chans, num_block=self.num_block, linear_layer_dim=self.final_cond_dim, squeezer=self.squeezer)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = f"CondGlow-nf{num_flow}-nb{num_block}-nc{num_cond_chans}-fil{filter_size}-spl{first_split_ratio}"

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def build_inn(self):

        C = self.input_shape[0]
        filter_size = self.filter_size

        def subnet_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def subnet_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        def add_block(nodes, num_flow, subnet, condition, clamp=1.0, permutation='invconv'):
            for k in range(num_flow):
                nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                        {'subnet_constructor':subnet, 'clamp':clamp},
                                        conditions=condition))
                # nodes.append(Ff.Node(nodes[-1], model_tools.ActNormCoarse, {'inverse': True, 'scaling': 1.}))
                # nodes.append(Ff.Node(nodes[-1], model_tools.ActNorm2, {'inverse': True, 'scaling': 1.}))
                if permutation=='random':
                    nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
                elif permutation=='invconv':
                    nodes.append(Ff.Node(nodes[-1], model_tools.Inv1x1Conv, {}))
            return nodes

        def split(nodes, split_nodes, section_sizes):
            nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':section_sizes, 'dim':0}))
            split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
            return nodes, split_nodes

        nodes = [Ff.InputNode(*self.input_shape)]
        conditions = []
        ncc = self.num_cond_chans
        chans = C
        cs = self.cond_shape[1] // 2
        for i in range(self.num_block - 1):
            conditions.append(Ff.ConditionNode(ncc, cs, cs))
            ncc *= 2; cs //= 2
        conditions.append(Ff.ConditionNode(self.final_cond_dim))

        split_nodes = []

        # Sparsifier
        # if self.sparsifier != None:
        #     nodes.append(Ff.Node(nodes[-1], model_tools.SparsifierHardThreshold,
        #     {
        #         'sparsifier' : self.sparsifier,
        #         'threshold'  : self.sparsity_weight,
        #         'epsilon'    : 1.e-03,
        #     }))
        nodes.append(Ff.Node(nodes[-1], model_tools.ActNormCoarse, {'inverse': True, 'scaling': 1./6}))

        if self.input_shape[0] == 3:
            nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
        else:
            nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
        subnet = subnet_conv(filter_size, 3)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[0], permutation='random')
        section_sizes = [ int(4*chans*self.first_split_ratio), int(4*chans*(1-self.first_split_ratio)) ]

        nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
        chans = int(4*chans*self.first_split_ratio)
        filter_size *= 2

        for bl in range(1,self.num_block-1):
            # nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))
            nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'order_by_wavelet': True, 'rebalance':0.5}))
            subnet = subnet_conv(filter_size, 3)
            nodes = add_block(nodes, self.num_flow, subnet, conditions[bl], clamp=0.6, permutation='random')
            section_sizes = [ 2*chans, 2*chans ]
            nodes, split_nodes = split(nodes, split_nodes, section_sizes=section_sizes)
            chans *= 2
            filter_size *= 2
            
        # fully connected part
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        subnet = subnet_fc(filter_size)
        nodes = add_block(nodes, self.num_flow, subnet, conditions[-1], clamp=0.6, permutation='random')

        # concat the split nodes
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)

    def change_sparsity_weight(self, new_weight):
        self.model.module_list[0].threshold = new_weight

    # def initialize_actnorm(self, data_batch, degradation_op):
    #     with torch.no_grad():
    #         data_batch_rev = degradation_op.rev(data_batch)
    #         x = self.sample(data_batch.shape[0], data_batch_rev)
    #         std = np.std(x)
    #         self.model.module_list[0].final_scale *= 1/std/6
    #         self.model.module_list[0].final_scale.to(self.device)

    def initialize_actnorm(self, data_batch, degradation_op):
        for m in self.model.module_list:
            if isinstance(m, model_tools.ActNormCoarse):
                m.init_on_next_batch = True
        with torch.no_grad():
            data_batch_rev = degradation_op.rev(data_batch)
            x = self.sample(data_batch.shape[0], data_batch_rev)
            # After 1 pass through self.model, init_on_next_batch on all the actnorms automatically is set to False again.

    def forward(self, input, cond, rev=False):

        if not rev:
            z, jac = self.model([input], self.cond_net(cond), jac=True)
            return z, jac

        else:
            x_posts, jac_rev = self.model(input, self.cond_net(cond), rev=True)
            return x_posts, jac_rev

    def sample(self, num_samples, cond, temp=1., cutoff_dim=None, cutoff_mul=None):
        if cutoff_dim == None: cutoff_dim = self.cutoff_dim
        if cutoff_mul == None: cutoff_mul = self.cutoff_mul
        # z = torch.randn(num_samples, np.prod(self.input_shape)).to(self.device) * temp
        z = self.latent_dist.sample([num_samples, np.prod(self.input_shape)]) * temp
        cutoff_dim = int(np.prod(self.input_shape) * cutoff_dim)
        z[:, :cutoff_dim] *= cutoff_mul
        if cond.shape[0] == 1:
            cond = torch.cat([cond]*num_samples)
        else: assert cond.shape[0] == num_samples, "batch size of condition must be equal to 1 or num_samples"
        x,_ = self.model(z, self.cond_net(cond), rev=True)
        x = x.cpu().detach().numpy()
        return x

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False, importance_weighting=False, cutoff_dim=None, cutoff_mul=None):
        if cutoff_dim == None: cutoff_dim = self.cutoff_dim
        if cutoff_mul == None: cutoff_mul = self.cutoff_mul 
        return get_conditional_loss_freia(self, data_batch, degradation_op, num_z=num_z, reg_parameter=reg_parameter, num_bits=num_bits, tiled=tiled, importance_weighting=importance_weighting, cutoff_dim=cutoff_dim, cutoff_mul=cutoff_mul)

    def save(self, *args, **kwargs):
        # torch.save(self.model.state_dict(), *args, **kwargs)
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict, strict=False)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        input_shape: list, 
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        self.trainable_parameters = self.parameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


if __name__ == '__main__':

    import degradations
    import scipy.misc as misc
    from skimage.transform import resize

    input_shape = [3,64,64]
    degradation = degradations.GaussianBlurNoise(input_shape=input_shape, kernel_sigma=1.5)
    # degradation = degradations.FourierSamplerGaussianNoise(input_shape=input_shape, random=False, mask_file='../masks_mri/cartesian_4fold_64.npy')

    # sparsifier = sparsifiers.SparsifyingTransform(input_shape)
    sparsifier = sparsifiers.WaveletTransform([1,64,64], level=2)

    # post_model_args = {'num_flow': 2, 'num_block': 4, 'num_cond_chans': 16, 'compression_ratio': 1, 'num_noncond_blocks': 1, 'gamma': 1e-08, 'filter_size': 64}
    # post_model_args = {'num_flow': 2, 'num_block': 4, 'num_cond_chans': 16, 'filter_size': 32, 'final_cond_dim': 512, 'first_split_ratio': 0.25, 'squeezer': 'IRevNetSqueeze'}
    post_model_args = {'num_flow': 4, 'num_block': 4, 'num_cond_chans': 32, 'filter_size': 32, 'final_cond_dim': 512, 'first_split_ratio': 0.25, 'squeezer': 'HaarSqueeze'}
    # post_model_args = {'num_flow': 4, 'num_block': 4, 'num_cond_chans': 32, 'filter_size': 32, 'final_cond_dim': 512, 'first_split_ratio': 0.25, 'squeezer': 'HaarSqueeze'}
    post_model = CondGlow2(input_shape, cond_shape=degradation.output_shape, **post_model_args)
    post_model = post_model.to(post_model.device)

    xgt = np.load('xgt.npy')
    xgt = resize(xgt, (64,64))
    xgt = torch.Tensor(xgt.reshape(1,1,64,64))
    xgt = torch.cat([xgt]*4, axis=0)
    xgt = torch.cat([xgt]*3, axis=1)
    cond = degradation(xgt).to('cuda')
    x = post_model.sample(4, degradation.rev(cond[0:1]))
    print(x.shape)
    loss, x_posts = post_model.get_loss(cond, num_z=2, tiled=True, degradation_op=degradation)
    
    # squeeze = HaarSqueeze(1)
    # coeffs = squeeze(x_posts[1,0:1].detach().cpu()).numpy()
    # coeffs = np.squeeze(coeffs)
    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # temp = '/home/vak2/Documents/temp.png'
    # plt.hist(coeffs[3].flatten(), bins=np.linspace(-7,7,100));
    # plt.hist(coeffs[2].flatten(), bins=np.linspace(-7,7,100));
    # plt.hist(coeffs[1].flatten(), bins=np.linspace(-7,7,100));
    # plt.hist(coeffs[0].flatten(), bins=np.linspace(-7,7,100));
    # plt.savefig(temp);plt.close()
