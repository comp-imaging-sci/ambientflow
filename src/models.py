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
from model_tools import *
import torch.distributions as dists
# from tqdm import tqdm

import FrEIA.framework as Ff
import FrEIA.modules as Fm

torch.manual_seed(1234)
np.random.seed(42)

# from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
# from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                        nn.Linear(512,  c_out))

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   1), nn.ReLU(),
                        nn.Conv2d(256,  c_out, 1))

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

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False):
        
        batch_size = len(data_batch)
        n_pixel = np.prod(self.input_shape)
        num_bins = 2 ** num_bits

        if not tiled:
            x_posts = []
            jac_rev = torch.zeros(size=(batch_size,)).to(self.device)
            data_fidelity_prob = torch.zeros(size=(batch_size,)).to(self.device)
            for i in range(num_z):
                # z = torch.randn(batch_size, np.prod(self.input_shape)).to(self.device)
                z = self.latent_dist.sample([batch_size, np.prod(self.input_shape)])
                x_post,jac_r = self.model(z, self.cond_net(data_batch), rev=True)
                x_posts.append(x_post)
                jac_rev += jac_r/ num_z
                data_fidelity_prob += degradation_op.log_prob(data_batch, x_post) / num_z
            x_posts = torch.stack(x_posts, dim=0)
        else:
            # z = torch.randn(batch_size*num_z, np.prod(self.input_shape)).to(self.device)
            z = self.latent_dist.sample([batch_size*num_z, np.prod(self.input_shape)])
            data_batch_tiled = torch.cat([data_batch]*num_z, 0)
            x_posts, jac_rev = self.model(z, self.cond_net(data_batch_tiled), rev=True)
            data_fidelity_prob = degradation_op.log_prob(data_batch, x_posts)
            x_posts = torch.reshape(x_posts, [num_z, batch_size, *self.input_shape])

        loss = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
        loss = - torch.mean(loss / np.log(2) / n_pixel )
        return loss, x_posts

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

    def get_loss(self, data_batch, num_bits=0):

        npixel = np.prod(self.input_shape)
        if num_bits == 0:
            _,log_p,logdet = self(data_batch.to(self.device))
            loss = log_p + logdet
        else:
            num_bins = 2 ** num_bits
            dbatch = data_batch + torch.rand_like(data_batch) / num_bins
            _,log_p,logdet = self(dbatch.to(self.device))
            loss = log_p + logdet - np.log(num_bins) * npixel
        return - torch.mean( loss / np.log(2) / npixel )

    def save(self, *args, **kwargs):
        torch.save(self.state_dict(), *args, **kwargs)

    def load(self, path_to_pkl):
        state_dict = torch.load(path_to_pkl)
        self.load_state_dict(state_dict)


class CondNet2(nn.Module):
    '''conditioning network'''
    def __init__(self, cond_shape, cond_layer_thicknesses=[64, 128, 128, 512]):
        super().__init__()

        ct = cond_layer_thicknesses

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(3,  ct[0], 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(ct[0], ct[0], 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(ct[0], ct[1], 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(ct[1], ct[1], 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(ct[1], ct[2], 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(4),
                                         Flatten(),
                                         nn.Linear(ct[2] * cond_shape[1] // 16 * cond_shape[2] // 16, ct[3]))])

    def forward(self, c):
        outputs = [c]
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
        device                  = None):

        super().__init__()

        self.device = device if device != None else DEVICE
        self.input_shape = input_shape
        self.cond_shape = cond_shape
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.cond_layer_thicknesses = cond_layer_thicknesses
        # nodes = [Ff.InputNode(*input_shape, name='cond_input')]
        # ndim_x = np.prod(input_shape)

        self.model = self.build_inn().to(device)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters: p.data = 0.02 * torch.randn_like(p)
        self.cond_net = CondNet2(cond_shape=self.cond_shape, cond_layer_thicknesses=cond_layer_thicknesses).to(self.device)
        self.trainable_parameters += list(self.cond_net.parameters())
        self.identifier = "CondConvINN2-" + '-'.join(['{}']*(len(num_fc_layers) + len(num_conv_layers))).format(*num_conv_layers, *num_fc_layers)

        # define the latent variable distribution 
        self.latent_dist = dists.normal.Normal(loc=torch.tensor(0, dtype=torch.float32).to(self.device), scale=torch.tensor(1, dtype=torch.float32).to(self.device))

    def build_inn(self):

        input_shape = self.input_shape
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
        conditions = [Ff.ConditionNode(ct[0], input_shape[1], input_shape[2]),
                      Ff.ConditionNode(ct[1], input_shape[1] // 2, input_shape[2] // 2),
                      Ff.ConditionNode(ct[2], input_shape[1] // 4, input_shape[1] // 4),
                      Ff.ConditionNode(ct[3])]

        split_nodes = []

        subnet = sub_conv(32, 3)
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
        section_sizes = [ input_shape[0], 3*input_shape[0] ]
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
        section_sizes = [2*input_shape[0], 2*input_shape[0]]
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

    def get_loss(self, data_batch, degradation_op, num_z=1, reg_parameter=1, num_bits=0, tiled=False):
        
        batch_size = len(data_batch)
        n_pixel = np.prod(self.input_shape)
        num_bins = 2 ** num_bits

        if not tiled:
            x_posts = []
            jac_rev = torch.zeros(size=(batch_size,)).to(self.device)
            data_fidelity_prob = torch.zeros(size=(batch_size,)).to(self.device)
            zs = self.latent_dist.sample([batch_size*num_z, np.prod(self.input_shape)])
            for i in range(num_z):
                # z = torch.randn(batch_size, np.prod(self.input_shape)).to(self.device)
                z = zs[i*batch_size:(i+1)*batch_size]
                x_post,jac_r = self.model(z, self.cond_net(degradation_op.rev(data_batch)), rev=True)
                x_posts.append(x_post)
                jac_rev += jac_r/ num_z
                data_fidelity_prob_single = degradation_op.log_prob(data_batch, x_post)
                data_fidelity_prob += data_fidelity_prob_single / num_z
            x_posts = torch.stack(x_posts, dim=0)
        else:
            # z = torch.randn(batch_size*num_z, np.prod(self.input_shape)).to(self.device)
            z = self.latent_dist.sample([batch_size*num_z, np.prod(self.input_shape)])
            data_batch_tiled = torch.cat([data_batch]*num_z, 0)
            x_posts, jac_rev = self.model(z, self.cond_net(degradation_op.rev(data_batch_tiled)), rev=True)
            data_fidelity_prob = degradation_op.log_prob(data_batch_tiled, x_posts)
            x_posts = torch.reshape(x_posts, [num_z, batch_size, *self.input_shape])
            jac_rev = jac_rev.reshape(num_z, batch_size)
            jac_rev = torch.mean(jac_rev, axis=0)
            data_fidelity_prob = data_fidelity_prob.reshape(num_z, batch_size)
            data_fidelity_prob = torch.mean(data_fidelity_prob, axis=0)

        loss = reg_parameter*data_fidelity_prob + jac_rev - np.log(num_bins) * n_pixel
        loss = - torch.mean(loss / np.log(2) / n_pixel )
        return loss, x_posts

    def save(self, *args, **kwargs):
        torch.save(self.model.state_dict(), *args, **kwargs)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234-241.
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

    degradation = degradations.GaussianBlurNoise(input_shape=[3,64,64])
    cond_net = CondConvINN2()
    test_image = misc.face()
    test_image = misc.imresize(test_image, [64,64,3])
    test_image = np.swapaxes(test_image.T, 1,2)
    test_image = np.stack([test_image, test_image[::-1], test_image[:,::-1]])
    loss = cond_net.get_loss(test_image, degradation, num_z=4, reg_parameter=1)
    loss_tiled = cond_net.get_loss(test_image, degradation, num_z=4, reg_parameter=1, tiled=True)
    print(loss, loss_tiled)

