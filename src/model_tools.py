""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from FrEIA.modules import InvertibleModule
from typing import Callable, Union
import sparsifiers

logabs = lambda x: torch.log(torch.abs(x))

def forward_squeeze(input):
    # input dim: NxCxHxW
    b_size, n_channel, height, width = input.shape
    squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
    squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
    out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
    return out


class Squeeze(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        # input dim: NxCxHxW
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        return out

    def reverse(self, input):
        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        out = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        return out


class IRevNetSqueeze(nn.Module):
    def __init__(self, in_channel, block_size=2):
        super().__init__()

        self.channels = in_channel
        self.block_size = block_size
        self.block_size_sq = self.block_size**2

        self.downsample_kernel = torch.zeros(4, 1, 2, 2)

        self.downsample_kernel[0, 0, 0, 0] = 1
        self.downsample_kernel[1, 0, 0, 1] = 1
        self.downsample_kernel[2, 0, 1, 0] = 1
        self.downsample_kernel[3, 0, 1, 1] = 1

        self.downsample_kernel = torch.cat([self.downsample_kernel] * self.channels, 0)
        self.downsample_kernel = nn.Parameter(self.downsample_kernel)
        self.downsample_kernel.requires_grad = False

    def forward(self, input):
        return F.conv2d(input, self.downsample_kernel, stride=self.block_size, groups=self.channels)

    def reverse(self, output):
        return F.conv_transpose2d(output, self.downsample_kernel,
                                    stride=self.block_size, groups=self.channels)


class HaarSqueeze(nn.Module):
    def __init__(self, in_channel, order_by_wavelet=True):

        super().__init__()
        self.in_channel = in_channel
        self.order_by_wavelet = order_by_wavelet

        self.haar_weights = torch.ones(4, 1, 2, 2) / 2

        self.haar_weights[1, 0, 0, 1] = -1/2
        self.haar_weights[1, 0, 1, 1] = -1/2

        self.haar_weights[2, 0, 1, 0] = -1/2
        self.haar_weights[2, 0, 1, 1] = -1/2

        self.haar_weights[3, 0, 1, 0] = -1/2
        self.haar_weights[3, 0, 0, 1] = -1/2

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channel, 0)

        if order_by_wavelet:
            permutation = []
            for i in range(4):
                permutation += [i + 4 * j for j in range(self.in_channel)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            # clever trick to invert a permutation
            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, input):
        out = F.conv2d(input, self.haar_weights, bias=None, stride=2, groups=self.in_channel)
        if self.order_by_wavelet:
            return out[:, self.perm]
        else: return out

    def reverse(self, output):
        if self.order_by_wavelet:
            x_perm = output[:, self.perm_inv]
        else:
            x_perm = output
        return F.conv_transpose2d(output, self.haar_weights, bias=None, stride=2, groups=self.in_channel)


class EVSCouplingBlock(_BaseCouplingBlock):
    ''' Coupling block that supports evidence-softmax based coupling to promote sparsity. 
    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 epsilon: float = 1.e-06,
                 avg_scale: float = 1.,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN",
                 split_len: Union[float, int] = 0.5):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. Two of these subnetworks will be
            initialized in the block.
          epsilon:
          avg_scale:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation,
                         split_len=split_len)
        self.epsilon = epsilon
        self.avg_scale = avg_scale

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1, u2, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = a2[:, :self.split_len1], a2[:, self.split_len1:]
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, :self.split_len2], a1[:, self.split_len2:]
        s1 = self.clamp * self.f_clamp(s1)
        # s1_idn = torch.abs(s1.detach()) - self.clamp # no gradients
        s1_idn = s1 - self.clamp 
        s1_avg = self.avg_scale * torch.mean(s1_idn, dim=tuple(range(1, self.ndims + 1)), keepdim=True)
        nzcond = (s1_idn <= s1_avg) + self.epsilon
        # nzcond = torch.sigmoid( -100*(s1_idn - s1_avg) )
        # ev_s1 = 1./( nzcond + self.epsilon ) * torch.exp(s1)
        # ev_s1_inv = ( nzcond + self.epsilon ) * torch.exp(-s1)
        ev_s1 = 1./( nzcond ) * torch.exp(s1)
        ev_s1_inv = ( nzcond ) * torch.exp(-s1)

        t1 = nzcond * t1
        # j2 = torch.sum(s1 * nzcond, dim=tuple(range(1, self.ndims + 1)))
        j2 = torch.sum(s1 - torch.log(nzcond), dim=tuple(range(1, self.ndims + 1))) 

        if rev:
            y2 = (x2 - t1) * ev_s1_inv
            return y2, -j2
        else:
            y2 = ev_s1 * x2 + t1
            return y2, j2


class SparsifierHardThreshold(InvertibleModule):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height dimensions.'''

    def __init__(self, dims_in, dims_c = None,
                 sparsifier = sparsifiers.SparsifyingTransform([1,64,64]),
                 threshold: float = 1.,
                 epsilon: float = 1.e-06):
        '''See docstring of base class (FrEIA.modules.InvertibleModule) for more.

        Args:
          sparsifier: sparsifying transform (such as orthogonal wavelet transform)
          ASSUMES THAT THE SPARSIFYING TRANSFORM IS ISOMETRIC (LOGDET IS 0).
          threshold: Threshold for hard thresholding
          
        '''
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][0]
        self.sparsifier = sparsifier
        self.threshold = threshold
        self.epsilon = epsilon
        self.final_scale = nn.Parameter(torch.ones((1,)), requires_grad=False)

        self.jac_fwd = 0.
        self.jac_rev = 0.

    def forward(self, x, c=None, jac=True, rev=False):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        clip_thresh = 0.51
        inp = x[0] #* self.final_scale # N x C x H x W 
        coeffs = self.sparsifier(inp + 0.5) # N x C*H*W

        #number total entries except for batch dimension:
        self.ndims = len(coeffs.shape) - 1

        coeff_abs = torch.abs(coeffs)
        coeff_avg = self.threshold * torch.mean(coeff_abs, dim=tuple(range(1, self.ndims + 1)), keepdim=True)
        nzcond = (coeff_abs >= coeff_avg) + self.epsilon

        # ASSUMES THAT THE SPARSIFIER IS ISOMETRIC
        # jac = -torch.cat([torch.log(self.final_scale)]*inp.shape[0])
        jac = 0
        # jac += -torch.sum(np.log(1/28) + torch.log((abs(inp) <= clip_thresh) + 1e-06), dim=(1,2,3))
        jac +=  -torch.sum(torch.log(nzcond), dim=tuple(range(1, self.ndims + 1))) 

        if not rev:
            raise ValueError("This function is only good for reverse computation, it can be highly unstable in the reverse computation.")
            # out = coeffs / nzcond
            # out = self.sparsifier.inv(out)
            # return (out,), jac
        else:
            out = coeffs * nzcond
            out = self.sparsifier.inv(out) - 0.5
            return (out,), -jac

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("HaarDownsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("HaarDownsampling can only transform 2D images"
                             "of the shape CxWxH (channels, width, height)")

        c, w, h = input_dims[0]

        return ((c, w, h),)

class Inv1x1Conv(InvertibleModule):
    '''Given an invertible matrix M, a 1x1 convolution is performed using M as
    the convolution kernel. Effectively, a matrix muplitplication along the
    channel dimension is performed in each pixel.'''

    def __init__(self, dims_in, dims_c=None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.
        '''
        super().__init__(dims_in, dims_c)

        # TODO: it should be possible to give conditioning instead of M, so that the condition
        # provides M and b on each forward pass.

        in_channel = dims_in[0][0]
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x, rev=False, jac=True):

        n_pixels = x[0][0, 0].numel()
        weight = self.calc_weight()

        j = n_pixels * torch.sum(self.w_s)

        if not rev:
            out = F.conv2d(x[0], weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            return (out,), -j
            # return (F.conv2d(x[0], weight),), j
        else:
            return (F.conv2d(x[0], weight),), j
            # out = F.conv2d(x[0], weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            # return (out,), -j

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError(f"{self.__class__.__name__} requires 3D input (channels, height, width)")
        return input_dims



class ActNormCoarse(InvertibleModule):
    '''A technique to achieve a stable initlization.

    First introduced in Kingma et al 2018: https://arxiv.org/abs/1807.03039
    The module is similar to a traditional batch normalization layer, but the
    data mean and standard deviation is only computed for the first batch of
    data. To ensure invertibility, the mean and standard devation are kept
    fixed from that point on.
    Using ActNorm layers interspersed throughout an INN ensures that
    intermediate outputs of the INN have standard deviation 1 and mean 0, so
    that the training is stable at the start, avoiding exploding or zeroed
    outputs.
    Just as with standard batch normalization layers, ActNorm contains
    additional channel-wise scaling and bias parameters.
    '''

    def __init__(self, dims_in, dims_c=None, init_data: Union[torch.Tensor, None] = None, inverse: bool = False, scaling: float = 1.):
        '''
        Args:
          init_data: If ``None``, use the first batch of data passed through this
            module to initialize the mean and standard deviation.
            If ``torch.Tensor``, use this as data to initialize instead of the
            first real batch.
        '''

        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in[0]
        self.inverse = inverse
        self.scaling = scaling
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

        if init_data:
            self._initialize_with_data(init_data)
        else:
            # self.init_on_next_batch = True
            self.init_on_next_batch = False

        def on_load_state_dict(*args):
            # when this module is loading state dict, we SHOULDN'T init with data,
            # because that will reset the trained parameters. Registering a hook
            # that disable this initialisation.
            self.init_on_next_batch = False
        self._register_load_state_dict_pre_hook(on_load_state_dict)

    def _initialize_with_data(self, data):
        # Initialize to mean 0 and std 1 with sample batch
        # 'data' expected to be of shape (batch, channels[, ...])
        # If actnorm is computed during a reverse computation, flip the direction of actnorm
        sign = (-1)**int(self.inverse)

        assert all([data.shape[i+1] == self.dims_in[i] for i in range(len(self.dims_in))]),\
            "Can't initialize ActNorm layer, provided data don't match input dimensions."
        self.scale.data.view(-1)[:] \
            = sign * torch.log(self.scaling / data.std())

        data = data * self.scale.exp()
        self.bias.data.view(-1)[:] \
            = sign * -data.mean()
        self.init_on_next_batch = False

    def forward(self, x, rev=False, jac=True):
        if self.init_on_next_batch:
            self._initialize_with_data(x[0])

        jac = (self.scale.sum() * np.prod(self.dims_in[1:])).repeat(x[0].shape[0])
        if rev:
            jac = -jac

        if not rev:
            return [x[0] * self.scale.exp() + self.bias], jac
        else:
            return [(x[0] - self.bias) / self.scale.exp()], jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class ActNorm2(InvertibleModule):
    '''A technique to achieve a stable initlization.

    First introduced in Kingma et al 2018: https://arxiv.org/abs/1807.03039
    The module is similar to a traditional batch normalization layer, but the
    data mean and standard deviation is only computed for the first batch of
    data. To ensure invertibility, the mean and standard devation are kept
    fixed from that point on.
    Using ActNorm layers interspersed throughout an INN ensures that
    intermediate outputs of the INN have standard deviation 1 and mean 0, so
    that the training is stable at the start, avoiding exploding or zeroed
    outputs.
    Just as with standard batch normalization layers, ActNorm contains
    additional channel-wise scaling and bias parameters.
    '''

    def __init__(self, dims_in, dims_c=None, init_data: Union[torch.Tensor, None] = None, inverse: bool = False, scaling: float = 1.):
        '''
        Args:
          init_data: If ``None``, use the first batch of data passed through this
            module to initialize the mean and standard deviation.
            If ``torch.Tensor``, use this as data to initialize instead of the
            first real batch.
        '''

        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in[0]
        self.inverse = inverse
        self.scaling = scaling
        self.scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

        if init_data:
            self._initialize_with_data(init_data)
        else:
            # self.init_on_next_batch = True
            self.init_on_next_batch = False

        def on_load_state_dict(*args):
            # when this module is loading state dict, we SHOULDN'T init with data,
            # because that will reset the trained parameters. Registering a hook
            # that disable this initialisation.
            self.init_on_next_batch = False
        self._register_load_state_dict_pre_hook(on_load_state_dict)

    def _initialize_with_data(self, data):
        # Initialize to mean 0 and std 1 with sample batch
        # 'data' expected to be of shape (batch, channels[, ...])
        # If actnorm is computed during a reverse computation, flip the direction of actnorm
        sign = (-1)**int(self.inverse)

        assert all([data.shape[i+1] == self.dims_in[i] for i in range(len(self.dims_in))]),\
            "Can't initialize ActNorm layer, provided data don't match input dimensions."
        self.scale.data.view(-1)[:] \
            = sign * torch.log(self.scaling / data.transpose(0,1).contiguous().view(self.dims_in[0], -1).std(dim=-1))

        data = data * self.scale.exp()
        self.bias.data.view(-1)[:] \
            = sign * -data.transpose(0,1).contiguous().view(self.dims_in[0], -1).mean(dim=-1)
        self.init_on_next_batch = False

    def forward(self, x, rev=False, jac=True):
        if self.init_on_next_batch:
            self._initialize_with_data(x[0])

        jac = (self.scale.sum() * np.prod(self.dims_in[1:])).repeat(x[0].shape[0])
        if rev:
            jac = -jac

        if not rev:
            return [x[0] * self.scale.exp() + self.bias], jac
        else:
            return [(x[0] - self.bias) / self.scale.exp()], jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims



class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output, jac=False):
        if jac:
            _, _, height, width = output.shape
            log_abs = logabs(self.scale)
            logdet = height * width * torch.sum(log_abs)
            return output / self.scale - self.loc, -logdet

        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output, jac=False):
        out = F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )
        if jac:
            _, _, height, width = output.shape
            logdet = (
                height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
            )
            return out, -logdet
        
        return out


class Inj2xConv2d(nn.Module):
    def __init__(self, in_channel, gamma=0.):
        super().__init__()

        self.gamma = gamma
        w1 = torch.randn(in_channel//2, in_channel//2)
        w2 = torch.randn(in_channel//2, in_channel//2)
        q1,_ = torch.linalg.qr(w1)
        q2,_ = torch.linalg.qr(w2)
        q = torch.concat([q1, q2], dim=1)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)
    
    def forward(self, input):
        _, _, height, width = input.shape

        self.w_gram = torch.matmul( self.weight.squeeze(), self.weight.squeeze().T )
        self.w_gram += self.gamma **2 * torch.eye(*self.w_gram.shape).to(self.w_gram.device)
        out = F.conv2d(input, self.weight)
        logdet = (
            0.5 * height * width * torch.slogdet(self.w_gram.double())[1].float()
        )

        return out, logdet

    def reverse(self, output, jac=False):

        self.w_gram = torch.matmul( self.weight.squeeze(), self.weight.squeeze().T )
        self.w_gram += self.gamma **2 * torch.eye(*self.w_gram.shape).to(self.w_gram.device)
        weight_inv = torch.matmul(self.weight.squeeze().T, self.w_gram.inverse()) # Moore-Penrose pseudoinverse
        out = F.conv2d(
            output, weight_inv.unsqueeze(2).unsqueeze(3)
        )

        if jac:
            _, _, height, width = output.shape
            logdet = (
                0.5 * height * width * torch.slogdet(self.w_gram.double())[1].float()
            )
            return out, -logdet
        
        return out            


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output, jac=False):
        weight = self.calc_weight()

        out = F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

        if jac:
            _, _, height, width = output.shape
            logdet = height * width * torch.sum(self.w_s)
            return out, -logdet
        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)

        return out

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True, condition=False, num_cond_layers=0, fully_connected=False):
        super().__init__()

        self.affine = affine
        self.condition = condition
        self.num_cond_layers = int(num_cond_layers * int(self.condition))

        if not fully_connected:
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2 + self.num_cond_layers, filter_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(filter_size, filter_size, 1),
                nn.ReLU(inplace=True),
                ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_channel // 2 + self.num_cond_layers, filter_size),
                nn.ReLU(inplace=True),
                nn.Linear(filter_size, filter_size),
                nn.ReLU(inplace=True),
                nn.ZeroLinear(filter_size, in_channel if self.affine else in_channel // 2),
            )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input, cond=None):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            if self.condition:
                log_s, t = self.net(
                    torch.concat([in_a, cond], dim=1)
                    ).chunk(2, 1)
            else:
                log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            if self.condition:
                net_out = self.net(
                    torch.concat([in_a, cond], dim=1)
                    )
            else:
                net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = 0

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output, cond=None, jac=False):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            if self.condition:
                log_s, t = self.net(
                    torch.concat([out_a, cond], dim=1)
                    ).chunk(2, 1)
            else:
                log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

            logdet = torch.sum(torch.log(s).view(output.shape[0], -1), 1)

        else:
            if self.condition:
                net_out = self.net(
                    torch.concat([out_a, cond], dim=1)
                    )
            else:
                net_out = self.net(out_a)
            in_b = out_b - net_out

            logdet = 0

        if jac:
            return torch.cat([out_a, in_b], 1), -logdet
        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, filter_size=512, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine, filter_size=filter_size)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        logdet = logdet + det2

        return out, logdet

    def reverse(self, output, jac=False):
        if jac:
            input, logdet = self.coupling.reverse(output, jac=jac)
            input, det1 = self.invconv.reverse(input, jac=jac)
            input, det2 = self.actnorm.reverse(input, jac=jac)
            logdet += (det1 + det2)
            return input, logdet

        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class CondFlow(nn.Module):
    def __init__(self, in_channel, affine=True, filter_size=512, num_cond_layers=32, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine, filter_size=filter_size, condition=True, num_cond_layers=num_cond_layers)

    def forward(self, input, cond):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out, cond)

        logdet = logdet + det1 + det2

        return out, logdet

    def reverse(self, output, cond, jac=False):
        if jac:
            input, logdet = self.coupling.reverse(output, cond, jac=jac)
            input, det1 = self.invconv.reverse(input, jac=jac)
            input, det2 = self.actnorm.reverse(input, jac=jac)
            logdet += (det1 + det2)
            return input, logdet

        input = self.coupling.reverse(output, cond)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class InjFlow(nn.Module):
    def __init__(self, in_channel, affine=True, filter_size=512, gamma=0.):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        self.injconv = Inj2xConv2d(in_channel, gamma=gamma)

        self.coupling = AffineCoupling(in_channel//2, affine=affine, filter_size=filter_size)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.injconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1 + det2

        return out, logdet

    def reverse(self, output, jac=False):
        if jac:
            input, logdet = self.coupling.reverse(output, jac=jac)
            input, det1 = self.injconv.reverse(input, jac=jac)
            input, det2 = self.actnorm.reverse(input, jac=jac)
            logdet += (det1 + det2)
            return input, logdet

        input = self.coupling.reverse(output)
        input = self.injconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class InvBlock(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, filter_size=512, conv_lu=True, extract_z=True, squeeze=True, squeezer=Squeeze):
        super().__init__()

        squeeze_dim = in_channel * 4 if squeeze else in_channel

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu, filter_size=filter_size))

        self.split = split
        self.extract_z = extract_z
        self.squeeze = squeeze
        if self.squeeze: self.squeezer = squeezer(in_channel=in_channel)

        if split:
            self.prior = ZeroConv2d(squeeze_dim // 2, squeeze_dim)

        elif (not split) and extract_z:
            self.prior = ZeroConv2d(squeeze_dim, squeeze_dim * 2)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        if self.squeeze:
            # squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            # squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            # out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            out = self.squeezer(input)
        else: out = input

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.extract_z:
            if self.split:
                out, z_new = out.chunk(2, 1)
                mean, log_sd = self.prior(out).chunk(2, 1)
                log_p = gaussian_log_p(z_new, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)

            else:
                zero = torch.zeros_like(out)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                log_p = gaussian_log_p(out, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
                z_new = out

            return out, logdet, log_p, z_new
        
        return out, logdet, 0, None

    def reverse(self, output, eps=None, reconstruct=False, jac=False):
        b_size, n_channel, height, width = output.shape
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

            if jac:
                mean = torch.zeros_like(eps)
                log_sd = torch.zeros_like(eps)
                log_p = gaussian_log_p(eps, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)

        else:
            if self.extract_z:
                if self.split:
                    mean, log_sd = self.prior(input).chunk(2, 1)
                    z = gaussian_sample(eps, mean, log_sd)
                    input = torch.cat([output, z], 1)
                else:
                    zero = torch.zeros_like(input)
                    # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                    mean, log_sd = self.prior(zero).chunk(2, 1)
                    z = gaussian_sample(eps, mean, log_sd)
                    input = z
                if jac:
                    log_p = gaussian_log_p(z, mean, log_sd)
                    log_p = -log_p.view(b_size, -1).sum(1)
            else:
                input = output
                log_p = 0

        logdet = 0
        for flow in self.flows[::-1]:
            if jac:
                input, det = flow.reverse(input, jac=True)
                logdet += det
            else:
                input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape
        if self.squeeze:
            # unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
            # unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
            # out = unsqueezed.contiguous().view(
            #     b_size, n_channel // 4, height * 2, width * 2
            # )
            out = self.squeezer.reverse(input)
        else: out = input

        if jac:
            return out, logdet, log_p
        return out


class InjBlock(nn.Module):
    def __init__(self, in_channel, n_flow=1, gamma=1e-08, affine=True, filter_size=512, conv_lu=True, extract_z=True, squeeze=True, squeezer=Squeeze):
        super().__init__()

        squeeze_dim = in_channel * 4 if squeeze else in_channel
        in_dim = squeeze_dim

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(InjFlow(in_dim, affine=affine, filter_size=filter_size, gamma=gamma))
            in_dim = in_dim // 2

        self.extract_z = extract_z
        self.squeeze = squeeze
        if self.squeeze: self.squeezer = squeezer(in_channel=in_channel)

        if extract_z:
            self.prior = ZeroConv2d(in_dim, in_dim * 2)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        if self.squeeze:
            # squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            # squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            # out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            out = self.squeezer(input)
        else: out = input

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.extract_z:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

            return out, logdet, log_p, z_new
        
        return out, logdet, 0, None

    def reverse(self, output, eps=None, reconstruct=False, jac=False):
        input = output
        b_size = input.shape[0]

        if reconstruct:
            input = eps
            if jac:
                mean = torch.zeros_like(eps)
                log_sd = torch.zeros_like(eps)
                log_p = gaussian_log_p(eps, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)

        else:
            if self.extract_z:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
                if jac:
                    log_p = gaussian_log_p(z, mean, log_sd)
                    log_p = -log_p.view(b_size, -1).sum(1)
            else:
                input = output
                log_p = 0

        logdet = 0
        for flow in self.flows[::-1]:
            if jac:
                input, det = flow.reverse(input, jac=True)
                logdet += det
            else:
                input = flow.reverse(input)

        _, n_channel, height, width = input.shape
        if self.squeeze:
            # unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
            # unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
            # out = unsqueezed.contiguous().view(
            #     b_size, n_channel // 4, height * 2, width * 2
            # )
            out = self.squeezer.reverse(input)
        else: out = input

        if jac:
            return out, logdet, log_p
        return out


class InvCondBlock(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, filter_size=512, num_cond_layers=32, conv_lu=True, extract_z=True, squeeze=True, squeezer=Squeeze):
        super().__init__()

        squeeze_dim = in_channel * 4 if squeeze else in_channel
        self.num_cond_layers = num_cond_layers * 4 if squeeze else num_cond_layers
        self.extract_z = extract_z
        self.squeeze = squeeze
        if self.squeeze: 
            self.squeezer = squeezer(in_channel=in_channel)
            self.cond_squeezer = squeezer(in_channel=num_cond_layers)

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(CondFlow(squeeze_dim, affine=affine, conv_lu=conv_lu, filter_size=filter_size, num_cond_layers=self.num_cond_layers))

        self.split = split

        if split:
            self.prior = ZeroConv2d(squeeze_dim // 2, squeeze_dim)

        elif (not split) and extract_z:
            self.prior = ZeroConv2d(squeeze_dim, squeeze_dim * 2)

    def forward(self, input, cond):
        b_size, n_channel, height, width = input.shape
        if self.squeeze:
            # squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
            # squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            # out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
            out = self.squeezer(input)
            cond = self.cond_squeezer(cond)
        else: out = input

        logdet = 0

        for flow in self.flows:
            out, det = flow(out, cond)
            logdet = logdet + det

        if self.extract_z:
            if self.split:
                out, z_new = out.chunk(2, 1)
                mean, log_sd = self.prior(out).chunk(2, 1)
                log_p = gaussian_log_p(z_new, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)

            else:
                zero = torch.zeros_like(out)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                log_p = gaussian_log_p(out, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)
                z_new = out

            return out, logdet, log_p, z_new

        return out, logdet, 0, None

    def reverse(self, output, cond, eps=None, reconstruct=False, jac=False):
        input = output
        b_size = input.shape[0]
        if self.squeeze:
            cond = self.cond_squeezer(cond)

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

            if jac:
                mean = torch.zeros_like(eps)
                log_sd = torch.zeros_like(eps)
                log_p = gaussian_log_p(eps, mean, log_sd)
                log_p = log_p.view(b_size, -1).sum(1)

        else:
            if self.extract_z:
                if self.split:
                    mean, log_sd = self.prior(input).chunk(2, 1)
                    z = gaussian_sample(eps, mean, log_sd)
                    input = torch.cat([output, z], 1)

                else:
                    zero = torch.zeros_like(input)
                    # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                    mean, log_sd = self.prior(zero).chunk(2, 1)
                    z = gaussian_sample(eps, mean, log_sd)
                    input = z
                if jac:
                    log_p = gaussian_log_p(z, mean, log_sd)
                    log_p = -log_p.view(b_size, -1).sum(1)
            else:
                input = output
                log_p = 0

        logdet = 0
        for flow in self.flows[::-1]:
            if jac:
                input, det = flow.reverse(input, cond, jac=True)
                logdet += det
            else:
                input = flow.reverse(input, cond)

        _, n_channel, height, width = input.shape
        if self.squeeze:
            # unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
            # unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
            # out = unsqueezed.contiguous().view(
            #     b_size, n_channel // 4, height * 2, width * 2
            # )
            out = self.squeezer.reverse(input)
        else: out = input

        if jac:
            return out, logdet, log_p
        return out

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

