""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
import ptwt
import wavelet_utils as wtutils

class SparsifyingTransform(object):

    def __init__(self, input_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.fixed_subspace = np.zeros((np.prod(self.input_shape)), dtype=bool)

    def __call__(self, input):
        return input.reshape(input.shape[0], -1)

    def inv(self, output):
        return output.reshape(output.shape[0], *self.input_shape)

    def split_leading_coeffs(self, input, k):

        assert k >= np.sum(self.fixed_subspace), "Number of significant coeffs k is smaller than the mandatory subspace dimension"

        if isinstance(input, np.ndarray):
            print("Operating on NumPy array")
            module = np
        else: module = torch

        # get locations of leading k coeffs in each sample
        coeffs = self(input) # N x n
        if module == torch:
            coeffs_mod = coeffs.detach().clone()
        else:
            coeffs_mod = coeffs.copy()
        coeffs_mod[:,self.fixed_subspace] = 0
        sorting_coeff_idxs = module.argsort(module.abs(coeffs_mod))
        sorting_coeff_idxs = module.flip(sorting_coeff_idxs, [-1])
        leading_coeff_locs = module.argsort(sorting_coeff_idxs) < (k - np.sum(self.fixed_subspace))
        leading_coeff_locs[:,self.fixed_subspace] = True
        if module == torch: 
            leading_coeff_locs = leading_coeff_locs.detach()

        leading_coeffs = coeffs * leading_coeff_locs
        trailing_coeffs = coeffs * (~leading_coeff_locs)

        return leading_coeffs, trailing_coeffs

    def split_leading_components(self, input, k):
        leading_coeffs, trailing_coeffs = self.split_leading_coeffs(input, k)
        return self.inv(leading_coeffs), self.inv(trailing_coeffs)


class WaveletTransform(SparsifyingTransform):

    def __init__(self, input_shape, wavelet='haar', level=2, fixed_subspace_ratio=1/16, mode='zero', **kwargs):

        super().__init__(input_shape)
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level
        self.mode = mode
        self.fixed_subspace = np.zeros((np.prod(self.input_shape)), dtype=bool)
        self.fixed_subspace[:int( len(self.fixed_subspace) * fixed_subspace_ratio )] = True

    def __call__(self, input):
        if isinstance(input, np.ndarray):
            coeffs = pywt.wavedec2(input, self.wavelet, level=self.level, mode=self.mode)
        elif isinstance(input, torch.Tensor):
            coeffs = ptwt.wavedec2(input, self.wavelet, level=self.level, mode=self.mode)

        c, shapes = wtutils.coeffs_to_array(coeffs)
        self.shapes = shapes
        return c

    def inv(self, output, shapes=None):
        if shapes == None: shapes = self.shapes
        coeffs = wtutils.array_to_coeffs(output, shapes)

        if isinstance(output, np.ndarray):
            img = pywt.waverec2(coeffs, self.wavelet)
        elif isinstance(output, torch.Tensor):
            img = ptwt.waverec2(coeffs, self.wavelet)
        return img

class GradientTransform(SparsifyingTransform):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        C,H,W = input_shape
        self.fixed_subspace = np.zeros((2*C,H,W), dtype=bool)
        self.fixed_subspace[:C,0,:] = True
        self.fixed_subspace[C:,:,0] = True
        self.fixed_subspace = self.fixed_subspace.flatten()

    def __call__(self, input):
        if isinstance(input, np.ndarray):   module = np
        elif isinstance(input, torch.Tensor):  module = torch
        input_x = module.roll(input, 1, 2)
        input_x[:,:,0] = 0
        input_y = module.roll(input, 1, 3)
        input_y[:,:,:,0] = 0
        input_dx = input - input_x
        input_dy = input - input_y

        if isinstance(input, np.ndarray):
            coeffs = np.concatenate([input_dx, input_dy], 1)
        else:
            coeffs = torch.cat([input_dx, input_dy], dim=1)
        # return coeffs.reshape(coeffs.shape[0], -1)
        return coeffs
    
    def inv(self, output):
        if isinstance(output, np.ndarray):   module = np
        elif isinstance(output, torch.Tensor):  module = torch
        C,H,W = self.input_shape
        # o = output.reshape(-1,2*C,H,W)
        o = output
        return 0.5 * ( module.cumsum(o[:,:C], 2) + module.cumsum(o[:,C:], 3) )


class SparsifyingRegularizer(nn.Module):
    """ This is a module which takes as input an image and a sparsifying transform, and outputs a regularizer
    """
    def __init__(self, input_shape, sparsifying_transform, num_leading_coeffs, 
    penalty_norm                = 1,
    weight_norm                 = 1,
    ):
        """
        input_shape                 : Shape of the input image
        sparsifying_transform       : Sparsifying transform used.
        num_leading_coeffs          : Number of leading coeffs, should be a divisor of prod(input_shape)
        conv_type_switch_thresh     : Threshold for the number of channels for switching from regular convolution to group convolution
        penalty_norm
        weight_norm 
        """
        super().__init__()

        self.input_shape = input_shape
        self.sparsifying_transform = sparsifying_transform
        self.num_leading_coeffs = num_leading_coeffs
        self.penalty_norm = penalty_norm
        self.weight_norm = weight_norm

        final_num_channels = np.prod(input_shape)
        C,H,W = input_shape
        ch_in = C
        layers = []
        while ch_in*16 <= C*H*W:
            groups = int(ch_in > C*16)*ch_in + int(ch_in <= C*16)
            layers.append(
                nn.Conv2d(ch_in, ch_in*16, 3, stride=2, padding=1, groups=groups), 
            )
            layers.append(nn.AvgPool2d(2))
            layers.append(nn.ReLU())
            ch_in *= 16

        groups = int(ch_in > C*16)*ch_in + int(ch_in <= C*16)
        if ch_in < C*H*W:
            layers.append(
                nn.Conv2d(ch_in, ch_in*4, 3, stride=2, padding=1, groups=groups),
            )
        else: layers = layers[:-1]
        
        self.weighting_net = nn.Sequential(*layers)
        self.trainable_parameters = list(self.weighting_net.parameters())

    def forward(self, input):
        _, trailing_coeffs = self.sparsifying_transform.split_leading_coeffs(
            input, self.num_leading_coeffs
        )
        coeff_weights = self.weighting_net(input)
        assert(coeff_weights.shape[1] == np.prod(self.input_shape)), "Improper number of channels for the coeff weights"
        assert(coeff_weights.shape[2] == coeff_weights.shape[3] == 1), "Improper dimensions of coeff weights"
        coeff_weights = coeff_weights.squeeze(3).squeeze(2)
        weight_norm = torch.norm(coeff_weights, p=self.weight_norm, dim=1, keepdim=True)
        coeff_weights = coeff_weights / weight_norm * (coeff_weights.shape[1])**(1/self.weight_norm)
        penalty = torch.norm(
            # trailing_coeffs / coeff_weights,
            trailing_coeffs,
            p   = self.penalty_norm, 
            dim = 1
        ) 
        return penalty / (trailing_coeffs.shape[1])**(1/self.penalty_norm)
        

def sparsifying_regularizer2(x, sp_trans : SparsifyingTransform, threshold_weight=0.1):

    # coeffs = coeffs + 0.5
    # t = int(threshold_weight*np.prod(x.shape[1:]))
    # _, coeffs = sp_trans.split_leading_coeffs(x, t)
    C = x.shape[1]
    n_pixel = np.prod(x.shape[1:])
    coeffs = sp_trans(x)
    coeff_abs = torch.abs(coeffs) # N*num_z x C*H*W
    # coeff_avg = threshold_weight * torch.mean(coeff_abs, dim=1, keepdim=True)
    zero_cond = (coeff_abs < threshold_weight)
    # zero_cond = torch.prod(zero_cond, dim=1, keepdim=True)
    zero_cond = zero_cond[:,0:C] * zero_cond[:,C:2*C]
    zero_cond = torch.cat([zero_cond]*2, dim=1)

    penalty = torch.norm(coeffs * zero_cond, p=1)
    # penalty = torch.norm(coeffs, p=1)
    return penalty / np.log(2) / n_pixel

def sparsifying_regularizer3(x, sp_trans : SparsifyingTransform, threshold_weight=0.1):

    # coeffs = coeffs + 0.5
    # t = int(threshold_weight*np.prod(x.shape[1:]))
    # _, coeffs = sp_trans.split_leading_coeffs(x, t)
    C = x.shape[1]
    n_pixel = np.prod(x.shape[1:])
    coeffs = sp_trans(x)
    coeff_abs = torch.abs(coeffs) # N*num_z x C*H*W
    coeff_sorted,_ = coeff_abs.flatten().sort()
    threshw = coeff_sorted[-int(len(coeff_sorted)*threshold_weight)-1]
    zero_cond = (coeff_abs < threshw)
    # zero_cond = torch.prod(zero_cond, dim=1, keepdim=True)
    zero_cond = zero_cond[:,0:C] * zero_cond[:,C:2*C]
    zero_cond = torch.cat([zero_cond]*2, dim=1)

    penalty = torch.norm(coeffs * zero_cond, p=1)
    # penalty = torch.norm(coeffs, p=1)
    return penalty / np.log(2) / n_pixel


if __name__ == '__main__':

    import imageio as io
    from skimage.transform import resize

    # img = np.load('xgt_s.npy')
    # img = np.swapaxes(img,0,2)
    # img = np.stack([img, img], axis=0)
    # img = img[:,0:1]
    # img = resize(img, [2,1,128,128])

    # input_shape = [1,128,128]

    # # T = SparsifyingTransform([3,64,64])
    # T = WaveletTransform(input_shape, wavelet='haar')
    # nlc = np.prod(input_shape) // 4
    # sparse_reg = SparsifyingRegularizer(input_shape, T, nlc)

    # imgt = torch.tensor(img, requires_grad=True, dtype=torch.float32)
    # penalty = sparse_reg(imgt)

    # img = np.load('xgt_ell.npy')
    input_shape = [1,128,128]
    # img = img.reshape(1,1,32,32)
    T = GradientTransform(input_shape)
    # T = WaveletTransform(input_shape)
    # coeffs = T(img)