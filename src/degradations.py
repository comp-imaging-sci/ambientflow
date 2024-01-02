""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import utils
import torchvision
import torchvision.transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GaussianNoise(object):
    def __init__(self, input_shape=[3,32,32], mean=0., std=0.2, num_bits=0, device=None):
        self.device = device if device != None else DEVICE
        self.std = std
        self.mean = mean
        self.num_bits = num_bits
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.rng = torch.Generator()
        
    def __call__(self, tensor):
        # if self.num_bits:
        #     t = tensor * (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1))
        #     t = t / (2**self.num_bits - 1)
        # else: t = tensor
        t = tensor
        # if seed != None:    self.rng.manual_seed(seed)
        return t + torch.randn(*tensor.shape, generator=self.rng) * self.std + self.mean

    def rev(self, tensor, **kwargs):
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

    def log_prob(self, y, x):
        # if self.num_bits:
        #     t = x + 0.5
        #     t *= (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1).to(device))
        #     t = t.float()
        #     t /= (2**self.num_bits - 1)
        #     t -= 0.5
        # else: t = x
        # return utils.log_normal( y - t - self.mean, std=self.std)
        return utils.log_normal( y - x - self.mean, std=self.std)


class GaussianBlurNoise(object):
    def __init__(self, kernel_sigma=1.5, mean=0., std=0.2, input_shape=[3,28,28], num_bits=0, device=None):
        self.device = device if device != None else DEVICE
        self.kernel_sigma = kernel_sigma
        self.std = std
        self.mean = mean
        self.num_bits = num_bits
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.rng = torch.Generator()
        
        W = 2*int(4*kernel_sigma + 0.5) + 1
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=W, sigma=self.kernel_sigma)
        
    def __call__(self, tensor):
        # if self.num_bits:
        #     t = tensor * (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1))
        #     t = t / (2**self.num_bits - 1)
        # else: t = tensor
        t = tensor

        # Gaussian blur
        t = self.blur(t)
        return t + torch.randn(*tensor.shape, generator=self.rng) * self.std + self.mean

    def rev(self, tensor, **kwargs):
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_sigma={self.kernel_sigma}, mean={self.mean}, std={self.std}, input_shape={self.input_shape}, num_bits={self.num_bits})'

    def log_prob(self, y, x):
        # if self.num_bits:
        #     t = x + 0.5
        #     t *= (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1).to(device))
        #     t = t.float()
        #     t /= (2**self.num_bits - 1)
        #     t -= 0.5
        # else: t = x
        t = self.blur(x)
        return utils.log_normal( y - t - self.mean, std=self.std)


class BlockMask(object):
    def __init__(self, start_coord=[16,16], height=16, width=16, input_shape=[3,32,32], num_bits=0, device=None):
        self.device = device if device != None else DEVICE
        self.start_coord = start_coord
        self.height = height
        self.width = width
        self.num_bits = num_bits
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.rng = torch.Generator()
        self.mask = np.ones(input_shape)
        self.mask[:,
                self.start_coord[0]:self.start_coord[0]+self.height,
                self.start_coord[1]:self.start_coord[1]+self.width] = 0
        
    def __call__(self, tensor, **kwargs):
        # if self.num_bits:
        #     t = tensor * (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1))
        #     t = t / (2**self.num_bits - 1)
        # else: t = tensor
        t = tensor.clone()
        t[:,:,
          self.start_coord[0]:self.start_coord[0]+self.height,
          self.start_coord[1]:self.start_coord[1]+self.width] = 0
        return t
    
    def fwd_noiseless(self, tensor, **kwargs):
        return self(tensor)

    def rev(self, tensor, **kwargs):
        t = tensor.clone()
        t[:, 
          self.start_coord[0]:self.start_coord[0]+self.height,
          self.start_coord[1]:self.start_coord[1]+self.width] = 0
        return t
    
    def __repr__(self):
        return self.__class__.__name__ + f'(start_coord={self.start_coord}, height={self.height}, width={self.width}, input_shape={self.input_shape}, num_bits={self.num_bits})'

    def log_prob(self, y, x):
        # if self.num_bits:
        #     t = x + 0.5
        #     t *= (2**self.num_bits - 1)
        #     t = torch.bucketize(t, boundaries=torch.arange(2**self.num_bits + 1).to(device))
        #     t = t.float()
        #     t /= (2**self.num_bits - 1)
        #     t -= 0.5
        # else: t = x
        t = self(x)
        return utils.log_normal(y - t, std=1/255)
    

class FourierSamplerGaussianNoise(object):
    def __init__(self, random=False, mask_file=None, samp=1, center_fraction=0.1, mean=0, std=0.1, input_shape=[1,128,128], num_bits=0, device=None):
        """
        random          : Instead of loading from file, generate random masks. (A different mask for each measurement)
        mask_file       : File path to load the undersampling mask from. Located in `../masks_mri/`
        samp            : Undersampling ratio. samp and center_fraction are only relevant if random=True. Else load mask from the mask file or default to fully sampled FFT.
        center_fraction : Fully sampled FFT fraction near the 0 frequency. Between 0-1. Must be smaller than 1/samp.
        mean            : Mean of the noise
        std             : Standard deviation of the noise
        input_shape     :
        num_bits        :
        device          :
        """
        self.device = device if device != None else DEVICE
        self.samp = samp
        self.center_fraction = center_fraction
        self.mask_file = mask_file
        self.random = random
        self.mean = mean
        self.std = std
        self.num_bits = num_bits
        self.input_shape = input_shape
        self.output_shape = [2*input_shape[0], input_shape[1], input_shape[2]]
        self.rng = torch.Generator()
        
        if self.mask_file != None:
            self.mask = torch.Tensor(np.load(self.mask_file))
        else:
            self.mask = torch.Tensor(np.ones(input_shape))

        self.mask_device = self.mask.to(self.device)
        
    def __call__(self, tensor, use_device=False):

        t = self.fwd_noiseless(tensor, use_device=use_device)
        mask = self.mask_device if use_device else self.mask
        noise = torch.randn(*tensor.shape, dtype=torch.cfloat, generator=self.rng) * self.std / np.sqrt(2) + self.mean
        if use_device: noise = noise.to(self.device)
        t = t + noise 
        return t * mask

    def fwd_noiseless(self, tensor, use_device=False):

        # masked fft
        if self.random:
            L = self.mask.shape[-1]
            num_low_freqs = int(self.center_fraction*L)
            num_high_freqs = L - num_low_freqs
            prob = (L / self.samp - num_low_freqs) / (L - num_low_freqs)
            probs = prob * torch.ones([num_high_freqs])
            samp_values = torch.bernoulli(probs, generator=self.rng)
            self.mask[:,:,:num_high_freqs//2] = samp_values[:num_high_freqs//2]
            self.mask[:,:,num_high_freqs//2+num_low_freqs:] = samp_values[num_high_freqs//2:]
            self.mask[:,:,num_high_freqs//2:num_high_freqs//2+num_low_freqs] = 1
            # self.mask = self.mask[0].T.unsqueeze(0)
            self.mask_device = self.mask.to(self.device)

        mask = self.mask_device if use_device else self.mask
        t = torch.fft.fft2(tensor, norm="ortho")
        t = torch.fft.fftshift(t, dim=[-2,-1]) * mask
        return t

    def rev(self, meas, mode="twochannel", use_device=True):

        t = meas * self.mask_device if use_device else meas * self.mask
        t = torch.fft.ifftshift(t, dim=[-2,-1])
        t = torch.fft.ifft2(t, norm="ortho")
        if mode == "complex":
            return t
        if mode == "twochannel":
            return torch.concat([ t.real, t.imag ], axis=-3)
        elif mode == "real":
            return t.real
        return t

    def __repr__(self):
        return self.__class__.__name__ + f'(samp={self.samp}, mask_file={self.mask_file}, mean={self.mean}, std={self.std}, input_shape={self.input_shape}, num_bits={self.num_bits})'

    def log_prob(self, y, x):

        t = torch.fft.fft2(x, norm="ortho")
        t = torch.fft.fftshift(t, dim=[-2,-1]) * self.mask_device
        return utils.log_normal( y - t - self.mean, std=self.std)
    

if __name__ == '__main__':

    import imageio as io

    fwd = GaussianNoise([3,64,64], std=0.2)
    # fwd = FourierSamplerGaussianNoise(samp=4, mask_file='../masks_mri/cartesian_4fold_128.npy')
    # fwd = FourierSamplerGaussianNoise(random=True, mask_file=None, samp=2, center_fraction=0.2)
    # x = np.load("xgt.npy")
    # x = x.reshape(1,1,128,128)
    # x = torch.Tensor(x)
    # y = fwd(x)
    # mask = fwd.mask
    
