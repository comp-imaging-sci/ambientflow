""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
import imageio as io
from skimage.transform import resize as skresize
import os
import glob


def discretize(t, num_bits):
    t = t * (2**num_bits - 1)
    t = torch.bucketize(t, boundaries=torch.arange(2**num_bits + 1))
    t = t / (2**num_bits - 1)
    return t


class MNISTDataset(torchvision.datasets.MNIST):

    def __init__(self, ambient=False, degradation=None, train=True, power_of_two=False, num_bits=0, data_root_path='/home/vak2/Documents/ambientflow/data',  **kwargs):

        transform = [
            transforms.ToTensor(),
            transforms.Lambda( lambda x : torch.cat([x,x,x],0) ),
        ]
        if power_of_two:
            transform.append(transforms.Pad((32-28)//2))
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
        if ambient:
            transform.append(degradation)
        transform.append(transforms.Lambda( lambda x : x - 0.5 ))
        transform = transforms.Compose(transform)

        super(MNISTDataset, self).__init__(root=data_root_path, train=train, transform=transform)

class CelebADataset(torchvision.datasets.CelebA):

    def __init__(self, ambient=False, degradation=None, train=True, input_shape=[3,128,128], num_bits=0, data_root_path='/shared/anastasio1/Other/ambientflow/data', **kwargs):

        if train == True:   split = 'train'
        else:               split = 'test'

        def standardize(x): return x - 0.5
        transform = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(input_shape[1:]),
            transforms.RandomHorizontalFlip(),
        ]
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
        if ambient:
            transform.append(degradation)
        transform.append(transforms.Lambda(standardize))
        transform = transforms.Compose(transform)

        super(CelebADataset, self).__init__(root=data_root_path, split=split, transform=transform)

class CelebAHQDataset(torchvision.datasets.VisionDataset):

    def __init__(self, ambient=False, degradation=None, train=True, input_shape=[3,128,128], num_bits=0, on_the_fly=True, data_root_path='/shared/anastasio1/Other/ambientflow/data', augmentation='', flip=True, **kwargs):

        if train == True:   split = 'train'
        else:               split = 'test'

        self.augmentation = augmentation

        # def standardize(x): return x - 0.5
        transform = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(input_shape[1:], antialias=True),
        ]
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
        if ambient:
            transform.append(degradation)
        if flip:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.Lambda(lambda x: x - 0.5))
        transform = transforms.Compose(transform)

        super(CelebAHQDataset, self).__init__(root=data_root_path, transform=transform, target_transform=None)

        self.base_folder = 'CelebAMask-HQ'
        if split == 'train':
            self.image_filenames = sorted(glob.glob(
                os.path.join(self.root, self.base_folder, 'CelebA-HQ-img', '*.jpg' )
            ))[:-500]
        elif split == 'test':
            self.image_filenames = sorted(glob.glob(
                os.path.join(self.root, self.base_folder, 'CelebA-HQ-img', '*.jpg' )
            ))[-500:]

        if ambient and (not on_the_fly):
            self.noise_seeds = np.fromfile('/home/vak2/system_seeds.dat', np.int32)
        self.ambient = ambient
        self.on_the_fly = on_the_fly
        self.degradation = degradation

    def __getitem__(self, index: int):

        X = io.imread(self.image_filenames[index])
        # X = np.swapaxes(X.T, 1,2)
        if self.ambient and (not self.on_the_fly):
            self.degradation.rng.manual_seed(int(self.noise_seeds[index]))

        if self.transform is not None:
            X = self.transform(X)

        X,label = self.augment(X)

        return X, label  # label is used to inform about the kind of augmentation. It will be used to reverse the augmentation when needed.

    def augment(self, X):
        if 'vf' in self.augmentation:
            label = np.random.randint(2)
            if label==1: X = torch.flip(X, dims=[1])
        else: label = 0

        return X, label

    def deaugment(self, X_batch, labels):
        X_batch_out = X_batch.clone()
        if 'vf' in self.augmentation:
            X_batch_out[...,labels==1,:,:,:] = torch.flip(X_batch[...,labels==1,:,:,:], dims=[len(X_batch.shape)-2,])
        return X_batch_out, labels

    def __len__(self):
        return len(self.image_filenames)


class FastMRIT2Dataset(torchvision.datasets.VisionDataset):

    def __init__(self, ambient=False, degradation=None, train=True, input_shape=[1,128,128], num_bits=0, volume_norm=True, on_the_fly=True, data_root_path='/shared/radon/SOMS/MRI_data', **kwargs):

        if train == True:   split = 'train'
        else:               split = 'test'

        # def standardize(x): return x - 0.5
        transform = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(input_shape[1:], antialias=True),
        ]
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
        transform.append(transforms.Lambda(lambda x: x - 0.5))
        if ambient:
            transform.append(degradation)
        transform = transforms.Compose(transform)

        super(FastMRIT2Dataset, self).__init__(root=data_root_path, transform=transform, target_transform=None)

        self.base_folder = 'multicoil_AXT2_3T_volumes'
        fnames = glob.glob(
            os.path.join(self.root, self.base_folder, 'vol_*.npy' )
        )
        fnames = sorted(fnames, key=lambda x: int(os.path.basename(x).strip('vol_').strip('.npy')))
        if split == 'train':
            self.fnames = fnames[:-20]
        elif split == 'test':
            self.fnames = fnames[-20:]

        self.volume_norm = volume_norm
        self.volume_max = np.load(os.path.join(self.root, self.base_folder+'_max_values.npy'))    

        if ambient and (not on_the_fly):
            self.noise_seeds = np.fromfile('/home/vak2/system_seeds.dat', np.int32)
        self.ambient = ambient
        self.on_the_fly = on_the_fly
        self.degradation = degradation

    def __getitem__(self, index: int):

        idx = index % self.__len__()
        slice_idx = np.random.randint(10)
        flip = np.random.randint(2)
        fname = self.fnames[idx]
        X = np.load(fname, mmap_mode='r')[slice_idx]
        X = X / self.volume_max[idx] if self.volume_norm else X / X.max()
        if flip: X = X[:,::-1]
        X = X[::-1].copy()

        if self.ambient and (not self.on_the_fly):
            self.degradation.rng.manual_seed(int(self.noise_seeds[20*idx + 2*slice_idx + flip]))

        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def get_all_noiseless(self):
        data = []
        for i,f in enumerate(self.fnames):
            print(i)
            X = np.load(f, mmap_mode='r')[:10]
            X = X / self.volume_max[i] if self.volume_norm else X / X.max()

            X = np.swapaxes(X, 0, 1)
            X = np.swapaxes(X, 1, 2)
            if self.transform is not None:
                X = self.transform(X)
            data.append(X)

        return data

    def augment(self, X):
        return X, 0

    def deaugment(self, X, labels):
        return X, labels

    def __len__(self):
        return len(self.fnames)


class EllipsesDataset(torchvision.datasets.VisionDataset):

    def __init__(self, ambient=False, degradation=None, train=True, input_shape=[1,32,32], num_bits=0, on_the_fly=True, data_root_path='/shared/radon/SOMS/ellipses', **kwargs):

        if train == True:   split = 'train'
        else:               split = 'test'

        # def standardize(x): return x - 0.5
        transform = [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(input_shape[1:], antialias=True),
        ]
        if num_bits:
            transform.append(
                transforms.Lambda( lambda x : discretize(x, num_bits) )
            )
        transform.append(transforms.Lambda(lambda x: x - 0.5))
        if ambient:
            transform.append(degradation)
        transform = transforms.Compose(transform)

        super(EllipsesDataset, self).__init__(root=data_root_path, transform=transform, target_transform=None)

        self.data = np.load(os.path.join(data_root_path, 'ellipses.npy'), mmap_mode='r')
        if split == 'train':
            self.data = self.data[:-500]
        if split == 'test':
            self.data = self.data[-500:]

        if ambient and (not on_the_fly):
            self.noise_seeds = np.fromfile('/home/vak2/system_seeds.dat', np.int32)
        self.ambient = ambient
        self.on_the_fly = on_the_fly
        self.degradation = degradation

    def __getitem__(self, index: int):

        flip_x = np.random.randint(2)
        flip_y = np.random.randint(2)
        X = self.data[index]
        if flip_x: X = X[::-1]
        if flip_y: X = X[:,::-1]
        X = X.copy()

        if self.ambient and (not self.on_the_fly):
            self.degradation.rng.manual_seed(int(self.noise_seeds[4*index + 2*flip_x + flip_y]))

        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    dset = MNISTDataset(train=True, power_of_two=True)
    # dset = CelebAHQDataset(train=True, ambient=True, input_shape=[3,64,64], on_the_fly=False)
    # dset = FastMRIT2Dataset(train=True, ambient=False, volume_norm=True, on_the_fly=False)
    # dset = EllipsesDataset(ambient=False)