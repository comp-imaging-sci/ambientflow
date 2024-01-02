""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.utils.data

verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }


def generate(labels, tot_dataset_size, fwd_model=None, noise=None, shuffle=True, seed=0, **kwargs):
    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(seed)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.1)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    if shuffle:
        shuffling = np.random.permutation(N)
    else: shuffling = np.arange(N)

    if fwd_model == None: pass

    if fwd_model == 'simple_slant_squish':
        R = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2)],
            [1/np.sqrt(2),  1/np.sqrt(2)]
        ])
        A = R.T @ np.diag([np.sqrt(3), 1/np.sqrt(3)]) @ R
        pos = (A @ pos.T).T

    if noise == None: pass

    elif noise == 'gaussian':
        pos_noise = np.random.randn(*pos.shape) * kwargs['noise_std']
        pos += pos_noise

    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels

    # test_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # train_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # return test_loader, train_loader


def generate_posterior_samples(tot_dataset_size, cond, **kwargs):

    samples = []
    cond = cond.numpy()
    while True:
        sample_pxgy = cond + np.random.randn(*cond.shape) * kwargs['noise_std']

        acceptance_prob = np.mean([
            np.exp(
                - (sample_pxgy - vert)**2 / 2 / (0.1**2)
            )
        for vert in np.array(verts)]) / np.mean([
            np.exp(
                - (verts[0] - vert)**2 / 2 / (0.1**2)
            )
        for vert in np.array(verts)])

        if np.random.uniform(0,1) >= acceptance_prob:
            samples.append(sample_pxgy)

        if len(samples) >= tot_dataset_size:
            break
        
    return np.array(samples)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    temp = '/home/vak2/Documents/temp.png'

    N = 10000
    idx = np.random.randint(10000)
    dat,lab = generate('all', N)
    dat_n = dat + np.random.randn(*dat.shape) * 0.1
    postsamp =  generate_posterior_samples(1000, dat_n[idx], noise_std=0.2)

    print(lab[idx])

    plt.scatter(dat_n[:,0], dat_n[:,1], alpha=0.01, edgecolor='none')
    plt.scatter(dat[:,0], dat[:,1], alpha=0.01, edgecolor='none')
    plt.scatter(postsamp[:,0], postsamp[:,1], alpha=0.01, edgecolor='none')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.savefig(temp)
    plt.close()









