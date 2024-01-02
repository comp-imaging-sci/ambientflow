""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cmap = 'gist_heat'

def plot_true_distribution(scale=0.15):
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
    
    x,y = np.meshgrid(np.linspace(-4,4,128), np.linspace(-4,4,128))

    pdf = np.zeros_like(x)
    for v in verts:
        pdf += np.exp( - ( (x - v[0])**2 + (y - v[1])**2 ) / 2 / scale**2 )

    plt.imshow(pdf, cmap=cmap)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.axis('equal')
    plt.axis('off')
    plt.title("True distribution")
    plt.savefig('../results/toy/true_distribution.svg', bbox_inches='tight', pad=0)
    plt.savefig('../results/toy/true_distribution.png', bbox_inches='tight', pad=0)
    plt.close()
    

def plot_noisy_distribution(scale=0.15, noise_std=0.45):
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
    
    x,y = np.meshgrid(np.linspace(-4,4,128), np.linspace(-4,4,128))

    pdf = np.zeros_like(x)
    for v in verts:
        pdf += np.exp( - ( (x - v[0])**2 + (y - v[1])**2 ) / 2 / (scale**2 + noise_std**2) )

    plt.imshow(pdf, cmap=cmap)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.axis('equal')
    plt.axis('off')
    plt.title("Measurement distribution")
    plt.savefig('../results/toy/meas_distribution.svg', bbox_inches='tight', pad=0)
    plt.savefig('../results/toy/meas_distribution.png', bbox_inches='tight', pad=0)
    plt.close()


def plot_inn_distribution(scale=0.15, path='', name=''):

    import scipy.stats as st
    from skimage.transform import resize

    data = np.load(path)[:2000]
    # bins = [ 
    #     np.linspace(-4, 4, 128),
    #     np.linspace(-4, 4, 128),
    # ]
    # kernel = st.gaussian_kde(data.T)
    # x,y = np.meshgrid(bins[0], bins[1])
    # positions = np.vstack([x.ravel(), y.ravel()])
    # counts = np.reshape(kernel(positions).T, x.shape)
    counts,_,_ = np.histogram2d(data[...,0], data[...,1], bins=64, range=[[-4,4],[-4,4]])
    counts = resize(counts, (128,128), anti_aliasing=True)

    plt.imshow(counts, cmap=cmap)
    # plt.scatter(data[...,0], data[...,1], alpha=0.3, edgecolor='none')
    # plt.xlim([-4,4])
    # plt.ylim([-4,4])
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"{name} distribution")
    plt.savefig(f'../results/toy/{name}_distribution.svg', bbox_inches='tight', pad=0)
    plt.savefig(f'../results/toy/{name}_distribution.png', bbox_inches='tight', pad=0)  
    plt.close()
    



if __name__ == '__main__':

    vanilla_path_0p7 = '/home/vak2/Documents/ambientflow/results/toy/toy_vanilla2/007-AllInOneBlock-24/data-029000-0.7.npy'
    vanilla_path_1p0 = '/home/vak2/Documents/ambientflow/results/toy/toy_vanilla2/007-AllInOneBlock-24/data-029000-1.0.npy'
    ambient_path_0p7 = '/home/vak2/Documents/ambientflow/results/toy/toy_ambient2/029-AllInOneBlock-24/data-027250-0.7.npy'
    ambient_path_1p0 = '/home/vak2/Documents/ambientflow/results/toy/toy_ambient2/029-AllInOneBlock-24/data-027250-1.0.npy'

    plot_true_distribution()
    plot_noisy_distribution()
    plot_inn_distribution(
        path = ambient_path_1p0,
        name = 'ambient-1.0'
    )
    plot_inn_distribution(
        path = vanilla_path_1p0,
        name = 'vanilla-1.0'
    )