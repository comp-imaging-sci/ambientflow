""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar
Email     : vak2@illinois.edu 
"""

import argparse
import sys
from sklearn.utils.extmath import randomized_svd
import glob
import imageio as io
import numpy as np
from numpy import pi, sin, cos
import numpy.linalg as la
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_reals", type=str)
parser.add_argument("--path_to_fakes", type=str)
parser.add_argument("--num_images", type=int, default=5000)
parser.add_argument("--svd_comps", type=int, default=1000)
parser.add_argument("--results_dir", type=str, default='')
args = parser.parse_args()


fnames_real = glob.glob(os.path.join(args.path_to_reals, '*.png'))[:args.num_images]
fnames_fake = glob.glob(os.path.join(args.path_to_fakes, '*.png'))[:args.num_images]

img_real = []
img_fake = []
for i in range(args.num_images):
    print(i)
    img_real.append(io.imread(fnames_real[i]))
    img_fake.append(io.imread(fnames_fake[i]))

img_real = np.stack(img_real, axis=0)/255
img_fake = np.stack(img_fake, axis=0)/255

# Mean
mean_real = np.mean(img_real, axis=0)
mean_fake = np.mean(img_fake, axis=0)
io.imsave( os.path.join(args.results_dir, "mean_real.png"), mean_real )
io.imsave( os.path.join(args.results_dir, "mean_fake.png"), mean_fake )
print(la.norm(mean_real - mean_fake)**2 / la.norm(mean_real)**2)

# Autocorrelation
def radial_profile(data, center):
    # https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def autocorrelation(imgs):
    X = np.stack(np.meshgrid(np.arange(imgs.shape[1]), imgs.shape[1]), axis=-1)
    rw = imgs.shape[1]//2
    papoulis_filter = lambda r:  (r<=rw)*( 1/pi * abs( sin(pi*r/rw) ) + ( 1 - r/rw )*cos(pi*r/rw) )
    papoulis_filter = papoulis_filter( la.norm(X - rw, axis=-1) )
    # imgs_fil = imgs*papoulis_filter
    imgs_fil = imgs
    ac = fftconvolve(imgs_fil, imgs_fil[:,::-1,::-1], axes=[1,2], mode='full')
    ac = np.mean(ac, axis=0)
    rw = ac.shape[0]//2
    # radprof = radial_profile(ac, [rw,rw])
    radprof = ac[rw, rw:]
    radprof = radprof / radprof.max()
    ac = ac / ac.max()
    return ac, radprof

ac_real, ac_radprof_real = autocorrelation(img_real)
ac_fake, ac_radprof_fake = autocorrelation(img_fake)
plt.plot(ac_radprof_real[:len(ac_radprof_real)//2], lw=8)
plt.plot(ac_radprof_fake[:len(ac_radprof_real)//2], '--', lw=8)
plt.xlabel("Distance from the center")
plt.ylabel("Autocorrelation")
plt.legend(["Real images", "Synthetic images"])
plt.savefig( os.path.join(args.results_dir, 'autocorrelation_radprof.png'), bbox_inches='tight' )
plt.savefig( os.path.join(args.results_dir, 'autocorrelation_radprof.svg'), bbox_inches='tight' );plt.close()

io.imsave( os.path.join(args.results_dir, 'autocorrelation_real.png'), ac_real)
io.imsave( os.path.join(args.results_dir, 'autocorrelation_fake.png'), ac_fake)

img_real = (img_real - mean_real.reshape(1, *mean_real.shape)).reshape(args.num_images, -1).T
img_fake = (img_fake - mean_fake.reshape(1, *mean_fake.shape)).reshape(args.num_images, -1).T

U_real,s_real,_ = randomized_svd(img_real, args.svd_comps)
U_fake,s_fake,_ = randomized_svd(img_fake, args.svd_comps)

c_real = U_real @ (U_real * s_real).T
c_fake = U_fake @ (U_fake * s_fake).T
print( la.norm(c_real - c_fake)**2 / la.norm(c_real)**2 )
print(la.norm(mean_real - mean_fake)**2 / la.norm(mean_real)**2)

plt.loglog(np.arange(1,len(s_real)+1), s_real**2)
plt.loglog(np.arange(1,len(s_real)+1), s_fake**2, '--')
plt.xlim([1,len(s_real)])
plt.ylabel(r"$i$-th Singular value of covariance matrix", math_fontfamily='stixsans')
plt.xlabel(r"$i$", math_fontfamily='stixsans')
plt.savefig( os.path.join(args.results_dir, 'svals.png'), bbox_inches='tight' )
plt.savefig( os.path.join(args.results_dir, 'svals.svg'), bbox_inches='tight' );plt.close()





