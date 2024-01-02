""" Copyright (c) 2022-2023 authors
Author    : Varun A. Kelkar, Rucha Deshpande
Email     : vak2@illinois.edu, r.deshpande@wustl.edu
"""

# import cv2
import numpy as np
import imageio as io
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, logging
import argparse
import os.path as p

from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from skimage import measure
from skimage.transform import resize
import glob
import pandas as pd
import scipy.linalg as la

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def clean_mask(input_mask):
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1
    cleaned_mask = labels_mask
    return cleaned_mask

def get_mask(arr):
    width = 10

    arr = np.where(arr > np.percentile(arr,99), np.percentile(arr,99), arr)
    arr_in = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))*255

    blurred = gaussian(arr_in, sigma=1)

    thresh = np.max(blurred[:,:width])
    binary = blurred > thresh

    mask_ret = (binary_fill_holes(binary)*1).astype(np.uint8)
    mask_ret = clean_mask(mask_ret)

    area_out = np.count_nonzero(mask_ret)/(mask_ret.shape[0]*mask_ret.shape[1])

    if area_out < 0.1:
        mask_ret = np.ones((arr.shape[0], arr.shape[1]), dtype = np.uint8)
        area_out = 1

    # mask_ret_im = sitk.GetImageFromArray(mask_ret)

    return mask_ret, area_out

settings = {}
settings['binCount'] = 32 #binCount
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkLinear
settings['sigma'] = [1, 2]
settings['distances'] = [1,2,3]
# settings['level'] = 3 # Wavelet level of decompsition, default 1, count staarts at 0

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllImageTypes()

logger = radiomics.logger
logger.setLevel(logging.CRITICAL)

def extract_features(fnames):

    for i,f in enumerate(fnames):
        print(f"{i}/{len(fnames)}", end='\r')

        img = io.imread(f)

        try:
            mask, _ = get_mask(img)
            sitk_img = sitk.GetImageFromArray(img)
            sitk_mask = sitk.GetImageFromArray(mask)
            feature_dict = dict(extractor.execute(sitk_img, sitk_mask))
            df = pd.DataFrame(feature_dict.items()).T
            df = pd.DataFrame(df.values[1:], columns=df.iloc[0])
            for c in df.columns:
                if 'diagnostics' in c:
                    df.drop(c, axis=1, inplace=True)
            if i == 0:
                features = df.copy()
            else:
                features = pd.concat([features, df])

        except Exception as e:
            print(e)
            df = features.iloc[-1].copy()
            df *= np.nan
            features = features.append(df)

    return features

def compute_pca(data, num_components=2, pca=None):
    """
    Data is in the shape n_features x n_points
    """
    scalar = StandardScaler()
    scalar.fit(data)
    data_scaled = scalar.transform(data)
    if pca == None:
        pca = PCA(n_components=num_components, random_state=1234)

    data_pcs = pca.fit_transform(data_scaled)
    return data_pcs, pca

def compute_contour_value(hist, percent):
    mode = hist.max()
    levels = np.arange(mode+1)
    masses = np.array([np.sum( hist*(hist < l) ) for l in levels])
    masses /= np.sum(hist)
    percents = np.arange(101)
    interp_levels = np.interp( percents/100, masses, levels )
    return interp_levels[int(percent)]

def compute_contour_value2(hist, percent):
    mode = hist.max()
    levels = np.linspace(0, mode, 100)
    masses = np.array([np.sum( hist*(hist < l) ) for l in levels])
    masses /= np.sum(hist)
    percents = np.arange(101)
    interp_levels = np.interp( percents/100, masses, levels )
    return interp_levels[int(percent)]

def save_features(features_path, features):
    print(f"Saving features to {features_path}")
    features.to_csv(features_path, index=False)

def compute_pca_all(features_list):
    features_list = [ ff.dropna(axis=0) for ff in features_list ]
    features_all = [ff.to_numpy().astype(float) for ff in features_list]
    features_size = [ff.shape[0] for ff in features_all]
    features_all = np.concatenate(features_all, axis=0)
    features_pcs,_ = compute_pca(features_all, num_components=2)
    split_locs = np.cumsum(features_size)[:-1]
    features_pcs_list = np.split(features_pcs, split_locs)
    return features_pcs_list

def plot_histograms(features_pcs_list, num_bins=50):
    F = np.concatenate(features_pcs_list, axis=0)
    xbins = np.linspace(F[:,0].min(), F[:,0].max(), num_bins)
    ybins = np.linspace(F[:,1].min(), F[:,1].max(), num_bins)

    hists = []
    for i, f in enumerate(features_pcs_list):
        hist,_,_ = np.histogram2d(f[:,0], f[:,1], bins=[xbins, ybins])
        plt.imshow(hist, cmap='hot')
        plt.savefig(f'temp_{i}.png', bbox_inches='tight')
        plt.close()

def compute_frd(real_pcs, fake_pcs):
    real_mean = np.mean(real_pcs, axis=0)
    fake_mean = np.mean(fake_pcs, axis=0)
    real_cov = (real_pcs - real_mean.reshape(1,-1)).T @ (real_pcs - real_mean.reshape(1,-1))
    fake_cov = (fake_pcs - fake_mean.reshape(1,-1)).T @ (fake_pcs - fake_mean.reshape(1,-1))
    fd2 = la.norm( real_mean - fake_mean )**2 + \
            np.trace( real_cov + fake_cov - 2*la.sqrtm( real_cov @ fake_cov )  )
    return fd2

def plot_histograms2(features_pcs_list, num_bins=100):
    F = np.concatenate(features_pcs_list, axis=0)
    xbins = np.linspace(F[:,0].min(), F[:,0].max(), num_bins)
    ybins = np.linspace(F[:,1].min(), F[:,1].max(), num_bins)

    xx, yy = np.meshgrid(xbins, ybins)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    for i, f in enumerate(features_pcs_list):
        # hist,_,_ = np.histogram2d(f[:,0], f[:,1], bins=[xbins, ybins])
        kernel = st.gaussian_kde(f.T, bw_method=0.1)
        dens = kernel(positions).T.reshape(xx.shape)
        if i == 0:
            dens0 = dens.copy()
            c0 = compute_contour_value2(dens0, 20)
        c = compute_contour_value2(dens, 20)
        
        plt.imshow(dens, cmap='GnBu')
        if i != 0:
            plt.contour(dens0, levels=[c0], linestyles='dashed')
        plt.contour(dens, levels=[c])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'temp_{i}.png', bbox_inches='tight')
        # plt.savefig(p.join(results_dir, f'pca_plot_{names[i]}.png'), bbox_inches='tight')
        # plt.savefig(p.join(results_dir, f'pca_plot_{names[i]}.svg'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':

    num_images = 2000
    results_dir = '/home/vak2/Documents/ambientflow/results/mri/radiomic_features'
    names = ['degraded-4x-0.1', 'plstv-4x-0.1', 'ambient-4x-0.1']

    path_to_obj = '/shared/aristotle/SOMS/varun/ambientflow/data/FastMRIT2Dataset-real-image-data/*.png'
    path_to_van = '/shared/aristotle/SOMS/varun/ambientflow/results/mri/vanilla/008-Glow-flow16-block5-lr0.0001-bits0/fake-010152-0.7-10000/*.png'
    # path_to_deg = '/shared/aristotle/SOMS/varun/ambientflow/data/FastMRIT2Dataset-FourierSamplerGaussianNoise-0x-0.1-image-data/*.png'
    path_to_deg = '/shared/aristotle/SOMS/varun/ambientflow/data/FastMRIT2Dataset-FourierSamplerGaussianNoise-4x-0.1-image-data/*.png'
    # path_to_pls = '/shared/curie/SOMS/varun/ambientflow/results/mri/recon/fista-FastMRIT2Dataset-FourierSamplerGaussianNoise-0x-0.1/dataset/*.png'
    path_to_pls = '/shared/curie/SOMS/varun/ambientflow/results/mri/recon/fista-FastMRIT2Dataset-FourierSamplerGaussianNoise-4x-0.1/dataset/*.png'
    # path_to_amb = '/shared/anastasio1/SOMS/varun/ambientflow/results/mri/ambient/condglow/001-anastasio6-1gpu-degFourierSamplerGaussianNoise-Glow-CondGlow3-reg5.0-spparam0.03-thw0.05-bit0-lr0.0001-nz2-impw1/fake-019906-0.85-10000/*.png'
    path_to_amb = '/shared/radon/SOMS/varun/ambientflow/results/mri/ambient/condglow/003-anastasio6-1gpu-degFourierSamplerGaussianNoise-Glow-CondGlow3-reg5.0-spparam0.03-thw0.05-bit0-lr0.0001-nz2-impw1/fake-019301-0.85-10000/*.png'

    fnames_obj = sorted(glob.glob(path_to_obj))[:num_images]
    fnames_van = sorted(glob.glob(path_to_van))[:num_images]
    fnames_deg = sorted(glob.glob(path_to_deg))[::-1][:num_images]
    fnames_pls = sorted(glob.glob(path_to_pls))[::-1][:num_images]
    fnames_amb = sorted(glob.glob(path_to_amb))[:num_images]

    features_list = []
    for i, fnames in enumerate([fnames_deg, fnames_pls, fnames_amb]):

        print(f"Computing features for {names[i]}")
        features = extract_features(fnames)
        features_list.append(features)
        save_features(p.join(results_dir, f'features_{names[i]}.csv'), features)

        


