import os
import sys
from importlib import reload
sys.path.append("/home/siyich/Func-Spec/utils")
sys.path.append("/home/siyich/Func-Spec/net3d")
sys.path.append("/home/siyich/Func-Spec/dataload")

from vicclr import VICCLR
from augmentation import *
from helpers import *
from dataloader import *

import random
import math
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import scipy.stats
import numpy as np
# import cupy as np

def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution

Arguments:

Required:
TR: repetition time at which to generate the HRF (in seconds)

Optional:
p: list with parameters of the two gamma functions:
                                                     defaults
                                                    (seconds)
   p[0] - delay of response (relative to onset)         6
   p[1] - delay of undershoot (relative to onset)      16
   p[2] - dispersion of response                        1
   p[3] - dispersion of undershoot                      1
   p[4] - ratio of response to undershoot               6
   p[5] - onset (seconds)                               0
   p[6] - length of kernel (seconds)                   32

"""

    p=[float(x) for x in p]

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf = scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    # hrf = scipy.stats.gamma.pdf(u.get(),p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u.get(),p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts = np.array(range(np.int64(p[6]/TR)))*fMRI_T
    # hrf=hrf[list(good_pts)]
    good_pts = good_pts.astype(int).tolist()
    hrf = hrf[good_pts]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/np.sum(hrf)
    return hrf

folder = "encoder1"
return_layers = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# create hrf
p  = [5, 16, 1, 1, 6, 0, 32]
srate = 5
hrf = spm_hrf(1/srate, p)

for key in return_layers:
    print("Loading layer:", key)

    # load features
    features = []
    tests = []
    for i in range(18):
        features_train = np.load(os.path.join(folder, f"{key}_features{i+1}.npy"))
        features.append(features_train)
    for i in range(5):
        features_test = np.load(os.path.join(folder, f"{key}_features_test{i+1}.npy"))
        tests.append(features_test)

    print("Concatenate layer:", key)

    T,D = features[0].shape
    features = np.concatenate(features)
    tests = np.concatenate(tests)

    print("Standardize layer:", key)

    # standardize
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    scaler2 = StandardScaler()
    scaler2.fit(tests)
    tests = scaler2.transform(tests)

    print("PCA layer:", key)

    # PCA reduction while making sure 0.99 variance is preserved
    n_components = 0.99
    pca = PCA(n_components=n_components, svd_solver='full')
    features = pca.fit_transform(features)
    tests = pca.transform(tests) # use the same PCA as train

    print("HRF layer:", key)

    # hrf and downsample
    features_hrf_list = []
    tests_hrf_list = []
    for i in range(18):
        features_pca = features[i*T:(i+1)*T,:]
        features_hrf = np.apply_along_axis(lambda m: np.convolve(m, hrf, mode='full'), axis=0, arr=features_pca)
        features_ds = features_hrf[5*srate:5*srate+2400,:]
        features_ds = features_ds[::2*srate,:]
        np.save(os.path.join(folder, f'ds_{key}_features{i+1}.npy'), features_ds) # T, d
        features_hrf_list.append(features_ds)
    for i in range(5):
        tests_pca = tests[i*T:(i+1)*T,:]
        tests_hrf = np.apply_along_axis(lambda m: np.convolve(m, hrf, mode='full'), axis=0, arr=tests_pca)
        features_ds_test = tests_hrf[5*srate:5*srate+2400,:]
        features_ds_test = features_ds_test[::2*srate,:]
        torch.save(os.path.join(folder, f'ds_{key}_features_test{i+1}.npy', features_ds_test)) # T, d
        tests_hrf_list.append(features_ds_test)

    print("Saving layer:", key)

    # save concatenated features
    features_hrf_list = np.concatenate(features_hrf_list)
    tests_hrf_list = np.concatenate(tests_hrf_list)
    np.save(os.path.join(folder, 'ds_{key}_features_all.npy'), features_hrf_list)
    np.save(os.path.join(folder, 'ds_{key}_features_test_all.npy'), tests_hrf_list)










