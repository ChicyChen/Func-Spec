import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
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

import logging
import time
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# do a second pass pca to combine all layers
# python net3d_alllayers_together.py

import scipy.stats
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='dorsal_feature/adjust4_encoder2_input3_srate0.5', type=str)
parser.add_argument('--pca', action='store_true')

args = parser.parse_args()

# do_pca = True
do_pca = args.pca

root = args.root
folder = os.path.join(root, "downsampled")
out_folder = os.path.join(root, "together")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


return_layers = ['stem', 'layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']


features = []
tests = []
for key in return_layers:
    # load features
    print("Loading layer:", key)
    feature = np.load(os.path.join(folder, f'ds_{key}_features_all.npy')) # T,d
    test_feature = np.load(os.path.join(folder, f'ds_{key}_features_test_all.npy')) # T,d
    features.append(feature) # n_key, T, d
    tests.append(test_feature)

print("Concatenate train")

features = np.concatenate(features, axis = 1) # T, D
t1,D = features.shape

print("Concatenate test")

tests = np.concatenate(tests, axis = 1) # T, D
t2,_ = tests.shape

print("Together size:", features.shape)

if do_pca:
    # print("Standardize all")
    # # standardize
    # scaler = StandardScaler()
    # scaler.fit(features)
    # features = scaler.transform(features)
    # tests = scaler.transform(tests)


    print("PCA all")
    # PCA reduction while making sure 0.99 variance is preserved
    n_components = 0.99
    pca = PCA(n_components=n_components, svd_solver='full')
    features = pca.fit_transform(features)
    tests = pca.transform(tests) # use the same PCA as train

print("Save output")

np.save(os.path.join(out_folder, f'ds_features_all_2pca{do_pca}.npy'), features)
np.save(os.path.join(out_folder, f'ds_features_test_all_2pca{do_pca}.npy'), tests)








