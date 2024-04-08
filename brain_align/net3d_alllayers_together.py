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


folder = "encoder1"
return_layers = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

features = []
tests = []
for key in return_layers:
    # load features
    feature = np.load(os.path.join(folder, 'ds_{key}_features_all.npy')) # T,d
    test_feature = np.load(os.path.join(folder, 'ds_{key}_features_test_all.npy')) # T,d
    features.append(feature) # n_key, T, d
    tests.append(tests)
features = np.concatenate(features, axis = 1) # T, D
tests = np.concatenate(tests, axis = 1) # T, D

# standardize
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
scaler2 = StandardScaler()
scaler2.fit(tests)
tests = scaler2.transform(tests)

# PCA reduction while making sure 0.99 variance is preserved
n_components = 0.99
pca = PCA(n_components=n_components, svd_solver='full')
features = pca.fit_transform(features)
tests = pca.transform(tests) # use the same PCA as train

np.save(os.path.join(folder, 'ds_features_all.npy'), features)
np.save(os.path.join(folder, 'ds_features_test_all.npy'), tests)








