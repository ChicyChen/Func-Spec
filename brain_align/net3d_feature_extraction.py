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

# load pretrained model
cuda = torch.device('cuda')
resnet_path = "/home/siyich/Func-Spec/checkpoints/ucf1.0_pcn_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse/resnet_epoch400.pth.tar"
resnet = models.video.r3d_18()
resnet.load_state_dict(torch.load(resnet_path))
resnet = resnet.to(cuda)
resnet.eval()
encoder = NetHook(resnet, layer = -2)
encoder = encoder.to(cuda)
encoder.eval()

# process video data used in fmri recording
frame_root = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/frames"
num_seg = 18
num_images = 14400
transform_inconsistent=default_transform()
# loop through 18 segs
frame_folder = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/frames/seg1"
representation_list = []
for i in range(num_images - 8):
    # construct clip and preprocess
    idx_block = range(i, i+8)
    clip = [pil_loader(os.path.join(frame_folder, '/im-'+str(i+1)+'.jpg')) for i in idx_block]
    clip = videofmri_transform(clip) # apply same transform
    clip = clip.to(cuda)
    representation = encoder(clip)
    representation_list.append(representation)
features = torch.vstack(representation_list) # T, D
torch.save(features, 'features.pt')

# PCA reduction
n_components = 3
pca = PCA(n_components)
features_pca = pca.fit_transform(features)
torch.save(features_pca, 'features_pca.pt') # T, d
