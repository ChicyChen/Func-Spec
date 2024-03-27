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

hook_layer = -2
out_folder = "encoder2_simple_av"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# load pretrained model
cuda = torch.device('cuda')
# resnet_path = "/data/checkpoints_yehengz/2streams_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_Summation_prob_derivative0.25_prob_average0.25/resnet1_epoch400.pth.tar"
resnet_path = "/data/checkpoints_yehengz/2streams_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_Summation_prob_derivative0.25_prob_average0.25/resnet2_epoch400.pth.tar"
resnet = models.video.r3d_18()
resnet.load_state_dict(torch.load(resnet_path))
resnet = resnet.to(cuda)
resnet.eval()
encoder = NetHook(resnet, layer = hook_layer)
encoder = encoder.to(cuda)
encoder.eval()
# transforms = videofmri_transform() # strong augmentation
transforms = videofmri_transform_simple() # simple augmentation



# process video data used in fmri recording
frame_root = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/frames" # without downsampling, 30 Hz
num_images = 14400
# frame_root = "/data/3/human/Human_Visual_Experiments/movie_stimuli/sections-8min-18/Concatenated_sections/movie_frame_10Hz" # with downsampling, 10 Hz
# num_images = 4800

srate = 5
num_list = np.arange(1,num_images+1,int(30/srate))

transform_inconsistent=default_transform()

# loop through 18 segs
num_seg = 18
num_test = 5
frame_folder_list = [os.path.join(frame_root, "seg"+str(i+1)) for i in range(num_seg)]
test_folder_list = [os.path.join(frame_root, "test"+str(i+1)) for i in range(num_test)]


# frame_folder = os.path.join(frame_root, "seg1") # 30 Hz
# frame_folder = os.path.join(frame_root, "test1") # 30 Hz
# frame_folder = os.path.join(frame_root, "section1") # 10 Hz


# save the features of each segment
with torch.no_grad():
    seg_num = 1
    for frame_folder in frame_folder_list:
        representation_list = []
        for i in range(len(num_list) - 8):
            # construct clip and preprocess
            idx_block = num_list[i:i+8]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            # clip = [pil_loader(os.path.join(frame_folder, str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)
            # clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            average = torch.mean(clip, dim = 2, keepdim = True)
            clip = torch.repeat_interleave(average, 7, dim = 2)
            representation = encoder(clip)
            representation_list.append(representation)
        print(len(representation_list))
        features = torch.vstack(representation_list) # T, D: T, 512
        torch.save(features, os.path.join(out_folder, f'features{seg_num}.pt'))
        seg_num += 1

# save the features of each test segment
with torch.no_grad():
    seg_num = 1
    for frame_folder in test_folder_list:
        representation_list = []
        for i in range(len(num_list) - 8):
            # construct clip and preprocess
            idx_block = num_list[i:i+8]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            # clip = [pil_loader(os.path.join(frame_folder, str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)
            # clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            average = torch.mean(clip, dim = 2, keepdim = True)
            clip = torch.repeat_interleave(average, 7, dim = 2)
            representation = encoder(clip)
            representation_list.append(representation)
        print(len(representation_list))
        features = torch.vstack(representation_list) # T, D: T, 512
        torch.save(features, os.path.join(out_folder, f'features_test{seg_num}.pt'))
        seg_num += 1

# torch.save(features, 'features_test.pt')



