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

# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

# python net3d_alllayers_extraction.py

return_layers = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
# return_layers = ['stem']
# return_layers = ['layer1']



resnet_folder = '/data/checkpoints_yehengz/2streams2projs_rdra/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_epochs400_seed233_operation_Summation_prob_derivative0.25_prob_average0.75'
resnet_path = os.path.join(resnet_folder, 'resnet1_epoch400.pth.tar')
# resnet_path = os.path.join(resnet_folder, 'resnet2_epoch400.pth.tar')

out_folder = "encoder1"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# load pretrained model
cuda = torch.device('cuda')
resnet = models.video.r3d_18()
resnet.load_state_dict(torch.load(resnet_path))
resnet = resnet.to(cuda)
resnet.eval()
hooks = {layer_name: NetHook(resnet, layer=layer_name) for layer_name in return_layers}

# transforms = videofmri_transform() # strong augmentation
transforms = videofmri_transform_simple() # simple augmentation

# process video data used in fmri recording
frame_root = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/frames" # without downsampling, 30 Hz
num_images = 14400

srate = 5
num_list = np.arange(1,num_images+1,int(30/srate))

# loop through 18 segs
num_seg = 18
num_test = 5
frame_folder_list = [os.path.join(frame_root, "seg"+str(i+1)) for i in range(num_seg)]
test_folder_list = [os.path.join(frame_root, "test"+str(i+1)) for i in range(num_test)]


print("Process training")
# save the features of each segment
with torch.no_grad():
    seg_num = 1
    for frame_folder in frame_folder_list:
        # representation_list = []
        print("Reading:", frame_folder)
        representation_list = {'stem':[], 'layer1':[], 'layer2':[], 'layer3':[], 'layer4':[], 'avgpool':[]}
        # representation_list = {'stem':[]}
        # representation_list = {'avgpool':[]}
        # representation_list = {'layer1':[]}


        for i in range(len(num_list) - 8):
            # construct clip and preprocess
            idx_block = num_list[i:i+8]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)

            # derivatives
            # clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            # average
            # average = torch.mean(clip, dim = 2, keepdim = True)
            # clip = torch.repeat_interleave(average, 7, dim = 2)
            
            for key in return_layers:
                representation_list[key].append(hooks[key](clip).cpu().detach().numpy())

        for key in return_layers:
            print("Saving layer:", key)
            # features = torch.vstack(representation_list[key]) # T, D: T, 512
            features = np.vstack(representation_list[key]) # T, D: T, 512
            # torch.save(features, os.path.join(out_folder, f'{key}_features{seg_num}.pt'))
            np.save(os.path.join(out_folder, f'{key}_features{seg_num}.npy'), features)

        seg_num += 1


print("Process testing")
# save the features of each test segment
with torch.no_grad():
    seg_num = 1
    for frame_folder in test_folder_list:
        # representation_list = []
        print("Reading:", frame_folder)
        representation_list = {'stem':[], 'layer1':[], 'layer2':[], 'layer3':[], 'layer4':[], 'avgpool':[]}
        # representation_list = {'stem':[]}
        # representation_list = {'avgpool':[]}
        # representation_list = {'layer4':[]}


        for i in range(len(num_list) - 8):
            # construct clip and preprocess
            idx_block = num_list[i:i+8]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)

            # derivatives
            # clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            # average
            # average = torch.mean(clip, dim = 2, keepdim = True)
            # clip = torch.repeat_interleave(average, 7, dim = 2)

            for key in return_layers:
                # representation_list[key].append(hooks[key](clip))
                representation_list[key].append(hooks[key](clip).cpu().detach().numpy())

        for key in return_layers:
            print("Saving layer:", key)
            # features = torch.vstack(representation_list[key]) # T, D: T, 512
            features = np.vstack(representation_list[key]) # T, D: T, 512
            # torch.save(features, os.path.join(out_folder, f'{key}_features_test{seg_num}.pt'))
            np.save(os.path.join(out_folder, f'{key}_features_test{seg_num}.npy'), features)
        seg_num += 1





