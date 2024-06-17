import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/siyich/Func-Spec/utils")
sys.path.append("/home/siyich/Func-Spec/net3d")
sys.path.append("/home/siyich/Func-Spec/resnet_edit")
sys.path.append("/home/siyich/Func-Spec/dataload")

from vicclr import VICCLR
from augmentation import *
from helpers import *
from dataloader import *
from cal_diff import *

from resnet import r3d_18

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
import argparse

# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


ventral_path = '/data/checkpoints_yehengz/resnet_sdtd/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size512_projection128_proj_hidden2048_epochs400_seed233_operation_summation_width_deduc_ratio1.0_stem_deductFalse'
dorsal_path = '/data/checkpoints_yehengz/resnet_sdtd/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size512_projection128_proj_hidden2048_epochs400_seed233_operation_summation_width_deduc_ratio1.0_stem_deductFalse'

parser = argparse.ArgumentParser()

parser.add_argument('--resnet_folder', default=dorsal_path, type=str)
parser.add_argument('--resnet_name', default='resnet2_epoch400.pth.tar', type=str)
parser.add_argument('--outfolder_root', default='dorsal_feature', type=str)
parser.add_argument('--input_type', default=3, type=int)
parser.add_argument('--encoder_num', default=2, type=int)
parser.add_argument('--srate', default=6, type=float)
parser.add_argument('--width_deduction_ratio', default=1.41, type=float, help='weight decay')
                    

args = parser.parse_args()


# python net3d_alllayers_extraction.py

return_layers = ['stem', 'layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1']
# pool_sizes = {'stem':(1,8,8), 'layer1.0':(1,8,8), 'layer1.1':(1,8,8), 'layer2.0':(2,4,4), 'layer2.1':(2,4,4), 'layer3.0':(1,4,4), 'layer3.1':(1,4,4), 'layer4.0':(1,2,2), 'layer4.1':(1,2,2)} # adjust
# pool_sizes = {'stem':(1,16,16), 'layer1.0':(1,16,16), 'layer1.1':(1,16,16), 'layer2.0':(2,8,8), 'layer2.1':(2,8,8), 'layer3.0':(1,8,8), 'layer3.1':(1,8,8), 'layer4.0':(1,4,4), 'layer4.1':(1,4,4)} # adjust2
# pool_sizes = {'stem':(2,8,8), 'layer1.0':(2,8,8), 'layer1.1':(2,8,8), 'layer2.0':(4,4,4), 'layer2.1':(4,4,4), 'layer3.0':(2,4,4), 'layer3.1':(2,4,4), 'layer4.0':(2,2,2), 'layer4.1':(1,2,2)} # adjust3
pool_sizes = {'stem':(4,8,8), 'layer1.0':(4,8,8), 'layer1.1':(4,8,8), 'layer2.0':(8,4,4), 'layer2.1':(8,4,4), 'layer3.0':(4,4,4), 'layer3.1':(4,4,4), 'layer4.0':(4,2,2), 'layer4.1':(2,2,2)} # adjust4




input_type = args.input_type # 0: original, 1: d/dt, 2: average
resnet_folder = args.resnet_folder

resnet_path = os.path.join(resnet_folder, args.resnet_name)

out_folder = os.path.join(args.outfolder_root, f"adjust4_encoder{args.encoder_num}_input{input_type}_srate{args.srate}", "raw")
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# load pretrained model
cuda = torch.device('cuda')

resnet = models.video.r3d_18()
# resnet = r3d_18(width_deduction_ratio = 1.41, stem_deduct = False)

resnet.load_state_dict(torch.load(resnet_path))
resnet = resnet.to(cuda)
resnet.eval()

# hooks = {layer_name: NetHook(resnet, layer=layer_name, pool=True) for layer_name in return_layers}
# hooks = {layer_name: NetHook(resnet, layer=layer_name, pool=True, only_time = True) for layer_name in return_layers}
hooks = {layer_name: NetHook(resnet, layer=layer_name, pool=True, size = pool_sizes[layer_name]) for layer_name in return_layers}
# hooks = {layer_name: NetHook(resnet, layer=layer_name, pool=False) for layer_name in return_layers}


# transforms = videofmri_transform() # strong augmentation
transforms = videofmri_transform_simple() # simple augmentation

# process video data used in fmri recording
frame_root = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/frames" # without downsampling, 30 Hz
num_images = 14400

srate = args.srate
num_list = np.arange(1,num_images+1,int(30/srate))

# loop through 18 segs
num_seg = 18
num_test = 5
frame_folder_list = [os.path.join(frame_root, "seg"+str(i+1)) for i in range(num_seg)]
test_folder_list = [os.path.join(frame_root, "test"+str(i+1)) for i in range(num_test)]
num_frame = int(30/srate)
if num_frame > 8:
    num_frame = 8

print("Process training")
# save the features of each segment
with torch.no_grad():
    seg_num = 1
    for frame_folder in frame_folder_list:
        # representation_list = []
        print("Reading:", frame_folder)
        
        representation_list = {layername:[] for layername in return_layers}

        for i in range(len(num_list) - num_frame):
            # construct clip and preprocess
            idx_block = [num_list[i]+idx_diff for idx_diff in range(num_frame)]
            # idx_block = num_list[i:i+num_frame]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)

            if input_type == 1:
                # derivatives
                if random.random() < 0.75:
                    clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            if input_type == 2:
                # average
                if random.random() < 0.25:
                    average = torch.mean(clip, dim = 2, keepdim = True)
                    clip = torch.repeat_interleave(average, num_frame-1, dim = 2)
            if input_type == 3:
                grayscale_clip = RGB2Gray(clip.unsqueeze(1)) # B,N,T,H,W
                clip_sd = get_spatial_diff(grayscale_clip).squeeze(1)
                clip_td = get_temporal_diff(grayscale_clip).squeeze(1)
                # raise NotImplementedError
            
            if input_type != 3:
                for key in return_layers:
                    representation_list[key].append(hooks[key](clip).cpu().detach().numpy())
            else:
                for key in return_layers:
                    representation_list[key].append(hooks[key](clip_sd).cpu().detach().numpy() + hooks[key](clip_td).cpu().detach().numpy())

        for key in return_layers:
            print("Saving layer:", key)
            # features = torch.vstack(representation_list[key]) # T, D: T, 512
            features = np.vstack(representation_list[key]) # T, D: T, 512
            print("Layer feature shape:", features.shape)
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
        
        representation_list = {layername:[] for layername in return_layers}

        for i in range(len(num_list) - num_frame):
            # construct clip and preprocess
            idx_block = [num_list[i]+idx_diff for idx_diff in range(num_frame)]
            # idx_block = num_list[i:i+num_frame]
            clip = [pil_loader(os.path.join(frame_folder, 'im-'+str(i)+'.jpg')) for i in idx_block]
            clip = transforms(clip) # apply same transform
            clip = torch.stack(clip, 0) # T, C, H, W
            clip = clip.permute(1,0,2,3) # C, T, H, W
            clip = clip.unsqueeze(0) # 1, C, T, H, W: [1, 3, 8, 112, 112]
            clip = clip.to(cuda)

            if input_type == 1:
                # derivatives
                if random.random() < 0.75:
                    clip = clip[:,:,1:,:,:] - clip[:,:,:-1,:,:]
            if input_type == 2:
                # average
                if random.random() < 0.25:
                    average = torch.mean(clip, dim = 2, keepdim = True)
                    clip = torch.repeat_interleave(average, num_frame-1, dim = 2)
            if input_type == 3:
                grayscale_clip = RGB2Gray(clip.unsqueeze(1)) # B,N,T,H,W
                clip_sd = get_spatial_diff(grayscale_clip).squeeze(1)
                clip_td = get_temporal_diff(grayscale_clip).squeeze(1)
                # raise NotImplementedError

            if input_type != 3:
                for key in return_layers:
                    representation_list[key].append(hooks[key](clip).cpu().detach().numpy())
            else:
                for key in return_layers:
                    representation_list[key].append(hooks[key](clip_sd).cpu().detach().numpy() + hooks[key](clip_td).cpu().detach().numpy())

        for key in return_layers:
            print("Saving layer:", key)
            # features = torch.vstack(representation_list[key]) # T, D: T, 512
            features = np.vstack(representation_list[key]) # T, D: T, 512
            # torch.save(features, os.path.join(out_folder, f'{key}_features_test{seg_num}.pt'))
            np.save(os.path.join(out_folder, f'{key}_features_test{seg_num}.npy'), features)
        seg_num += 1





