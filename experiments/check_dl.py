import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/yehengz/Func-Spec/utils")
sys.path.append("/home/yehengz/Func-Spec/net3d")
sys.path.append("/home/yehengz/Func-Spec/dataload")

from swinclr import SWINCLR

import random
import math
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader import get_data_ucf, get_data_k400, get_data_mk200, get_data_mk400, get_data_minik
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import time
import matplotlib.pyplot as plt

from augmentation import *
from distributed_utils import init_distributed_mode

# python -m torch.distributed.launch --nproc_per_node=8 experiments/train_net3d.py --sym_loss
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_swin.py --sym_loss --epochs 100 --base_lr 1e-4
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d.py --epochs 400 --batch_size 64 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 0 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 0.0

parser = argparse.ArgumentParser()

parser.add_argument('--frame_root', default='/data', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')
# --frame_root /data
                    
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)

parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--resume_epoch', default=400, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrain_folder', default='', type=str)
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--batch_size', default=64, type=int)
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-6, type=float, help='weight decay')

parser.add_argument('--random', action='store_true')
parser.add_argument('--num_seq', default=2, type=int)
parser.add_argument('--seq_len', default=17, type=int)
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--inter_len', default=0, type=int)    # does not need to be positive

parser.add_argument('--sym_loss', action='store_true')


parser.add_argument('--feature_size', default=768, type=int)
parser.add_argument('--projection', default=2048, type=int)
parser.add_argument('--proj_hidden', default=2048, type=int)
parser.add_argument('--proj_layer', default=3, type=int)

parser.add_argument('--mse_l', default=1.0, type=float)
parser.add_argument('--std_l', default=1.0, type=float)
parser.add_argument('--cov_l', default=0.04, type=float)
parser.add_argument('--infonce', action='store_true') # default is false
parser.add_argument('--temperature', default = 0.1, type = float)

parser.add_argument('--base_lr', default=4.8, type=float)
parser.add_argument('--warm_up', action = "store_true") #default value is false

# Running
parser.add_argument("--num-workers", type=int, default=128)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')

# Distributed
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
# parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist-url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--r21d', action='store_true')
parser.add_argument('--mc3', action='store_true')
parser.add_argument('--s3d', action='store_true')

parser.add_argument('--mk200', action='store_true')
parser.add_argument('--mk400', action='store_true')
parser.add_argument('--minik', action='store_true')
parser.add_argument('--k400', action='store_true')
parser.add_argument('--fraction', default=1.0, type=float)


def train_one_epoch(args, model, train_loader, optimizer, epoch, gpu=None, scaler=None, train=True, diff=False, mix=False, mix2=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    # for data in train_loader:
    for step, data in enumerate(train_loader, start=epoch * len(train_loader)):
        # TODO: be careful with video size
        # N = 2 by default
        video, label = data # B, N, C, T, H, W
        # print(video.shape)
        label = label.to(gpu)
        video = video.to(gpu)


        # scheduled differentiation step
        if diff:
            video = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
        if mix:
            video_diff = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
            video = video[:,:,:,:-1,:,:]
            video[:,1,:,:,:,:] = video_diff[:,1,:,:,:,:]
        if mix2:
            video_diff = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
            video = video[:,:,:,:-1,:,:]
            video[:,0,:,:,:,:] = video_diff[:,0,:,:,:,:]

        lr = adjust_learning_rate(args, optimizer, train_loader, step)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(video)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.mean().item() 
    
    return total_loss/num_batches


def main():
    torch.manual_seed(233)
    np.random.seed(233)

    args = parser.parse_args()

    
    per_device_batch_size = args.batch_size // args.world_size
    print(per_device_batch_size)

    if args.k400:
        loader_method = get_data_k400
    elif args.mk200:
        loader_method = get_data_mk200
    elif args.mk400:
        loader_method = get_data_mk400
    elif args.minik:
        loader_method = get_data_minik
    else:
        loader_method = get_data_ucf

    train_loader = loader_method(batch_size=per_device_batch_size, 
                                mode='train', 
                                transform_consistent=None, 
                                transform_inconsistent=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                random=args.random,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                ddp=False,
                                dim=150,
                                fraction=args.fraction,
                                )
    # test_loader = get_data_ucf(batch_size=per_device_batch_size, 
    #                             mode='val',
    #                             transform_consistent=None, 
    #                             transform_inconsistent=default_transform2(),
    #                             seq_len=args.seq_len, 
    #                             num_seq=args.num_seq, 
    #                             downsample=args.downsample,
    #                             random=args.random,
    #                             inter_len=args.inter_len,
    #                             frame_root=args.frame_root,
    #                             ddp=True,
    #                             dim = 240,
    #                             fraction = args.fraction
    #                             )
    num_batches = len(train_loader)

    # for data in train_loader:
    for step, data in enumerate(train_loader, start = 0):
        # TODO: be careful with video size
        # N = 2 by default
        if step >0:
            break
        video, label = data # B, N, C, T, H, W
        # print(video.shape)
        
        print(video.shape)
    


if __name__ == '__main__':
    main()



# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_swin_base.py --sym_loss --infonce --epochs 400 --base_lr 7e-5 --temperature 0.1

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin_base/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr7e-05_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.1_epoch_num400 --epoch_num 400 --gpu '6' --swin