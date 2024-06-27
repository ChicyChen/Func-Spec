import os
import sys
import argparse

sys.path.append("/home/yehengz/Func-Spec/utils")
sys.path.append("/home/yehengz/Func-Spec/net3d")
sys.path.append("/home/yehengz/Func-Spec/dataload")
sys.path.append("/home/yehengz/Func-Spec/resnet_edit")

from resnet import r3d_18

from retrieval import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_test import get_data_ucf, get_data_hmdb
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--hmdb', action='store_true') # default value is False
# parser.add_argument('--mnist', action='store_true')
parser.add_argument('--random', action='store_true') # default value is False
parser.add_argument('--kinetics', action='store_true') # default value is False

parser.add_argument('--k', default=1, type=int)

parser.add_argument('--ckpt_folder', default='checkpoints/3dbase_ucf101_lr0.0001_wd1e-05', type=str) # need tp adjust to my_path
parser.add_argument('--epoch_num', default=400, type=int)

parser.add_argument('--num_seq', default=10, type=int)
parser.add_argument('--seq_len', default=16, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--inter_len', default=0, type=int)
# parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--img_size', default=112, type=int)
parser.add_argument('--r21d', action='store_true') # default is False
parser.add_argument('--mc3', action='store_true') # default is False
parser.add_argument('--s3d', action='store_true') # default is False
parser.add_argument('--swin', action='store_true') # default is False



parser.add_argument('--feature_concat', action = 'store_true') # default is False

parser.add_argument('--seed', default = 233, type = int) # seed used during training
parser.add_argument('--which_encoder', default = 0, type = int) # default is 1, the other option is 2; if is 0, then use the basic simclr structure, which has only on encoder
parser.add_argument('--width_deduction_ratio', default=1.0, type = float)
parser.add_argument('--stem_deduct', action='store_true') # default is false

def test_transform():
    transform = transforms.Compose([
        Scale(size=128),
        # RandomCrop(size=args.img_size, consistent=True),
        RandomSizedCrop(size=args.img_size, consistent=True),
        Scale(size=(112, 112)),
        ToTensor(), # range of the data becomes [0,1]
        Normalize() # range of the data becomes around [-2,2]
    ])
    return transform

def RGB2Gray(video):
    # formula of RGB to gray: Y = 0.2125 R + 0.7154 G + 0.0721 B
    # convert a batch of rgb video frames augmented in 2 different ways into grayscale
    # shape of the input "video" is [B, N, C, T, H, W]
    # shape of the output "grayscle" is [B, N, T, H, W]
    gray = 0.2125*video[:,:,0,:,:,:] + 0.7154*video[:,:,1,:,:,:] + 0.0721*video[:,:,2,:,:,:]

    return gray

def padding(data):
    # padding H and W by 1 with replicated value
    # F.pad(input, (1,1,1,1,0,0), mode = 'replicate') supposed to do the same thing
    # but when running F.pad(...) there is a strange error 
    pad_data_tmp = torch.cat((data[...,0:1], data, data[...,-1:]),4)
    pad_data = torch.cat((pad_data_tmp[...,0:1,:], pad_data_tmp, pad_data_tmp[...,-1:,:]),3)
    return pad_data

def poisson_blend(Ix,Iy,iteration=50):
    #Ix, Iy can only be np array...?
    #shape of Ix and Iy are now B, N, T, H, W
    device = Ix.device
    lap_blend = torch.zeros(Ix.shape,  device=device)


    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.detach().clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[...,1:-1,2:] -  Iy[...,1:-1,1:-1]
                    + Iy[...,2:,1:-1] -  Ix[...,1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[...,2:,1:-1] + lap_blend_old[...,0:-2,1:-1]
                                 + lap_blend_old[...,1:-1,2:] + lap_blend_old[...,1:-1,0:-2])

        lap_blend[...,1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend

def vizDiff(d,thresh=0.24):
    # shape of input is B,N,T,H,W
    device = d.device
    diff = d.detach().clone()
    rgb_diff = 0
    B,N,T,H,W = diff.shape
    rgb_diff = torch.zeros([B,N,3,T,H,W], device=device) #background is zero
    diff[abs(diff)<thresh] = 0
    rgb_diff[:,:,0,...][diff>0] = diff[diff>0] # diff[diff>0]
    rgb_diff[:,:,1,...][diff>0] = diff[diff>0]
    rgb_diff[:,:,2,...][diff>0] = diff[diff>0]

    rgb_diff[:,:,0,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,1,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,2,...][diff<0] = diff[diff<0]
    return rgb_diff

def get_spatial_diff(data):
    # data is grayscale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get SD_x and SD_y, both with shape B,N,T,H,W
    # step2: use poisson blending to get SD_xy
    # step3: based on the value of SD_xy in the last two dimensions, convert it back to B, N, C, T,H,W
    padded_data = padding(data)
    SD_x = (padded_data[...,1:-1,:-2] - padded_data[...,1:-1, 2:])/2
    SD_y = (padded_data[...,:-2,1:-1] - padded_data[...,2:,1:-1])/2
    SD_xy = poisson_blend(SD_x, SD_y)
    SD_xy = vizDiff(SD_xy)
    return SD_xy

def get_temporal_diff(data):
    # data is grascale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get TD, with shape B,N,T-1,H,W
    # step2 convert TD back to B,N,C,T,H,W
    TDiff = data[:,:,1:,:,:] - data[:,:,:-1,:,:]
    TDiff = vizDiff(TDiff)
    return TDiff



def extract_features(loader, model1, model2, test=True, diff=False, average=False, feature_concat = True):
    # encoder 1 extracts features from original frames
    # encoder 2 extracts features from spatial difference and temporal difference
    model.eval()

    features = []
    label_lst = []

    i = 0
    with torch.no_grad():
        for data_i in loader:
            # B, N, C, T, H, W
            # N = 2 by default, C is number of channel (C = 3), and T is the number of frames in a video clip
            input_tensor, label = data_i 
            input_tensor = input_tensor.to(torch.device('cuda'))
            gray_input = RGB2Gray(input_tensor)
            spatial_diff = get_spatial_diff(gray_input)
            temporal_diff = get_spatial_diff(gray_input) # the shape of temporal diff should be 
            B, N, C, T, H, W = input_tensor.shape
            print("The shape of the data input_tensor( in form of (B, N, C, T, H, W)) is: ", (B, N, C, T, H, W))
            h = model1(input_tensor.view(B*N, C, T, H, W))
            h_sd = model2(spatial_diff.view(B*N, C, T-1, H, W))
            h_td = model2(temporal_diff.view(B*N, C, T-1, H, W))


            
            # # kind 1
            if test:
                h = h.reshape(B, N, -1) # B, N, D
                if diff: # using difference between frames to do test
                    input_tensor_diff = input_tensor[:,:,:,1:,:,:] - input_tensor[:,:,:,:-1,:,:]
                    h_diff = model(input_tensor_diff.view(B*N, C, T-1, H, W))
                    h_diff = h_diff.reshape(B, N, -1)
                    print("diff")
                    if feature_concat:
                        print("concat features")
                        features.append(torch.cat((h, h_diff), -1))
                    else:
                        print("average features")
                        features.append((h+h_diff)/2)
                elif average:
                    frames_average = torch.mean(input_tensor, dim = 3, keepdim = True)
                    input_tensor_average = torch.repeat_interleave(frames_average, T, dim = 3)
                    h_average = model(input_tensor_average.view(B*N, C, T, H, W))
                    h_average = h_average.reshape(B, N, -1)
                    print("average frame")
                    if feature_concat:
                        print("concat features")
                        features.append(torch.cat((h, h_average), -1))
                    else:
                        print("average features")
                        features.append((h+h_average)/2)
                else:
                    print("original frame")
                    features.append(h)
                label_lst.append(label)
                
            # kind 2
            else:
                if diff:
                    input_tensor_diff = input_tensor[:,:,:,1:,:,:] - input_tensor[:,:,:,:-1,:,:]
                    h_diff = model(input_tensor_diff.view(B*N, C, T-1, H, W))
                    print("diff")
                    if feature_concat:
                        print("concat features")
                        features.append(torch.cat((h, h_diff), -1))
                    else:
                        print("average features")
                        features.append((h+h_diff)/2)
                elif average:
                    frames_average = torch.mean(input_tensor, dim = 3, keepdim = True)
                    input_tensor_average = torch.repeat_interleave(frames_average, T, dim = 3)
                    h_average = model(input_tensor_average.view(B*N, C, T, H, W))
                    print("average frame")
                    if feature_concat:
                        print("concat features")
                        features.append(torch.cat((h, h_average), -1))
                    else:
                        print("average features")
                        features.append((h+h_average)/2)
                else:
                    print("original frame")
                    features.append(h)
                label_lst.append(torch.ones(B,N)*label)

            i += 1
            if i % 10 == 0:
                print(i)
            # if i > 2:
            #     break

        h_total = torch.vstack(features)
        # print(h_total.shape)
        # # kind 1
        if test:
            h_total = torch.mean(h_total, dim=1)
        label_total = torch.vstack(label_lst)



    return h_total, label_total


def perform_knn(model, train_loader, test_loader, k=1, diff=False, average = False, feature_concat = True):
    model.eval()

    ssl_evaluator = Retrieval(model=model, k=k, device=cuda, num_seq=args.num_seq)
    h_train, l_train = extract_features(train_loader, model, diff=diff, average=average, feature_concat=feature_concat)

    train_acc = ssl_evaluator.knn(h_train, l_train, k=1)
    h_test, l_test = extract_features(test_loader, model, diff=diff, average=average, feature_concat=feature_concat)
    acc1, acc5, acc10  = ssl_evaluator.eval(h_test, l_test, l_train)

    # train_acc, val_acc = ssl_evaluator.fit(train_loader, test_loader)
    print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    print(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    print('-----------------')
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    logging.info('-----------------')
    return train_acc, acc1, acc5, acc10



def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    if args.swin:
        if args.which_encoder == 0:
            ckpt_path = os.path.join(ckpt_folder, 'swinTransformer_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 1:
            ckpt_path = os.path.join(ckpt_folder, 'swinTransformer1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 2:
            ckpt_path = os.path.join(ckpt_folder, 'swinTransformer2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
    else:
        if args.which_encoder == 0:
            ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 1:
            ckpt_path = os.path.join(ckpt_folder, 'resnet1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 2:
            ckpt_path = os.path.join(ckpt_folder, 'resnet2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network

    if not args.hmdb:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_retrieval.log'), level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_knn.log'), level=logging.INFO)
    logging.info('Started')

    if args.diff:
        logging.info(f"k-nn accuracy using differences between frames\n")
        if args.feature_concat:
            logging.info(f"concatenating features extracted from input and input_diff\n")
        else:
            logging.info(f"averageing features extracted from input and input_diff\n")
    elif args.average:
        logging.info(f"k-nn accuracy using average across frames\n")
        if args.feature_concat:
            logging.info(f"concatenating features extracted from input and average across frames\n")
        else:
            logging.info(f"averageing features extracted from input and average across frames\n")
    else:
        logging.info(f"k-nn accuracy using original frames \n")
    


    if not args.random:
        logging.info(ckpt_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')


    if args.swin:
        encoder = models.video.swin3d_t()
    else:

        if args.r21d:
            model_name = 'r21d18'
            if not args.kinetics:
                encoder = models.video.r2plus1d_18()
            else:
                encoder = models.video.r2plus1d_18(pretrained=True)
        elif args.mc3:
            model_name = 'mc318'
            if not args.kinetics:
                encoder = models.video.mc3_18()
            else:
                encoder = models.video.mc3_18(pretrained=True)
        elif args.s3d:
            model_name = 's3d'
            if not args.kinetics:
                encoder = models.video.s3d()
            else:
                encoder = models.video.s3d(pretrained=True)
            encoder.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            model_name = 'r3d18'
            if not args.kinetics:
                encoder = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct)
            else:
                encoder = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct, pretrained=True)

    # if not args.kinetics:
    #     resnet = models.video.r3d_18()
    #     # modify model
    #     # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # else:
    #     resnet = models.video.r3d_18(pretrained=True)
    #     # modify model
    #     # resnet.layer4[1].conv2[0] = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

    
    if not args.random and not args.kinetics:
        encoder.load_state_dict(torch.load(ckpt_path)) # load model
    if args.swin:
        encoder.head = torch.nn.Identity()
    else:

        encoder.fc = torch.nn.Identity()


    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(cuda)
    encoder.eval()

    if args.img_size == 224:
        dim = 240
    else:
        dim = 150

    if not args.hmdb:
        logging.info(f"k-nn accuracy performed on ucf \n")
        train_loader = get_data_ucf(batch_size=args.batch_size,
                                    mode='train',
                                    transform_consistent=None,
                                    transform_inconsistent=test_transform(),
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    downsample=args.downsample,
                                    inter_len=args.inter_len,
                                    dim=dim,
                                    frame_root='/data',
                                    # random=True,
                                    test=True
                                    )
        test_loader = get_data_ucf(batch_size=args.batch_size,
                                    mode='test',
                                    transform_consistent=None,
                                    transform_inconsistent=test_transform(),
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    inter_len=args.inter_len,
                                    downsample=args.downsample,
                                    dim=dim,
                                    frame_root='/data',
                                    # random=True,
                                    test=True
                                    )
    else:
        logging.info(f"k-nn accuracy performed on hmdb \n")
        train_loader = get_data_hmdb(batch_size=args.batch_size,
                                    mode='train',
                                    transform_consistent=None,
                                    transform_inconsistent=test_transform(),
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    inter_len=args.inter_len,
                                    downsample=args.downsample,
                                    dim=dim,
                                    frame_root='/data',
                                    # random=True,
                                    test=True
                                    )
        test_loader = get_data_hmdb(batch_size=args.batch_size,
                                    mode='test',
                                    transform_consistent=None,
                                    transform_inconsistent=test_transform(),
                                    seq_len=args.seq_len,
                                    num_seq=args.num_seq,
                                    inter_len=args.inter_len,
                                    downsample=args.downsample,
                                    dim=dim,
                                    frame_root='/data',
                                    # random=True,
                                    test=True
                                    )

    # random weight
    if args.random:
        logging.info(f"k-nn accuracy performed with random weight\n")
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average, args.feature_concat)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average, args.feature_concat)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average, args.feature_concat)




if __name__ == '__main__':
    main()
