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
# here are the paths to the weights of different models
# For experiment1: 2Streams
#/data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42
#/data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407
# checkpoints/2S/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse/encoder1
# checkpoints/2S/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse/encoder2
# checkpoints/2S/VICReg_ucf1.0_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse/encoder1
# checkpoints/2S/VICReg_ucf1.0_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse/encoder2
# For experiment2: No-Weight_Sharing
# checkpoints/NWS/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair0/encoder1
# checkpoints/NWS/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair0/encoder2
# checkpoints/NWS/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair1/encoder1
# checkpoints/NWS/SimCLR_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair1/encoder2
# checkpoints/NWS/VICReg_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair0/encoder1
# checkpoints/NWS/VICReg_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair0/encoder2
# checkpoints/NWS/VICReg_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair1/encoder1
# checkpoints/NWS/VICReg_ucf1.0_r3d18/symTrue_bs64_lr1.2_wd1e-06_ds3_sl8_nw_randFalse_fixed_pair1/encoder2

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

# Let's assume diff and average can not be both true


parser.add_argument('--seed', default = 233, type = int) # seed used during training
parser.add_argument('--swin', action='store_true') # default is false
parser.add_argument('--width_deduction_ratio', default=1.0, type = float)
parser.add_argument('--stem_deduct', action='store_true') # default is false

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400
# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400
# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --which_encoder 2
def test_transform():
    transform = transforms.Compose([
        Scale(size=128),
        # RandomCrop(size=args.img_size, consistent=True),
        RandomSizedCrop(size=args.img_size, consistent=True),
        Scale(size=(112, 112)),
        ToTensor(),
        Normalize()
    ])
    return transform


def extract_features(loader, model1, model2, test=True):
    model1.eval()
    model2.eval()

    features = []
    label_lst = []

    i = 0
    with torch.no_grad():
        for data_i in loader:
            # B, N, C, T, H, W
            # N = 2 by default, C is number of channel (C = 3), and T is the number of frames in a video clip
            input_tensor, label = data_i
            input_tensor = input_tensor.to(torch.device('cuda'))
            frames_average = torch.mean(input_tensor, dim = 3, keepdim = True)
            B, N, C, T, H, W = input_tensor.shape
            print("The shape of the data input_tensor( in form of (B, N, C, T, H, W)) is: ", (B, N, C, T, H, W))
            input_tensor_diff = input_tensor[:,:,:,1:,:,:] - input_tensor[:,:,:,:-1,:,:] # dX/dt, T = T-1
            print("The shape of input_tensor_diff is: ", input_tensor_diff.shape)
            input_tensor_average = torch.repeat_interleave(frames_average, (T-1), dim = 3)
            print("The shape of input_tensor_average is: ", input_tensor_average.shape)

            h1_diff = model1(input_tensor_diff.view(B*N, C, T-1, H, W))
            h1_average = model1(input_tensor_average.view(B*N, C, T-1, H, W))
            h2_diff = model2(input_tensor_diff.view(B*N, C, T-1, H, W))
            h2_average = model2(input_tensor_average.view(B*N, C, T-1, H, W))
            
            # kind 1
            if test:
                h1_diff = h1_diff.reshape(B, N, -1)
                h1_average = h1_average.reshape(B, N, -1)
                h2_diff = h2_diff.reshape(B, N, -1)
                h2_average = h2_average.reshape(B, N, -1)
                h = h1_diff + h1_average + h2_diff + h2_average
                features.append(h)
                label_lst.append(label)
            # kind 2
            else:
                h = h1_diff + h1_average + h2_diff + h2_average
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


def perform_knn(model1, model2, train_loader, test_loader, k=1):
    model1.eval()
    model2.eval()

    ssl_evaluator = Retrieval2Encoders(model1=model1, model2=model2, k=k, device=cuda, num_seq=args.num_seq)
    h_train, l_train = extract_features(train_loader, model1, model2)

    train_acc = ssl_evaluator.knn(h_train, l_train, k=1)
    h_test, l_test = extract_features(test_loader, model1, model2)
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
        ckpt_path1 = os.path.join(ckpt_folder, 'swinTransformer1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        ckpt_path2 = os.path.join(ckpt_folder, 'swinTransformer2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
    else:
        ckpt_path1 = os.path.join(ckpt_folder, 'resnet1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        ckpt_path2 = os.path.join(ckpt_folder, 'resnet2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network

    if not args.hmdb:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_retrieval.log'), level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_knn.log'), level=logging.INFO)
    logging.info('Started')
    logging.info('Test when using features from both encoders')
    logging.info('The extracted features is computed as f_1(dt) + f_1(avg) + f_2(dt) + f_2(avg)')
        
    if not args.random:
        logging.info(ckpt_path1)
        logging.info(ckpt_path2)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    if args.swin:
        encoder1 = models.video.swin3d_t()
        encoder2 = models.video.swin3d_t()
    else:
        if args.r21d:
            model_name = 'r21d18'
            if not args.kinetics:
                encoder1 = models.video.r2plus1d_18()
                encoder2 = models.video.r2plus1d_18()
            else:
                encoder1 = models.video.r2plus1d_18(pretrained=True)
                encoder2 = models.video.r2plus1d_18(pretrained=True)
        elif args.mc3:
            model_name = 'mc318'
            if not args.kinetics:
                encoder1 = models.video.mc3_18()
                encoder2 = models.video.mc3_18()
            else:
                encoder1 = models.video.mc3_18(pretrained=True)
                encoder2 = models.video.mc3_18(pretrained=True)
        elif args.s3d:
            model_name = 's3d'
            if not args.kinetics:
                encoder1 = models.video.s3d()
                encoder2 = models.video.s3d()
            else:
                encoder1 = models.video.s3d(pretrained=True)
                encoder2 = models.video.s3d(pretrained=True)
            encoder1.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            encoder2.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            model_name = 'r3d18'
            if not args.kinetics:
                encoder1 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct)
                encoder2 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct)
            else:
                encoder1 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct, pretrained=True)
                encoder2 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct, pretrained=True)

    # if not args.kinetics:
    #     resnet = models.video.r3d_18()
    #     # modify model
    #     # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # else:
    #     resnet = models.video.r3d_18(pretrained=True)
    #     # modify model
    #     # resnet.layer4[1].conv2[0] = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

    if not args.random and not args.kinetics:
        encoder1.load_state_dict(torch.load(ckpt_path1)) # load model1
        encoder2.load_state_dict(torch.load(ckpt_path2)) # load 
    if args.swin:
        encoder1.head = torch.nn.Identity()
        encoder2.head = torch.nn.Identity()
    else:
        encoder1.fc = torch.nn.Identity()
        encoder2.fc = torch.nn.Identity()

    encoder1 = nn.DataParallel(encoder1)
    encoder2 = nn.DataParallel(encoder2)
    encoder1 = encoder1.to(cuda)
    encoder2 = encoder2.to(cuda)
    encoder1.eval()
    encoder2.eval()

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
        perform_knn(encoder1, encoder2, train_loader, test_loader, args.k)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        perform_knn(encoder1,encoder2, train_loader, test_loader, args.k)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        perform_knn(encoder1, encoder2, train_loader, test_loader, args.k)




if __name__ == '__main__':
    main()


# python evaluation/eval_retrieval_self_emergence.py --ckpt_folder /data/checkpoints_yehengz/2s2p_rdra_mixed/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_Summation_prob_derivative1.0_prob_average1.0_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --width_deduction_ratio 1.41