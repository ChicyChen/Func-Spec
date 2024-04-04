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


parser.add_argument('--diff', action='store_true') # default is False
parser.add_argument('--average', action='store_true') # default is False
# Let's assume diff and average can not be both true

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
        ToTensor(),
        Normalize()
    ])
    return transform


def extract_features(loader, model, test=True, diff=False, average=False):
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
            frames_average = torch.mean(input_tensor, dim = 2, keepdim = True)
            B, N, C, T, H, W = input_tensor.shape
            print("The shape of the data input_tensor( in form of (B, N, C, T, H, W)) is: ", (B, N, C, T, H, W))
            input_tensor_diff = input_tensor[:,:,:,1:,:,:] - input_tensor[:,:,:,:-1,:,:] # dX/dt, T = T-1
            print("The shape of input_tensor_diff is: ", input_tensor_diff.shape)
            input_tensor_average = torch.repeat_interleave(frames_average, T, dim = 2)
            print("The shape of input_tensor_average is: ", input_tensor_average.shape)

            h = model(input_tensor.view(B*N, C, T, H, W))
            h_diff = model(input_tensor_diff.view(B*N, C, T-1, H, W))
            h_average = model(input_tensor_average.view(B*N, C, T, H, W))
            # # kind 1
            if test:
                h = h.reshape(B, N, -1) # B, N, D
                h_diff = h_diff.reshape(B, N, -1)
                h_average = h_average.reshape(B, N, -1)
                if diff:
                    print("diff")
                    features.append(torch.cat((h, h_diff), -1))
                elif average:
                    print("average")
                    features.append(torch.cat((h, h_average), -1))
                else:
                    print("0")
                    features.append(h)

                # if not diff:
                #     print("0")
                #     features.append(h)
                # else:
                #     print("1")
                #     features.append(torch.cat((h, h_diff), -1))
                label_lst.append(label)
            # kind 2
            else:
                if diff:
                    print("diff")
                    features.append(torch.cat((h, h_diff), -1))
                elif average:
                    print("average")
                    features.append(torch.cat((h, h_average), -1))
                else:
                    print("0")
                    features.append(h)

                # if not diff:
                #     print("0")
                #     features.append(h)
                # else:
                #     print("1")
                #     features.append(torch.cat((h, h_diff), -1))
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


def perform_knn(model, train_loader, test_loader, k=1, diff=False, average = False):
    model.eval()

    ssl_evaluator = Retrieval(model=model, k=k, device=cuda, num_seq=args.num_seq)
    h_train, l_train = extract_features(train_loader, model, diff=diff, average=average)

    train_acc = ssl_evaluator.knn(h_train, l_train, k=1)
    h_test, l_test = extract_features(test_loader, model, diff=diff, average=average)
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
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        perform_knn(encoder, train_loader, test_loader, args.k, args.diff, args.average)




if __name__ == '__main__':
    main()

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0001_wd1e-06_ds3_sl8_nw_randFalse --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0005_wd1e-06_ds3_sl8_nw_randFalse --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0001_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.2 --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0001_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.3 --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0001_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size4096_tau0.1 --epoch_num 100 --gpu '7' --swin
    
# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr1e-05_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.1 --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0002_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.1 --epoch_num 100 --gpu '7' --swin

# python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr0.0003_wd1e-06_ds3_sl8_nw_randFalse_warmupTrue_projection_size2048_tau0.1 --epoch_num 100 --gpu '7' --swin