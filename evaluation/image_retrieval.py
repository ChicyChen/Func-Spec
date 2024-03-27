import os
import sys
import argparse

sys.path.append("/home/yehengz/Func-Spec/utils")
sys.path.append("/home/yehengz/Func-Spec/net3d")
sys.path.append("/home/yehengz/Func-Spec/dataload")

from retrieval import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import torch.nn.functional as F

from dataloader_img import get_data_cifar10, get_data_cifar100
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--cifar10', action='store_true') # default is false
# parser.add_argument('--mnist', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--kinetics', action='store_true')

parser.add_argument('--k', default=1, type=int)

parser.add_argument('--ckpt_folder', default='checkpoints/3dbase_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=400, type=int)

parser.add_argument('--num_seq', default=10, type=int)
parser.add_argument('--seq_len', default=16, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--inter_len', default=0, type=int)
parser.add_argument('--img_size', default=112, type=int)
# parser.add_argument('--num_aug', default=1, type=int)

# parser.add_argument('--img_size', default=112, type=int)
parser.add_argument('--r21d', action='store_true')
parser.add_argument('--mc3', action='store_true')
parser.add_argument('--s3d', action='store_true')


parser.add_argument('--diff', action='store_true')
# parser.add_argument('--which_mode', default='train', type = str)
parser.add_argument('--which_encoder', default = 1, type = int) # default is 1, the only other option is 2
parser.add_argument('--img_num', default=8, type = int) # the value of T in the shape of input tensor
parser.add_argument('--knn_k', default=10, type = int)
parser.add_argument('--knn_t', default=0.07, type = float)
parser.add_argument('--weighted_knn', action='store_true') # default is false

parser.add_argument('--swin', action='store_true') # default is false


# python evaluation/eval_retrieval.py --ckpt_folder checkpoints/ucf1.0_pcn_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse --epoch_num 400
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin_2s/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr7e-05_wd1e-06_ds3_sl8_nw_randFalse_warmupFalse_projection_size2048_tau0.1_epoch_num400_operation_summation --epoch_num 400 --swin --gpu '7'
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin_2s/ucf1.0_nce_swin3dtiny/symTrue_bs64_lr7e-05_wd1e-06_ds3_sl8_nw_randFalse_warmupFalse_projection_size2048_tau0.1_epoch_num400_operation_summation --epoch_num 400 --swin --gpu '7' --which_encoder 2

def test_transform():
    transform = transforms.Compose([
        transforms.Resize(size = 128),
        transforms.RandomResizedCrop(112),
        transforms.Resize(size = (112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform





def extract_features(loader, model, test=True):
    model.eval()

    features = []
    label_lst = []

    i = 0
    with torch.no_grad():
        for data_i in loader:
            # B, N, C, T, H, W
            im1, im2, label= data_i
            #print("++++++++++Shape of label is: ", label.shape)
            #im1 and im2 are in shape [B,C,H,W], lable has a shape of [64]
            im1 = torch.unsqueeze(im1, dim = 1)
            im2 = torch.unsqueeze(im2, dim = 1)
            input_tensor = torch.cat((im1, im2), dim=1) # after this step, input tensor has a shape of [B, N=2, C, H, W]
            input_tensor = torch.unsqueeze(input_tensor, dim = 3) # after this step, input tensor's shape is [B, N=2, C, T=1, H, W]
            if args.img_num > 1:
                l = []
                for j in range(args.img_num):
                    l.append(input_tensor)
                input_tensor = torch.cat(l, dim=3) # in this case, input tensor has a shape of [B, N=2, C, T, H, W]
            input_tensor = input_tensor.to(torch.device('cuda'))
            #check shape, compare with video input_tensor

            B, N, C, T, H, W = input_tensor.shape
            print("=====The shape of input tensor is: ",B, N, C, T, H, W)

            h = model(input_tensor.view(B*N, C, T, H, W))
            # # kind 1
            if test:
                h = h.reshape(B, N, -1) # B, N, D
                #print("The shape of feature h after reshape(B, N, -1) is: ", h.shape) #[16, 2, 512]
                #print("What if I perform the mean operation here: ", torch.mean(h, dim=1).shape)
                features.append(h)
                label_lst.append(label)
                #print("!!!!!!labels added!")
            # kind 2
            else:
                features.append(h)
                label_lst.append(torch.ones(B,N)*label)

            i += 1
            # if i % 10 == 0:
            #     print(i)
            # if i > 2:
            #     break

        h_total = torch.vstack(features)
        #print("The shape of h_total is: ", h_total.shape) #[10000, 2, 512]
        # # kind 1
        if test:
            h_total = torch.mean(h_total, dim=1)
           # print("after the mean operation, the shape of h_total, which is the output shape, is:", h_total.shape)

        label_total = torch.vstack(label_lst)# at this stage, the shape of label total is (datasize/batch_size, batch_size)
        # print("The shape of label total after torch.vstack() is:", label_total.shape)
        label_total = label_total.flatten()# we further need flatten function, this operation is performed by rows,
        # so the output is [row1 row2 row3 ... rown].
        




    return h_total, label_total


def perform_knn(model, train_loader, test_loader, k=1):
    model.eval()

    ssl_evaluator = Retrieval(model=model, k=k, device=cuda, num_seq=args.num_seq)
    h_train, l_train = extract_features(train_loader, model)
    # logging.info(f"The shape of the feature bank we are using is: {h_train.shape}")
    # logging.info(f"The shape of the feature label we are using here is: {l_train.shape}")
    # logging.info(f"The unique value in l_train is: {torch.unique(l_train)}")

    train_acc = ssl_evaluator.knn(h_train, l_train, k=1)
    h_test, l_test = extract_features(test_loader, model)
    # logging.info(f"The shape of test feature we are using is: {h_test.shape}")
    # logging.info(f"The shape of l_test is: {l_test.shape}")
    # logging.info(f"The unique value in l_test is: {torch.unique(l_test)}")
    acc1, acc5, acc10  = ssl_evaluator.eval(h_test, l_test, l_train)
    # print("Here, the input of eval(feature, --, --) is h_test, the shape of h_test is: ", h_test.shape)

    #train_acc, val_acc = ssl_evaluator.fit(train_loader, test_loader)
    print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    print(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    print('-----------------')
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    logging.info('-----------------')
    return acc1, acc5, acc10


def test(model, train_loader, test_loader, args):
    model.eval()
    classes = len(test_loader.dataset.classes)
    total_top1, total_top5, total_top10, total_num, feature_bank, feature_labels = 0.0, 0.0, 0.0, 0, [], []
    with torch.no_grad():
        #generate feature bank
        for data_i in train_loader:
            im1, im2, label = data_i
            label = label.to(torch.device('cuda'))
            im1 = torch.unsqueeze(im1, dim=1)
            im2 = torch.unsqueeze(im2, dim=1)
            input_tensor = torch.cat((im1, im2), dim=1)
            input_tensor = torch.unsqueeze(input_tensor, dim = 3)
            if args.img_num > 1:
                l = []
                for j in range(args.img_num):
                    l.append(input_tensor)
                input_tensor = torch.cat(l, dim = 3)
            input_tensor = input_tensor.to(torch.device('cuda'))

            B, N, C, T, H, W = input_tensor.shape

            h = model(input_tensor.view(B*N, C, T, H, W))
            h = h.reshape(B, N, -1) # [16, 2, 512]
            h = torch.mean(h, dim=1) # [16, 512]
            #print("after the mean operation, the shape of h is: ", h.shape)
            h = torch.nn.functional.normalize(h, dim=1)
            feature_bank.append(h)
            feature_labels.append(label)

        feature_bank = torch.vstack(feature_bank).t()
        #print("The shape of my feature bank is: ", feature_bank.shape) # expected [512, num_data]
        feature_labels = torch.vstack(feature_labels)
        feature_labels = feature_labels.flatten()
        #print("The device that feature labeles on is:", feature_labels.device)
        #print("The shape of my feature labels is: ", feature_labels.shape) # expected [num_data]


        for data_test in test_loader:
            im1, im2, label = data_test
            label = label.to(torch.device('cuda'))
            im1 = torch.unsqueeze(im1, dim=1)
            im2 = torch.unsqueeze(im2, dim=1)
            input_tensor = torch.cat((im1, im2), dim=1)
            input_tensor = torch.unsqueeze(input_tensor, dim = 3)
            if args.img_num > 1:
                l = []
                for j in range(args.img_num):
                    l.append(input_tensor)
                input_tensor = torch.cat(l, dim = 3)
            input_tensor = input_tensor.to(torch.device('cuda'))

            B, N, C, T, H, W = input_tensor.shape

            h = model(input_tensor.view(B*N, C, T, H, W))
            h = h.reshape(B, N, -1) # [16, 2, 512]
            h = torch.mean(h, dim=1) # [16, 512]
            h = torch.nn.functional.normalize(h, dim=1)

            pred_labels = weighted_knn_predict(h, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)


            total_num = total_num + im1.size(0)
            total_top1 = total_top1 + (pred_labels[:, 0] == label).float().sum().item()
            total_top5 = total_top5 + torch.sum((pred_labels[:, :5] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top10 = total_top10 + torch.sum((pred_labels[:, :10] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    acc1 = 100 * total_top1 / total_num
    acc5 = 100 * total_top5 / total_num
    acc10 = 100 * total_top10 / total_num

    print(f"k-nn accuracy k= {args.knn_k} for test split: {acc1}, {acc5}, {acc10} \n")
    print('-----------------')
    logging.info(f"k-nn accuracy k= {args.knn_k} for test split: {acc1}, {acc5}, {acc10} \n")
    logging.info('-----------------')

    return acc1, acc5, acc10

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def weighted_knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels



def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    if args.swin:
        if args.which_encoder == 1:
            ckpt_path = os.path.join(ckpt_folder, 'swinTransformer1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 2:
            ckpt_path = os.path.join(ckpt_folder, 'swinTransformer2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
    else:
        if args.which_encoder == 1:
            ckpt_path = os.path.join(ckpt_folder, 'resnet1_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network
        elif args.which_encoder == 2:
            ckpt_path = os.path.join(ckpt_folder, 'resnet2_epoch%s.pth.tar' % args.epoch_num) # path to the weight of pretrain network

    if args.cifar10:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'corrected_cifar10_knn.log'), level=logging.INFO)
        # I changed the name of the log file to TT, the right one for storing all the results should be 'corrected_cifar10_knn.log'
    else:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'CIFAR100_knn.log'), level=logging.INFO)
    logging.info('Started')
    if args.weighted_knn:
        logging.info('The KNN used here is the WEIGHTED KNN')
        logging.info(f'k= {args.knn_k}, temperature= {args.knn_t}')
    else:
        logging.info('The KNN used here is OUR KNN')
    logging.info('Test when using features from ONLY ONE encoder')
    if args.cifar10:
        logging.info("The dataset used here is CIFAR10.")
    else:
        logging.info("The dataset used here is CIFAR100.")

    if not args.random:
        logging.info(ckpt_path)

    if args.img_num > 1:
        logging.info("We concatenate the same input tensor for %s times, so the shape of input tensor is [B, N=2, C, T=%s, H, W]" % (args.img_num, args.img_num))

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
                encoder = models.video.r3d_18()
            else:
                encoder = models.video.r3d_18(pretrained=True)

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

    # if args.img_size == 224:
    #     dim = 240
    # else:
    #     dim = 150

    if args.cifar10:
        logging.info(f"k-nn accuracy performed on cifar10 \n")
        train_loader = get_data_cifar10(transform=test_transform(),
                                    mode='train', 
                                    batch_size = args.batch_size,
                                    test=True,
                                    )
        
        test_loader = get_data_cifar10(transform=test_transform(),
                                    mode='test',
                                    batch_size = args.batch_size,
                                    test=True,
                                    )
    else:
        logging.info(f"k-nn accuracy performed on CIFAR100 \n")
        train_loader = get_data_cifar100(transform=test_transform(),
                                    mode='train',
                                    batch_size = args.batch_size,
                                    test=True,
                                    )
        
        test_loader = get_data_cifar100(transform=test_transform(),
                                    mode='test',
                                    batch_size = args.batch_size,
                                    test=True,
                                    )


    # random weight
    if args.random:
        logging.info(f"k-nn accuracy performed with random weight\n")
        if args.weighted_knn:
            test(encoder, train_loader, test_loader, args)
        else:
            perform_knn(encoder, train_loader, test_loader, args.k)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        if args.weighted_knn:
            test(encoder, train_loader, test_loader, args)
        else:
            perform_knn(encoder, train_loader, test_loader, args.k)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        if args.weighted_knn:
            test(encoder, train_loader, test_loader, args)
        else:
            perform_knn(encoder, train_loader, test_loader, args.k)




if __name__ == '__main__':
    main()

# Here is the python command that lead to test results based on the feature extracted from a single encoder using img_num(T) = 1
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --which_encoder 2
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --which_encoder 2
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --which_encoder 2

# Here is the python command that lead to test results based on the feature extracted from a single encoder using img_num(T) = 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --which_encoder 2 --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --which_encoder 2 --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --which_encoder 2 --img_num 8
    
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn 
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --concat --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --concat --weighted_knn
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --concat --weighted_knn
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 8
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 8
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 15
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 15
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --which_encoder 2 --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 15
# python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407 --epoch_num 400 --cifar10 --concat --weighted_knn --img_num 15