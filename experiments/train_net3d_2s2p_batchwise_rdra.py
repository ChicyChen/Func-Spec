import copy
import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/yehengz/Func-Spec/utils")
sys.path.append("/home/yehengz/Func-Spec/net3d")
sys.path.append("/home/yehengz/Func-Spec/dataload")
sys.path.append("/home/yehengz/Func-Spec/resnet_edit")

from vicclr2srdra import VICCLR2SRDRA
from resnet import r3d_18

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


parser = argparse.ArgumentParser()

parser.add_argument('--frame_root', default='/data', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')
# --frame_root /data

parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)

parser.add_argument('--epochs', default=400, type=int,
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

parser.add_argument('--random', action='store_true') # default is false
parser.add_argument('--num_seq', default=2, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=3, type=int)
parser.add_argument('--inter_len', default=0, type=int)    # does not need to be positive

parser.add_argument('--sym_loss', action='store_true') # default is false

parser.add_argument('--feature_size', default=512, type=int)
parser.add_argument('--projection', default=2048, type=int)
parser.add_argument('--proj_hidden', default=2048, type=int)
parser.add_argument('--proj_layer', default=3, type=int)

parser.add_argument('--mse_l', default=1.0, type=float)
parser.add_argument('--std_l', default=1.0, type=float)
parser.add_argument('--cov_l', default=0.04, type=float)
parser.add_argument('--infonce', action='store_true') #default is false

parser.add_argument('--base_lr', default=4.8, type=float)
# parser.add_argument('--base_lr', default=1.2, type=float)

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

parser.add_argument('--seed', default=233, type = int) # add a seed argument that allows different random initilization of weight
parser.add_argument('--concat', action='store_true') # default value is false, this arugment decide if we are summing two output from each encoders or concatenating them
parser.add_argument('--prob_derivative', default = 0.5, type = float)
parser.add_argument('--prob_average', default = 0.5, type = float) # cry for a better name for this argument!!!
parser.add_argument('--prob_2nd_derivative', default = 0.0, type = float)
parser.add_argument('--first_frame_duplicated', action='store_true')
parser.add_argument('--width_deduction_ratio', default = 1.0, type = float)
parser.add_argument('--stem_deduct', action='store_true') # default is false

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    # warmup_steps = 10 * len(loader)
    warmup_steps = 0
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


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
        video_rd, label = data # B, N, C, T, H, W
        video_ra = copy.deepcopy(video_rd)
        label = label.to(gpu)
        video_rd = video_rd.to(gpu)
        video_ra = video_ra.to(gpu)
        if random.random() < args.prob_derivative:
            video_rd = video_rd[:,:,:,1:,:,:] - video_rd[:,:,:,:-1,:,:]
            if random.random() < args.prob_2nd_derivative:
                video_rd = video_rd[:,:,:,1:,:,:] - video_rd[:,:,:,:-1,:,:]
        
        B, N, C, T, H, W = video_rd.shape # keep shape of two input constant
        if random.random() < args.prob_average:
            if args.first_frame_duplicated:
                frame_average = video_ra[:,:,:,0:1,:,:]  
            else:
                frame_average = torch.mean(video_ra, dim = 3, keepdim = True)
            video_ra = torch.repeat_interleave(frame_average, T, dim = 3)
        else:
            video_ra = video_ra[:,:,:,0:T,:,:]




        lr = adjust_learning_rate(args, optimizer, train_loader, step)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(video_rd, video_ra)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.mean().item()

    return total_loss/num_batches


def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    model_select = VICCLR2SRDRA

    if args.infonce:
        ind_name = 'nce2s'
    else:
        ind_name = 'pcn2s'

    if args.r21d:
        model_name = 'r21d18'
        resnet1 = models.video.r2plus1d_18()
        resnet2 = models.video.r2plus1d_18()
    elif args.mc3:
        model_name = 'mc318'
        resnet1 = models.video.mc3_18()
        resnet2 = models.video.mc3_18()
    elif args.s3d:
        model_name = 's3d'
        resnet1 = models.video.s3d()
        resnet1.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        resnet2 = models.video.s3d()
        resnet2.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        model_name = 'r3d18'
        resnet1 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct)
        resnet2 = r3d_18(width_deduction_ratio = args.width_deduction_ratio, stem_deduct = args.stem_deduct)

    if args.k400:
        dataname = 'k400'
    elif args.mk200:
        dataname = 'mk200'
    elif args.mk400:
        dataname = 'mk400'
    elif args.minik:
        dataname = 'minik'
    else:
        dataname = 'ucf'
    
    if args.concat:
        operation = "_concat"
        print('We are using concatenation.')
    else:
        operation = "_sum"
        print('We are using summation')

    if args.first_frame_duplicated:
        spatial_op = 'duplicated_1st'
    else:
        spatial_op = "averaged"

    ckpt_folder='/data/checkpoints_yehengz/2s2p_batch_guide/%s%s_%s_%s/sym%s_bs%s_lr%s_wd%s_ds%s_sl%s_nw_rand%s_feature_size%s_projection%s_proj_hidden%s_epochs%s_seed%s_op%s_prob_ddt%s_prob_static%s_prob_d2dt2%s_spatial_op_%s_width_deduc_ratio%s_stem_deduct%s' \
        % (dataname, args.fraction, ind_name, model_name, args.sym_loss, args.batch_size, args.base_lr, args.wd, args.downsample, args.seq_len, args.random, args.feature_size, args.projection, args.proj_hidden, args.epochs, args.seed, operation, args.prob_derivative, args.prob_average, args.prob_2nd_derivative, spatial_op, args.width_deduction_ratio, args.stem_deduct)

    # ckpt_folder='/home/siyich/Func-Spec/checkpoints/%s%s_%s_%s/prj%s_hidproj%s_hidpre%s_prl%s_pre%s_np%s_pl%s_il%s_ns%s/mse%s_loop%s_std%s_cov%s_spa%s_rall%s_sym%s_closed%s_sub%s_sf%s/bs%s_lr%s_wd%s_ds%s_sl%s_nw_rand%s' \
    #     % (dataname, args.fraction, ind_name, model_name, args.projection, args.proj_hidden, args.pred_hidden, args.proj_layer, args.predictor, args.num_predictor, args.pred_layer, args.inter_len, args.num_seq, args.mse_l, args.loop_l, args.std_l, args.cov_l, args.spa_l, args.reg_all, args.sym_loss, args.closed_loop, args.sub_loss, args.sub_frac, args.batch_size, args.base_lr, args.wd, args.downsample, args.seq_len, args.random)

    if args.rank == 0:
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'net3d_vic_train.log'), level=logging.INFO)
        logging.info('Started')


    model = model_select(
        resnet1,
        resnet2,
        hidden_layer = 'avgpool',
        feature_size = args.feature_size,
        projection_size = args.projection,
        projection_hidden_size = args.proj_hidden,
        proj_layer = args.proj_layer,
        sym_loss = args.sym_loss,
        mse_l = args.mse_l,
        std_l = args.std_l,
        cov_l = args.cov_l,
        infonce = args.infonce,
        concat = args.concat, # determine if we perform sum or concatenation operation on outputs of f1 and f2
    ).cuda(gpu)
    # sync bn does not works for ode
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_folder, 'net3d_epoch%s.pth.tar' % args.resume_epoch)
        ckpt = torch.load(pretrain_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    assert args.batch_size % args.world_size == 0

    per_device_batch_size = args.batch_size // args.world_size
    # print(per_device_batch_size)

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
                                ddp=True,
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

    train_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    # start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for i in epoch_list:
        train_loss = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler)

        # current_time = time.time()
        if args.rank == 0:
            if train_loss < lowest_loss:
                lowest_loss = train_loss

            train_loss_list.append(train_loss)
            print('Epoch: %s, Train loss: %s' % (i, train_loss))
            logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))



            
            if (i+1)%100 == 0:
                # save your improved network
                # save the weight of encoder1
                checkpoint_path1 = os.path.join(
                    ckpt_folder, 'resnet1_epoch%s.pth.tar' % str(i+1))
                torch.save(resnet1.state_dict(), checkpoint_path1)
                # save the weight of encoder2
                checkpoint_path2 = os.path.join(
                    ckpt_folder, 'resnet2_epoch%s.pth.tar' % str(i+1))
                torch.save(resnet2.state_dict(), checkpoint_path2)

                # save whole model and optimizer
                state = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                checkpoint_path = os.path.join(
                    ckpt_folder, 'net3d_epoch%s.pth.tar' % str(i+1))
                torch.save(state, checkpoint_path)
            # elif (i+1)<100 and (i+1)%10 == 0: # save weight at epoch 10, 20, 30, 40, 50, 60, 70, 80, 90
            #     # save your improved network
            #     # save the weight of encoder1
            #     checkpoint_path1 = os.path.join(
            #         ckpt_folder, 'resnet1_epoch%s.pth.tar' % str(i+1))
            #     torch.save(resnet1.state_dict(), checkpoint_path1)
            #     # save the weight of encoder2
            #     checkpoint_path2 = os.path.join(
            #         ckpt_folder, 'resnet2_epoch%s.pth.tar' % str(i+1))
            #     torch.save(resnet2.state_dict(), checkpoint_path2)

            #     # save whole model and optimizer
            #     state = dict(
            #         model=model.state_dict(),
            #         optimizer=optimizer.state_dict(),
            #     )
            #     checkpoint_path = os.path.join(
            #         ckpt_folder, 'net3d_epoch%s.pth.tar' % str(i+1))
            #     torch.save(state, checkpoint_path)


    if args.rank == 0:
        logging.info('Training from ep %d to ep %d finished' %
            (args.start_epoch, args.epochs))
        logging.info('Best epoch: %s' % best_epoch)

        # save your improved network
        # save the weight of encoder1
        checkpoint_path1 = os.path.join(
            ckpt_folder, 'resnet1_epoch%s.pth.tar' % str(args.epochs))
        torch.save(resnet1.state_dict(), checkpoint_path1)
        # save the weight of encoder2
        checkpoint_path2 = os.path.join(
            ckpt_folder, 'resnet2_epoch%s.pth.tar' % str(args.epochs))
        torch.save(resnet2.state_dict(), checkpoint_path2)
        state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
        checkpoint_path = os.path.join(
            ckpt_folder, 'net3d_epoch%s.pth.tar' % str(args.epochs))
        torch.save(state, checkpoint_path)


        plot_list = range(args.start_epoch, args.epochs)
        # plot training process
        plt.plot(plot_list, train_loss_list, label = 'train')

        plt.legend()
        plt.savefig(os.path.join(
            ckpt_folder, 'epoch%s_bs%s_loss.png' % (args.epochs, args.batch_size)))



if __name__ == '__main__':
    main()

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2p_batchwise_rdra.py --sym_loss --infonce --epochs 400 --seed 233 --prob_derivative 0.75 --prob_average 0.25 --first_frame_duplicated --width_deduction_ratio 1.41 --feature_size 363 --projection 1452 --proj_hidden 1452