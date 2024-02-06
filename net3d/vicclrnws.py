import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

import numpy as np
from operator import add

import torch.distributed as dist

from helpers import *


# main class

class VICCLRNWS(nn.Module):
    def __init__(
        self,
        net1,
        net2,
        hidden_layer = -2,
        feature_size = 2048,
        projection_size = 8192,
        projection_hidden_size = 8192,
        proj_layer = 2,
        sym_loss = False,
        mse_l = 1.0,
        std_l = 1.0,
        cov_l = 0.04,
        infonce = False,
    ):
        super().__init__()

        self.encoder1 = NetHook(net1, hidden_layer)
        self.encoder2 = NetHook(net2, hidden_layer)

        if proj_layer > 0:
            create_mlp_fn = MLP
            self.projector1 = create_mlp_fn(feature_size, projection_size, projection_hidden_size, proj_layer)
            self.projector2 = create_mlp_fn(feature_size, projection_size, projection_hidden_size, proj_layer)
        else:
            self.projector1 = nn.Identity()
            self.projector2 = nn.Identity()

        self.sym_loss = sym_loss

        self.mse_l = mse_l
        self.std_l = std_l
        self.cov_l = cov_l

        self.infonce = infonce

    def loss_fn(self, x, y):
        # x: B, N-1, D
        # y: B, N-1, D
        (B, M, D) = x.shape
        x = x.reshape(B*M, D)
        y = y.reshape(B*M, D)

        if self.infonce:
            loss = infoNCE(x, y)
        else:
            loss = vic_reg_nonorm_loss(x, y, self.mse_l, self.std_l, self.cov_l)

        return loss

    def forward(
        self,
        x # B, N, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        B, N, C, T, H, W = x.size()

        # Reading through the pytorch website of 3DResNet18, I confirmed that the output dimensions are permuted to (..., C, T, H, W) tensors.
        # By having the match of dimension, the code follows using two encoders that do not share weight with each other to encode (x1, x2)

        x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process




        # ground truth latents
        hidden1 = flatten(self.encoder1(x1.view(B*1, C, T, H, W))) # encoder1 forward of x1, now N = 1 so B*N becomes B*1
        hidden2 = flatten(self.encoder2(x2.view(B*1, C, T, H, W))) # encoder2 forward for x2, now N = 1 so B*N becomes B*1

        gt_z1 = self.projector1(hidden1) # projector1 forward after the encoder1 forward
        gt_z2 = self.projector2(hidden2) # projector2 forward after the encoder2 forward

        gt_z1 = gt_z1.reshape(B, 1, -1) # B, 1, D
        gt_z2 = gt_z2.reshape(B, 1, -1) # B, 1, D

        # no predictor, VICReg or SimCLR
        loss_one = self.loss_fn(gt_z1, gt_z2)
        if self.sym_loss:
            loss_two = self.loss_fn(gt_z2, gt_z1)
            loss = loss_one + loss_two
        else:
            loss = loss_one * 2
        return loss



class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]