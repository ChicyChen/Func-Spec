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

class VICCLR(nn.Module):
    def __init__(
        self,
        net,
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

        self.encoder = NetHook(net, hidden_layer)

        if proj_layer > 0:
            create_mlp_fn = MLP
            self.projector = create_mlp_fn(feature_size, projection_size, projection_hidden_size, proj_layer)
        else:
            self.projector = nn.Identity()

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

        # ground truth latents
        hidden = flatten(self.encoder(x.view(B*N, C, T, H, W))) # encoder forward

        gt_z_all = self.projector(hidden) # projector forward
        gt_z_all = gt_z_all.reshape(B, N, -1) # B, N, D

        # no predictor, VICReg or SimCLR
        loss_one = self.loss_fn(gt_z_all[:, :-1, :], gt_z_all[:, 1:, :]) 
        if self.sym_loss:
            loss_two = self.loss_fn(gt_z_all[:, 1:, :], gt_z_all[:, :-1, :])
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
