
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
# the VICCLRSDTD class is a two stream self-supervised learning model that inspired by the human visual system.
# Further inspired by the effective design in tianmouc, the model split into two stream.
# One stream sees the augmented data in a conventional ssl manner, 
# and the other stream sees both the spatial and temporal difference (SD and TD) of the augmented data
# The one sees SD and TD hopefully can serve as the dorsal path, while the other one hopefully can serve as the ventral path

class VICCLRSDTD(nn.Module): 
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
      concat = False,
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
      # different projector followed by two different encoder*(same architecture but with different initialization)

      self.sym_loss = sym_loss

      self.mse_l = mse_l
      self.std_l = std_l
      self.cov_l = cov_l

      self.infonce = infonce
      self.concat = concat

    def loss_fn(self, x, y):
      # x: B, N-1, D; D = output dimension after projection
      # y: B, N-1, D; D = output dimension after projection
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
        x , # x.shape = B, N, C, T, H, W
        x_sd, # spatial difference in the shape of B, N, C=3, T, H, W
        x_td # temporal differnce in the shape of B, N, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # do we need these 3 conditions?
        assert (x.shape == x_sd.shape), 'the shape of the data does not match the shape of its spatial difference'
        # assert (x.shape == x_td.shape), 'the shape of the data does not match the shape of its temporal difference'
        # assert (x_sd.shape == x_td.shape), 'the shape of the spatial difference does not match the shape of temporal difference'

        B, N, C, T, H, W = x.size()
        # x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        # x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process

        # ground truth latents
        hidden = flatten(self.encoder1(x.view(B*N, C, T, H, W))) # encoder 1 process original data
        hidden_sd = flatten(self.encoder2(x_sd.view(B*N, C, T, H, W))) # encoder 2 process spatial difference
        hidden_td = flatten(self.encoder2(x_td.view(B*N, C, T-1, H, W))) # encoder 2 process spatial difference

        gt_z_all1 = self.projector1(hidden)
        gt_z_all_sd = self.projector2(hidden_sd)
        gt_z_all_td = self.projector2(hidden_td)


        gt_z_all1 = gt_z_all1.reshape(B, N, -1) # B, N, D
        gt_z_all_sd = gt_z_all_sd.reshape(B, N, -1) # B, N, D
        gt_z_all_td = gt_z_all_td.reshape(B, N, -1) # B, N, D


        if self.concat:
            # print("Concatenation")
            gt_z_all_concat = torch.cat((gt_z_all1, gt_z_all_sd, gt_z_all_td), dim=-1) # B, N, #*D
            z1 = gt_z_all_concat[:, :-1, :]
            z2 = gt_z_all_concat[:, 1:, :]

        else: #sum
            # print("Summation")
            z1 = 0.5*gt_z_all1[:, :-1, :] + 0.25*gt_z_all_sd[:, :-1, :] + 0.25*gt_z_all_td[:, :-1, :]
            z2 = 0.5*gt_z_all1[:, 1:, :] + 0.25*gt_z_all_sd[:, 1:, :] + 0.25*gt_z_all_td[:, :-1, :]
            

        # no predictor, VICReg or SimCLR
        loss_one = self.loss_fn(z1, z2)
        if self.sym_loss:
            loss_two = self.loss_fn(z2, z1)
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

