# VICCLR with double encoder, the loss in defined with the direct summation of the vector output after the projection layer.
# The two encoders have the same architecture but different random initialization(weight are not shared)
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

class SWINCLR2S(nn.Module): # DE for double encoder
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
      temperature = 0.1
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
      self.temperature = temperature
    def loss_fn(self, x, y):
      # x: B, N-1, D; D = output dimension after projection
      # y: B, N-1, D; D = output dimension after projection
        (B, M, D) = x.shape
        x = x.reshape(B*M, D)
        y = y.reshape(B*M, D)

        if self.infonce:
            loss = infoNCE(x, y, temperature=self.temperature)
        else:
            loss = vic_reg_nonorm_loss(x, y, self.mse_l, self.std_l, self.cov_l)

        return loss

    def forward(
        self,
        x # x.shape = B, N, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        B, N, C, T, H, W = x.size()
        x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process

        # ground truth latents
        hiddenf1x1 = flatten(self.encoder1(x1.view(B, C, T, H, W))) # encoder1 forward x1
        hiddenf1x2 = flatten(self.encoder1(x2.view(B, C, T, H, W))) # encoder1 forward x2
        hiddenf2x1 = flatten(self.encoder2(x1.view(B, C, T, H, W))) # encoder2 forward x1
        hiddenf2x2 = flatten(self.encoder2(x2.view(B, C, T, H, W))) # encoder2 forward x2

        gt_z_f1x1 = self.projector1(hiddenf1x1)
        gt_z_f1x2 = self.projector1(hiddenf1x2)
        gt_z_f2x1 = self.projector2(hiddenf2x1)
        gt_z_f2x2 = self.projector2(hiddenf2x2)

        gt_z_f1x1 = gt_z_f1x1.reshape(B, N-1, -1)
        gt_z_f1x2 = gt_z_f1x2.reshape(B, N-1, -1)
        gt_z_f2x1 = gt_z_f2x1.reshape(B, N-1, -1)
        gt_z_f2x2 = gt_z_f2x2.reshape(B, N-1, -1)

        # gt_z_all1 = self.projector1(hidden1) # projector1 forward for output of encoder1
        # gt_z_all2 = self.projector2(hidden2) # projector2 forward for output of encoder2
        # gt_z_all1 = gt_z_all1.reshape(B, N, -1) # B, N, D
        # gt_z_all2 = gt_z_all2.reshape(B, N, -1) # B, N, D
        # # print("shape of gt_z_all is: ", gt_z_all1.shape)

        if self.concat:
            # print("Concatenation")
            # print("shape of gt_z_all after concatenation is: ", gt_z_all_concat.shape)
            z1 = torch.cat((gt_z_f1x1, gt_z_f2x1), dim=-1)
            z2 = torch.cat((gt_z_f1x2, gt_z_f2x2), dim=-1)

        else: #sum
            # print("Summation")
            z1 = gt_z_f1x1 + gt_z_f2x1
            z2 = gt_z_f1x2 + gt_z_f2x2
            

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