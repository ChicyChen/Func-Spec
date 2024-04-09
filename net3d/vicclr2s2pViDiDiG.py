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

class VICCLR2S2PVIDIDIG(nn.Module): # ViDiDi as guidance
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
      # different projector followed by two different encoder*(same architecture but with different initialization)

      self.sym_loss = sym_loss

      self.mse_l = mse_l
      self.std_l = std_l
      self.cov_l = cov_l

      self.infonce = infonce

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
        x, # x.shape = [B, N=2, C, T, H, W]
        rand_diff, # boolean, if data take random differerntiation in ViDiDi schedule
        epoch_index # use epoch index to track ViDiDi scheduler
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # x1 = x[:,:-1,:,:,:,:] # data to stream 1 with random data augmentation process and ViDiDi scheduling
        # x2 = x[:,1:,:,:,:,:] # data to stream 2 with random data augmentatoon process and ViDiDi scheduling
        B, N, C, T, H, W = x.size() # shape of x is [B, N=2, C, T, H, W]

        # ground truth latents
        hidden1 = flatten(self.encoder1(x.view(B*N, C, T, H, W))) # encoder1 forward
        hidden2 = flatten(self.encoder2(x.view(B*N, C, T, H, W))) # encoder2 forward

        gt_z_all1 = self.projector1(hidden1) # projector1 forward for output of encoder1
        gt_z_all2 = self.projector2(hidden2) # projector2 forward for output of encoder2
        gt_z_all1 = gt_z_all1.reshape(B, N, -1) # B, N, D
        gt_z_all2 = gt_z_all2.reshape(B, N, -1) # B, N, D

        
        if rand_diff: # x1, x2 must be derivative augmented
            # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
            # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
            z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
            z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

        else:
            if epoch_index%4 == 0: # (d/dx1, d/dx2)
                # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
                # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
                z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
                z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

            elif epoch_index%4 == 1: # (d/dx1, x2)
                # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
                # z2 = gt_z_all1[:, 1:, :] # z2 = f1(x2) + 0
                z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
                z2 = gt_z_all1[:, 1:, :] + 0*gt_z_all2[:, 1:, :]

            elif epoch_index%4 == 2: # (x1, d/dx2)
                # z1 = gt_z_all1[:, :-1, :] # z2 = f1(x1) + 0
                # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
                z1 = gt_z_all1[:, :-1, :] + 0*gt_z_all2[:, :-1, :]
                z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

            else: # (x1, x2)
                # z1 = gt_z_all1[:, :-1, :] # z1 = f1(x1) + 0
                # z2 = gt_z_all1[:, 1:, :] # z2 = f1(x2) + 0
                z1 = gt_z_all1[:, :-1, :] + 0*gt_z_all2[:, :-1, :]
                z2 = gt_z_all1[:, 1:, :] + 0*gt_z_all2[:, 1:, :]

            # z1 = gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
            # z2 = gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]        

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


    # old forward function and it is buggy. yeheng make a copy of it to understand how to add torch.no_grad()
    # def forward(
    #     self,
    #     x, # x.shape = [B, N=2, C, T, H, W]
    #     rand_diff, # boolean, if data take random differerntiation in ViDiDi schedule
    #     epoch_index, # use epoch index to track ViDiDi scheduler
    #     gpu
    # ):
    #     assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

    #     x1 = x[:,:-1,:,:,:,:] # data to stream 1 with random data augmentation process and ViDiDi scheduling
    #     x2 = x[:,1:,:,:,:,:] # data to stream 2 with random data augmentatoon process and ViDiDi scheduling
    #     B, N, C, T, H, W = x.size() # shape of x is [B, N=2, C, T, H, W]

    #     # ground truth latents, and then forward it by projector
    #     if rand_diff: # x1, x2 must be derivative augmented
    #         input_f1 = torch.zeros(x.size())
    #         input_f1 = input_f1.to(gpu)
    #         input_f2 = x
    #     else:
    #         if epoch_index%4 == 0: # (d/dx1, d/dx2)
    #             input_f1 = torch.zeros(x.size())
    #             input_f1 = input_f1.to(gpu)
    #             input_f2 = x
    #         elif epoch_index%4 == 1: # (d/dx1, x2)
    #             input_f1 = torch.cat((torch.zeros(x2.size()).to(gpu), x2), dim=1)
    #             input_f2 = torch.cat((x1, torch.zeros(x1.size()).to(gpu)), dim=1)
    #         elif epoch_index%4 == 2: # (x1, d/dx2)
    #             input_f1 = torch.cat((x1, torch.zeros(x1.size()).to(gpu)), dim=1)
    #             input_f2 = torch.cat((torch.zeros(x2.size()).to(gpu), x2), dim=1)
    #         else: # (x1, x2)
    #             input_f1 = x
    #             input_f2 = torch.zeros(x.size())
    #             input_f2 = input_f2.to(gpu)
            


    #     # ground truth latents
    #     hidden1 = flatten(self.encoder1(input_f1.view(B*N, C, T, H, W))) # encoder1 forward
    #     hidden2 = flatten(self.encoder2(input_f2.view(B*N, C, T, H, W))) # encoder2 forward

    #     gt_z_all1 = self.projector1(hidden1) # projector1 forward for output of encoder1
    #     gt_z_all2 = self.projector2(hidden2) # projector2 forward for output of encoder2
    #     gt_z_all1 = gt_z_all1.reshape(B, N, -1) # B, N, D
    #     gt_z_all2 = gt_z_all2.reshape(B, N, -1) # B, N, D

    #     z1 = gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
    #     z2 = gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

    #     # no predictor, VICReg or SimCLR
    #     loss_one = self.loss_fn(z1, z2)
    #     if self.sym_loss:
    #         loss_two = self.loss_fn(z2, z1)
    #         loss = loss_one + loss_two
    #     else:
    #       loss = loss_one * 2
    
    #     return loss