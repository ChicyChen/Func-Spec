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

class VICCLR2S2PRVIDIDIG(nn.Module): # ViDiDi as guidance
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
        input_e1, # input_e1.shape = [B, N=2, C, T, H, W]
        input_e2 # input_e2.shape = [B, N=2, C, T, H, W]
    ):
        assert not (self.training and input_e1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        # x1 = x[:,:-1,:,:,:,:] # data to stream 1 with random data augmentation process and ViDiDi scheduling
        # x2 = x[:,1:,:,:,:,:] # data to stream 2 with random data augmentatoon process and ViDiDi scheduling
        B1, N1, C1, T1, H1, W1 = input_e1.size() # shape of input_e1 is [B, N=2, C, T, H, W]
        B2, N2, C2, T2, H2, W2 = input_e2.size() # shape of input_e2, which might be different with the one of e1 due to differernt augmentation;

        # ground truth latents
        hidden1 = flatten(self.encoder1(input_e1.view(B1*N1, C1, T1, H1, W1))) # encoder1 forward
        hidden2 = flatten(self.encoder2(input_e2.view(B2*N2, C2, T2, H2, W2))) # encoder2 forward

        gt_z_all1 = self.projector1(hidden1) # projector1 forward for output of encoder1
        gt_z_all2 = self.projector2(hidden2) # projector2 forward for output of encoder2
        gt_z_all1 = gt_z_all1.reshape(B1, N1, -1) # B, N, D
        gt_z_all2 = gt_z_all2.reshape(B2, N2, -1) # B, N, D

        
        # if rand_diff: # x1, x2 must be derivative augmented
        #     # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
        #     # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
        #     z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
        #     z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

        # else:
        #     if epoch_index%4 == 0: # (d/dx1, d/dx2)
        #         # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
        #         # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
        #         z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
        #         z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

        #     elif epoch_index%4 == 1: # (d/dx1, x2)
        #         # z1 = gt_z_all2[:, :-1, :] # z1 = 0 + f2(x1)
        #         # z2 = gt_z_all1[:, 1:, :] # z2 = f1(x2) + 0
        #         z1 = 0*gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
        #         z2 = gt_z_all1[:, 1:, :] + 0*gt_z_all2[:, 1:, :]

        #     elif epoch_index%4 == 2: # (x1, d/dx2)
        #         # z1 = gt_z_all1[:, :-1, :] # z2 = f1(x1) + 0
        #         # z2 = gt_z_all2[:, 1:, :] # z2 = 0 + f2(x2)
        #         z1 = gt_z_all1[:, :-1, :] + 0*gt_z_all2[:, :-1, :]
        #         z2 = 0*gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]

        #     else: # (x1, x2)
        #         # z1 = gt_z_all1[:, :-1, :] # z1 = f1(x1) + 0
        #         # z2 = gt_z_all1[:, 1:, :] # z2 = f1(x2) + 0
        #         z1 = gt_z_all1[:, :-1, :] + 0*gt_z_all2[:, :-1, :]
        #         z2 = gt_z_all1[:, 1:, :] + 0*gt_z_all2[:, 1:, :]

        z1 = gt_z_all1[:, :-1, :] + gt_z_all2[:, :-1, :]
        z2 = gt_z_all1[:, 1:, :] + gt_z_all2[:, 1:, :]        

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