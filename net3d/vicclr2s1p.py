# VICCLR with double encoder, the loss in defined with the direct summation or concatenation of the vector output after the projection layer.
# The two encoders have the same architecture but different random initialization(weight are not shared)
# encoder1 have prob = p to see 1st order derivative of video frame and prob = (1-p) to see the video frame.
# encoder2 have prob = p to see average across frames and prob = (1-p) to see the video frame.
# VICCLR2SRD means VICreg, simCLR with 2 Streams and Random Derivative and Random Average across frames
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

class VICCLR2S1P(nn.Module):
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
          self.projector = create_mlp_fn(feature_size, projection_size, projection_hidden_size, proj_layer)
      else:
          self.projector = nn.Identity()

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
        x
    ):
        # print("The shape of video input is: ", video.shape)
        # print("The shape pf video_rand_derivative input is: ", video_rand_derivative.shape)
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'


        B, N, C, T, H, W = x.size()
        # x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        # x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process
        
        # ground truth latents
        hidden1 = flatten(self.encoder1(x.view(B*N, C, T, H, W))) # encoder1 forward
        hidden2 = flatten(self.encoder2(x.view(B*N, C, T, H, W))) # encoder2 forward
        #shapes of hidden1 and hidden2 are [16, 512] 

        if self.concat:
            hidden = torch.cat((hidden1, hidden2), dim=-1) # shape is [16, 1024]
        else:
            hidden = hidden1 + hidden2 # shape is still [16, 512]


        gt_z_all = self.projector(hidden) # projector forward for summation or concatenation of outputs from encoder 1 and encoder 2
        gt_z_all = gt_z_all.reshape(B, N, -1) # B, N, D

            
        z1 = gt_z_all[:, :-1, :]
        z2 = gt_z_all[:, 1:, :]

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