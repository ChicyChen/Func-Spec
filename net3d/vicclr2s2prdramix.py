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

class VICCLR2S2PRDRAMIX(nn.Module):
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
        video_dt, # video.shape = B, N, C, T-1, H, W
        video_avg # video.shape = B, N, C, T-1, H, W
    ):
        # print("The shape of video input is: ", video.shape)
        # print("The shape pf video_rand_derivative input is: ", video_rand_derivative.shape)
        assert not (self.training and video_dt.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        assert (video_dt.shape == video_dt.shape), 'The shape of all input must be the same'

        B, N, C, T, H, W = video_dt.size()
        # x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        # x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process
        
        # ground truth latents
        hidden1_dt = flatten(self.encoder1(video_dt.view(B*N, C, T, H, W))) # encoder1 forward derivative augmentation
        hidden1_avg = flatten(self.encoder1(video_avg.view(B*N, C, T, H, W))) # encoder1 forward average augmentation
        hidden2_dt = flatten(self.encoder2(video_dt.view(B*N, C, T, H, W))) # encoder2 forward derivative augmentation
        hidden2_avg = flatten(self.encoder2(video_avg.view(B*N, C, T, H, W))) # encoder2 forward average augmentation

        gt_z_all1_dt = self.projector1(hidden1_dt) # projector1 forward hidden1_dt
        gt_z_all1_avg = self.projector1(hidden1_avg) # projector1 forward hidden1_avg
        gt_z_all2_dt = self.projector2(hidden2_dt) # projector2 forward hidden2_dt
        gt_z_all2_avg = self.projector2(hidden2_avg) # projector2 forward hidden2_avg

        gt_z_all1_dt = gt_z_all1_dt.reshape(B, N, -1) # B, N, D
        gt_z_all1_avg = gt_z_all1_avg.reshape(B, N, -1) # B, N, D
        gt_z_all2_dt = gt_z_all2_dt.reshape(B, N, -1) # B, N, D
        gt_z_all2_avg = gt_z_all2_avg.reshape(B, N, -1) # B, N, D

        z1 = gt_z_all1_dt[:, :-1, :] + gt_z_all2_dt[:, :-1, :] + gt_z_all1_avg[:, :-1, :] + gt_z_all2_avg[:, :-1, :]
        z2 = gt_z_all1_dt[:, 1:, :] + gt_z_all2_dt[:, 1:, :] + gt_z_all1_avg[:, 1:, :] + gt_z_all2_avg[:, 1:, :]            

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