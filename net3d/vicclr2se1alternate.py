# VICCLR with double encoder, the loss in defined with the direct summation or concatenation of the vector output after the projection layer.
# The two encoders have the same architecture but different random initialization(weight are not shared)
# encoder1 sees 1st order derivative of video frame and the video frame, alternating through epochs
# (encoder 1 sees 1st order derivatibe of video frames in epoch 0, 2, 4, 6, 8, 10, ...)
# (encoder 2 sees orginal video frames in epoch 1, 3, 5, 7, 9, ...)
# encoder2 always sees video frame.
# VICCLR2SE1AD means VICreg, simCLR with 2 Streams and Encoder 1 Alternatively sees Derivative of video frames
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

class VICCLR2SE1AD(nn.Module):
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
        video, # video.shape = B, N, C, T-1, H, W
        video_derivative, # video.shape = B, N, C, T-1, H, W
        index # epoch index, starting from 0
    ):
        # print("The shape of video input is: ", video.shape)
        # print("The shape pf video_rand_derivative input is: ", video_rand_derivative.shape)
        assert not (self.training and video.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        assert (video.shape == video_derivative.shape), 'The shape of video input and video_derivative input must be the same'

        B, N, C, T, H, W = video.size() # video and video derivative have the same shape
        # x1 = x[:,0,:,:,:,:] # x1 shape is B, 1, C, T, H, W; x1 is the frame images with first data augmentation process
        # x2 = x[:,1,:,:,:,:] # x2 shape is B, 1, C, T, H, W; x2 is the frame images with second data augmentation process

        
        if index%2 == 0:
            # epoch index = 0, 2, 4, 6, 8, ..., encoder 1 should see the 1st order derivative
            # ground truth latents
            hidden1 = flatten(self.encoder1(video_derivative.view(B*N, C, T, H, W))) # encoder1 forward
            hidden2 = flatten(self.encoder2(video.view(B*N, C, T, H, W))) # encoder2 forward

        else:
            # epoch index = 1, 3, 5, 7, 9, ..., encoder 1 should see the original video frames
            hidden1 = flatten(self.encoder1(video.view(B*N, C, T, H, W))) # encoder1 forward
            hidden2 = flatten(self.encoder2(video.view(B*N, C, T, H, W))) # encoder2 forward


        gt_z_all1 = self.projector1(hidden1) # projector1 forward for output of encoder1
        gt_z_all2 = self.projector2(hidden2) # projector2 forward for output of encoder2
        gt_z_all1 = gt_z_all1.reshape(B, N, -1) # B, N, D
        gt_z_all2 = gt_z_all2.reshape(B, N, -1) # B, N, D 


        if self.concat:
            gt_z_all_concat = torch.cat((gt_z_all1, gt_z_all2), dim=-1) # B, N, 2*D
            z1 = gt_z_all_concat[:, :-1, :]
            z2 = gt_z_all_concat[:, 1:, :]

        else: #sum
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