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

from helpers import *

    
# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        clip_size,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        num_layer = 2,
        moving_average_decay = 0.996,
        use_momentum = True,
        asym_loss = False,
        use_projector = True,
        use_simsiam_mlp = False,
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer = hidden_layer, use_simsiam_mlp = use_simsiam_mlp, num_layer = num_layer, use_projector = use_projector)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size) # predict dfference instead

        self.asym_loss = asym_loss

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, clip_size, image_size, image_size, device=device), torch.randn(2, 3, clip_size, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self, step=None, total_steps=None):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder, step, total_steps)

    def loss_fn(self, x, y):
        return normalized_mse_loss(x, y)

    def forward(
        self,
        x # start time's input, B, 2, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        
        image_one, image_two = x[:,0,::], x[:,1,::]

        online_proj_one, _ = self.online_encoder(image_one)
        online_pred_one = self.online_predictor(online_proj_one)

        if not self.asym_loss: # sym loss, two way prediction
            online_proj_two, _ = self.online_encoder(image_two)
            online_pred_two = self.online_predictor(online_proj_two) 

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_two, _ = target_encoder(image_two)
            target_proj_two.detach_()
            target_proj_one, _ = target_encoder(image_one)
            target_proj_one.detach_()

        loss_one = self.loss_fn(online_pred_one, target_proj_two)

        if not self.asym_loss: # sym loss, two way prediction
            loss_two = self.loss_fn(online_pred_two, target_proj_one)
            loss = loss_one + loss_two
        else:
            loss = loss_one * 2
            
        return loss.mean()
