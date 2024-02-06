import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from augmentation import *

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from torchvision.transforms import Compose, Lambda

def get_data_cifar10(transform=None,
                    mode='train',
                    batch_size=16,
                    ddp=False,
                    random=False,
                    test=False,
                    ):

    print('Loading data for "%s" ...' % mode)
    if mode == 'train':
        dataset = CIFAR10Pair(root='/data/CIFAR10', train=True, transform=transform, download=True)
        print("using train split")
    else:
        dataset = CIFAR10Pair(root='/data/CIFAR10', train=False, transform=transform, download=True)
        print("using test split")

    if not ddp:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=True)

    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=128,
                                  pin_memory=True,
                                  drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def get_data_cifar100(transform=None,
                    mode='train',
                    batch_size=16,
                    ddp=False,
                    random=False,
                    test=False,
                    ):

    print('Loading data for "%s" ...' % mode)
    if mode == 'train':
        dataset = CIFAR100Pair(root='/data/CIFAR100', train=True, transform=transform, download=True)
    else:
        dataset = CIFAR100Pair(root='/data/CIFAR100', train=False, transform=transform, download=True)

    if not ddp:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=True)

    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=128,
                                  pin_memory=True,
                                  drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader



class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)


        if self.transform is not None:
          # important! do not mix up! im_1 and im_2 are the same image after different data augmentation,
          # so they have the same label!
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2, target

class CIFAR100Pair(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)

        
        if self.transform is not None:
            im1 = self.transform(img)
            im2 = self.transform(img)
        
        return im1, im2, target