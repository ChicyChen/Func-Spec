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

from torchvision.transforms import Compose, Lambda

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        

def pil_loader_resize(path, dim=150):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img.convert('RGB')
            width, height = img.size 
            ratio = min(width, height) / 128
            newsize = (int(width / ratio), int(height / ratio))
            # img.thumbnail((dim,dim), Image.Resampling.LANCZOS)
            img = img.resize(newsize)
            # print(img.size)
            return img

def get_data_ssv2(transform_consistent=None,
                transform_inconsistent=None,
                mode='train', 
                seq_len=8, 
                num_seq=2, 
                downsample=1, 
                return_label=True, 
                batch_size=16, 
                dim=150,
                csv_root='/faststorage/SSv2', # do not need to get this input
                frame_root='/faststorage/SSv2/frames',
                ddp=False,
                random=False,
                test=True,
                inter_len=0, # num of frames (after downsampling) between two clips
                fraction=1.0
                ):
    print('Loading data for "%s" ...' % mode)
    dataset = SSV2(mode=mode,
                        transform_consistent=transform_consistent,
                        transform_inconsistent=transform_inconsistent,
                        seq_len=seq_len,
                        num_seq=num_seq,
                        downsample=downsample,
                        return_label=return_label,
                        dim=dim,
                        csv_root='/faststorage/SSv2',
                        frame_root='/faststorage/SSv2/frames',
                        random=random,
                        test=test,
                        inter_len=inter_len,
                        fraction=fraction
                        )
    if not ddp:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=True)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=128,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=128,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def get_data_ssv2_sub(transform_consistent=None,
                transform_inconsistent=None,
                mode='train', 
                seq_len=8, 
                num_seq=2, 
                downsample=1, 
                return_label=True, 
                batch_size=16, 
                dim=150,
                csv_root='/faststorage/SSv2',
                frame_root='/faststorage/SSv2/frames',
                ddp=False,
                random=False,
                test=True,
                inter_len=0, # num of frames (after downsampling) between two clips
                fraction=1.0
                ):
    print('Loading data for "%s" ...' % mode)
    dataset = SSV2_sub(mode=mode,
                        transform_consistent=transform_consistent,
                        transform_inconsistent=transform_inconsistent,
                        seq_len=seq_len,
                        num_seq=num_seq,
                        downsample=downsample,
                        return_label=return_label,
                        dim=dim,
                        csv_root='/faststorage/SSv2',
                        frame_root='/faststorage/SSv2/frames',
                        random=random,
                        test=test,
                        inter_len=inter_len,
                        fraction=fraction
                        )
    if not ddp:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.distributed.DistributedSampler(dataset, shuffle=True)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=128,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=128,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


class SSV2(data.Dataset):
    def __init__(self,
                mode='train',
                transform_consistent=None,
                transform_inconsistent=None,
                seq_len:int=4,
                num_seq:int=3,
                downsample:int=4,
                which_split=1,
                return_label=False,
                dim=150,
                csv_root='/faststorage/SSv2',
                frame_root='/faststorage/SSv2/frames',
                random=False,
                test=False,
                fraction=1.0,
                inter_len:int=0 # num of frames (after downsampling) between two clips
                ):
        self.mode = mode
        self.transform_consistent = transform_consistent
        self.transform_inconsistent = transform_inconsistent
        self.seq_len=seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.dim = dim
        self.csv_root = csv_root
        self.frame_root = frame_root
        self.random = random
        self.test = test
        self.inter_len = inter_len
        self.total_len = ((self.seq_len + self.inter_len)*self.num_seq - self.inter_len)*self.downsample
        self.fraction = fraction
        
        begin_idxs = np.arange(self.num_seq)*self.downsample*(self.seq_len+self.inter_len) 
        inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
        self.base_seq_idx = (inter_idxs + begin_idxs).T.flatten()

        
        folder_name = 'annotations'

        # splits
        if mode == 'train':
            split = os.path.join(self.csv_root, folder_name, 'train_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        elif mode == 'val': # use val for test
            split = os.path.join(self.csv_root, folder_name, 'val_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test': # use val for test
            split = os.path.join(self.csv_root, folder_name, 'val_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            _, vlen, _ = row

            if vlen-self.total_len <= 0 and not self.random and not self.test:
                drop_idx.append(idx)
            if vlen-self.seq_len*self.downsample <= 0 and (self.random or self.test):
                drop_idx.append(idx)

        self.video_info = video_info.drop(drop_idx, axis=0)
        print("Droped number of videos:", len(drop_idx))

        if self.fraction < 1: 
            self.video_info = self.video_info.sample(frac=self.fraction)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        
        # if vlen-self.total_len <= 0: raise ValueError('video too short')
        if not self.random and not self.test:
            n = 1
            start_idx = np.random.choice(range(vlen-self.total_len), n)
            seq_idx = self.base_seq_idx + start_idx

            # seq_idx = np.arange(self.seq_len*self.num_seq)*self.downsample + start_idx
        elif self.random:
            # each clip is selected randomly without ordering
            n = self.num_seq
            begin_idxs = np.random.choice(range(vlen-self.seq_len*self.downsample), n) 
            inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
            seq_idx = (inter_idxs + begin_idxs).T.flatten()
        else:
            n = self.num_seq
            begin_idxs = np.linspace(0, vlen-self.seq_len*self.downsample, n, dtype=int)
            inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
            seq_idx = (inter_idxs + begin_idxs).T.flatten()


        return [seq_idx, vpath]


    def __getitem__(self, index):
        vpath, vlen, aid = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        
        seq = [pil_loader_resize(os.path.join(str(self.frame_root), str(vpath), '%06d.jpg' % (i+1)), self.dim) for i in idx_block]

        if self.transform_consistent is not None: 
                seq = self.transform_consistent(seq) # apply same transform

        if self.transform_inconsistent is not None: 
            for i in range(self.num_seq):
                sub_seq = self.transform_inconsistent(seq[i*self.seq_len:(i+1)*self.seq_len])
                seq[i*self.seq_len:(i+1)*self.seq_len] = sub_seq
                

        (C, H, W) = seq[0].size()
        seq = torch.stack(seq, 0)
        seq = seq.view(self.num_seq, self.seq_len, C, H, W) # N, T, C, H, W
        seq = seq.permute(0,2,1,3,4) # N, C, T, H, W
        # seq = seq.view(self.num_seq*self.seq_len, C, H, W) # N*T, C, H, W
        # seq = seq.permute(1,0,2,3) # C, N*T, H, W
        
        if self.return_label:
            label = torch.LongTensor([aid])
            return seq, label
        return seq

    def __len__(self):
        return len(self.video_info)
    

class SSV2_sub(data.Dataset):
    def __init__(self,
                mode='train',
                transform_consistent=None,
                transform_inconsistent=None,
                seq_len:int=4,
                num_seq:int=3,
                downsample:int=4,
                which_split=1,
                return_label=False,
                dim=150,
                csv_root='/faststorage/SSv2',
                frame_root='/faststorage/SSv2/frames',
                random=False,
                test=False,
                fraction=1.0,
                inter_len:int=0 # num of frames (after downsampling) between two clips
                ):
        self.mode = mode
        self.transform_consistent = transform_consistent
        self.transform_inconsistent = transform_inconsistent
        self.seq_len=seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.dim = dim
        self.csv_root = csv_root
        self.frame_root = frame_root
        self.random = random
        self.test = test
        self.inter_len = inter_len
        self.total_len = ((self.seq_len + self.inter_len)*self.num_seq - self.inter_len)*self.downsample
        self.fraction = fraction
        
        begin_idxs = np.arange(self.num_seq)*self.downsample*(self.seq_len+self.inter_len) 
        inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
        self.base_seq_idx = (inter_idxs + begin_idxs).T.flatten()

        
        folder_name = 'annotations_sub'

        # splits
        if mode == 'train':
            split = os.path.join(self.csv_root, folder_name, 'train_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        elif mode == 'val': # use val for test
            split = os.path.join(self.csv_root, folder_name, 'val_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test': # use val for test
            split = os.path.join(self.csv_root, folder_name, 'val_videofolder.csv')
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            _, vlen, _ = row

            if vlen-self.total_len <= 0 and not self.random and not self.test:
                drop_idx.append(idx)
            if vlen-self.seq_len*self.downsample <= 0 and (self.random or self.test):
                drop_idx.append(idx)

        self.video_info = video_info.drop(drop_idx, axis=0)
        print("Droped number of videos:", len(drop_idx))

        if self.fraction < 1: 
            self.video_info = self.video_info.sample(frac=self.fraction)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        
        # if vlen-self.total_len <= 0: raise ValueError('video too short')
        if not self.random and not self.test:
            n = 1
            start_idx = np.random.choice(range(vlen-self.total_len), n)
            seq_idx = self.base_seq_idx + start_idx

            # seq_idx = np.arange(self.seq_len*self.num_seq)*self.downsample + start_idx
        elif self.random:
            # each clip is selected randomly without ordering
            n = self.num_seq
            begin_idxs = np.random.choice(range(vlen-self.seq_len*self.downsample), n) 
            inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
            seq_idx = (inter_idxs + begin_idxs).T.flatten()
        else:
            n = self.num_seq
            begin_idxs = np.linspace(0, vlen-self.seq_len*self.downsample, n, dtype=int)
            inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
            seq_idx = (inter_idxs + begin_idxs).T.flatten()


        return [seq_idx, vpath]


    def __getitem__(self, index):
        vpath, vlen, aid = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        
        seq = [pil_loader_resize(os.path.join(str(self.frame_root), str(vpath), '%06d.jpg' % (i+1)), self.dim) for i in idx_block]

        if self.transform_consistent is not None: 
                seq = self.transform_consistent(seq) # apply same transform

        if self.transform_inconsistent is not None: 
            for i in range(self.num_seq):
                sub_seq = self.transform_inconsistent(seq[i*self.seq_len:(i+1)*self.seq_len])
                seq[i*self.seq_len:(i+1)*self.seq_len] = sub_seq
                

        (C, H, W) = seq[0].size()
        seq = torch.stack(seq, 0)
        seq = seq.view(self.num_seq, self.seq_len, C, H, W) # N, T, C, H, W
        seq = seq.permute(0,2,1,3,4) # N, C, T, H, W
        # seq = seq.view(self.num_seq*self.seq_len, C, H, W) # N*T, C, H, W
        # seq = seq.permute(1,0,2,3) # C, N*T, H, W
        
        if self.return_label:
            label = torch.LongTensor([aid])
            return seq, label
        return seq

    def __len__(self):
        return len(self.video_info)