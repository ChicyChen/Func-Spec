#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 233
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 233 --concat
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 42 --concat    
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 3407 --concat


