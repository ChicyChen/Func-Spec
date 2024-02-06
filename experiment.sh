#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --infonce --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --infonce --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --infonce --epochs 400 --which_fixed_pair 1 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --epochs 400 --which_fixed_pair 1 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --infonce --epochs 400 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --epochs 400 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --infonce --epochs 400 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --infonce --epochs 400 --which_fixed_pair 1 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --epochs 400 --seed 3407
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_nws.py --sym_loss --epochs 400 --which_fixed_pair 1 --seed 3407


