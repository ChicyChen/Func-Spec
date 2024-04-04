#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2prdramix.py --sym_loss --infonce --epochs 400 --seed 233 --width_deduction_ratio 1.41 --feature_size 363 --projection 1452 --proj_hidden 1452
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2prdramix.py --sym_loss --infonce --epochs 400 --seed 233 
