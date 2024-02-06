#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/mytune_net3d_lr.py --sym_loss --which_experiment 1
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/mytune_net3d_coeffs.py --sym_loss --which_experiment 1
