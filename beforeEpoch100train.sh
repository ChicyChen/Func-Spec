#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --infonce --epochs 400 --seed 233
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --infonce --epochs 400 --seed 42
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s.py --sym_loss --infonce --epochs 400 --seed 3407

# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30 --which_encoder 2

# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30 --which_encoder 2

# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 10 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 20 --which_encoder 2
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30
# python evaluation/eval_retrieval.py --ckpt_folder  --epoch_num 30 --which_encoder 2