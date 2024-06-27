#!/bin/bash

# base case

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_swin_ViDiDi.py --epochs 400 --batch_size 256 --projection 3072 --proj_hidden 3072 --sym_loss --base_lr 1e-3 --wd 5e-4 --warm_up --warm_up_epochs 20 --end_lr_fraction 0.01
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/swin_ViCReg_ViDiDi/ucf1.0_pcn_swin3dtiny/symTrue_bs256_lr0.001_end_lr_frac0.01_wd0.0005_ns2_ds3_sl8_il0_nw_randFalse_warmupTrue_warmup_epochs20_projection_size3072_proj_hidden3072_freeze_pat_embdFalse_mse_l1.0_std_l1.0_cov_l0.04_epoch_num400 --epoch_num 400 --swin --which_encoder 0


torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_sdtd.py --epochs 100 --batch_size 64 --sym_loss


