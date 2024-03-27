#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s_f1_alternate.py --sym_loss --infonce --epochs 400 --seed 42
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_summation --epoch_num 400
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_summation --epoch_num 400 --which_encoder 2
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_summation --epoch_num 400
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operation_summation --epoch_num 400 --concat
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s_f1_alternate.py --sym_loss --infonce --epochs 400 --seed 3407
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407_operation_summation --epoch_num 400
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407_operation_summation --epoch_num 400 --which_encoder 2
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407_operation_summation --epoch_num 400
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_E1Alternative_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed3407_operation_summation --epoch_num 400 --concat