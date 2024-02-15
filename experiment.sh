#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 233 --prob 0.2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.2 --epoch_num 400
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.2 --epoch_num 400 --which_encoder 2
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.2 --epoch_num 400
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.2 --epoch_num 400 --concat
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2srd.py --sym_loss --infonce --epochs 400 --seed 233 --prob 0.7
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.7 --epoch_num 400
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.7 --epoch_num 400 --which_encoder 2
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.7 --epoch_num 400
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.7 --epoch_num 400 --concat


