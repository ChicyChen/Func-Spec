#!/bin/bash
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.9 --epoch_num 400 --gpu '7'
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.9 --epoch_num 400 --which_encoder 2 --gpu '7'
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.9 --epoch_num 400 --gpu '7'
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams_rand_derivative/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation_prob0.9 --epoch_num 400 --concat --gpu '7'


