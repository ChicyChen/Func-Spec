#!/bin/bash
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 1 --gpu '6'
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 2 --gpu '7'
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --gpu '6'
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --concat --gpu '7'

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.25 --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.25 --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.25 --epoch_num 400 --gpu '6' 
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.25 --epoch_num 400 --concat --gpu '7'

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.75 --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.75 --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.75 --epoch_num 400 --gpu '6' 
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.25_prob_average0.75 --epoch_num 400 --concat --gpu '7' 

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.25 --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.25 --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.25 --epoch_num 400 --gpu '6' 
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.25 --epoch_num 400 --concat --gpu '7' 

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.75 --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.75 --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.75 --epoch_num 400 --gpu '6'
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj_rand_derivative_rand_average/ucf_rd1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_Summation_prob_derivative0.75_prob_average0.75 --epoch_num 400 --concat --gpu '7'

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --gpu '6' --img_num 1
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --concat --gpu '7' --img_num 1

python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 1 --gpu '6'
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --which_encoder 2 --gpu '7'
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --gpu '6' 
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2streams1proj/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed233_operation_summation --epoch_num 400 --concat --gpu '7'