#!/bin/bash

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_base.py --sym_loss --infonce --epochs 400 --feature_size 363 --projection 1452 --proj_hidden 1452 --width_deduction_ratio 1.41
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/simclr_base/ucf1.0_nce_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 0 --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/simclr_base/ucf1.0_nce_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 0 --gpu '7' --img_num 1 --width_deduction_ratio 1.41



torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2p.py --sym_loss --infonce --epochs 400 --seed 233 --concat --width_deduction_ratio 1.41 --feature_size 363 --projection 1452 --proj_hidden 1452

python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1 --width_deduction_ratio 1.41

python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_concatenation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --img_num 1 --width_deduction_ratio 1.41



torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2p.py --sym_loss --infonce --epochs 400 --seed 233 --width_deduction_ratio 1.41 --feature_size 363 --projection 1452 --proj_hidden 1452
                                                   
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1 --width_deduction_ratio 1.41

python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_base/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --img_num 1 --width_deduction_ratio 1.41



torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/train_net3d_2s2pViDiDi.py --sym_loss --infonce --epochs 400 --seed 233 --width_deduction_ratio 1.41 --feature_size 363 --projection 1452 --proj_hidden 1452
    
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 1 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --which_encoder 2 --gpu '7' --img_num 1 --width_deduction_ratio 1.41

python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --width_deduction_ratio 1.41
python evaluation/eval_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --gpu '6' --img_num 1 --width_deduction_ratio 1.41
python evaluation/image_retrieval_2encoders.py --ckpt_folder /data/checkpoints_yehengz/2s2p_ViDiDi/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_width_deduc_ratio1.41_stem_deductFalse --epoch_num 400 --concat --gpu '7' --img_num 1 --width_deduction_ratio 1.41