#!/bin/bash
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 40
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 40 --which_encoder 2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 50
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 50 --which_encoder 2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 60
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 60 --which_encoder 2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 70
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 70 --which_encoder 2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 80
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 80 --which_encoder 2
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 90
python evaluation/eval_retrieval.py --ckpt_folder /data/checkpoints_yehengz/2streams/first90/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_seed42_operationsummation --epoch_num 90 --which_encoder 2


