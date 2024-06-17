#!/bin/bash

python net3d_alllayers_extraction.py --srate 0.5
python net3d_alllayers_process.py --srate 0.5
python net3d_alllayers_together.py
python ridge_regression.py
python net3d_alllayers_together.py --pca
python ridge_regression.py --pca

# python ridge_regression.py --root base/adjust_encoder0_input0 --pca
# python ridge_regression.py --root base/adjust_encoder0_input0

# python net3d_alllayers_process.py --root deduct_d75a25/adjust_encoder1_input0
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder1_input0 --pca
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder1_input0
# python ridge_regression.py --root deduct_d75a25/adjust_encoder1_input0 --pca
# python ridge_regression.py --root deduct_d75a25/adjust_encoder1_input0

# python net3d_alllayers_process.py --root deduct_d75a25/adjust_encoder1_input1
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder1_input1 --pca
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder1_input1
# python ridge_regression.py --root deduct_d75a25/adjust_encoder1_input1 --pca
# python ridge_regression.py --root deduct_d75a25/adjust_encoder1_input1

# python net3d_alllayers_process.py --root deduct_d75a25/adjust_encoder2_input0
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder2_input0 --pca
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder2_input0
# python ridge_regression.py --root deduct_d75a25/adjust_encoder2_input0 --pca
# python ridge_regression.py --root deduct_d75a25/adjust_encoder2_input0

# python net3d_alllayers_process.py --root deduct_d75a25/adjust_encoder2_input2
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder2_input2 --pca
# python net3d_alllayers_together.py --root deduct_d75a25/adjust_encoder2_input2
# python ridge_regression.py --root deduct_d75a25/adjust_encoder2_input2 --pca
# python ridge_regression.py --root deduct_d75a25/adjust_encoder2_input2


# python net3d_alllayers_extraction.py --resnet_folder /data/checkpoints_yehengz/2s2p_batchwise_rdra/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_prob_derivative0.75_prob_average0.25_width_deduc_ratio1.41_stem_deductFalse \
#                                      --resnet_name resnet1_epoch400.pth.tar \
#                                      --encoder_num 1                        \
#                                      --outfolder_root deduct_d75a25         \
#                                      --encoder_num 1                        \
#                                      --input_type 0                         \
#                                      --width_deduction_ratio 1.41           

# python net3d_alllayers_extraction.py --resnet_folder /data/checkpoints_yehengz/2s2p_batchwise_rdra/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_prob_derivative0.75_prob_average0.25_width_deduc_ratio1.41_stem_deductFalse \
#                                      --resnet_name resnet1_epoch400.pth.tar \
#                                      --encoder_num 1                        \
#                                      --outfolder_root deduct_d75a25         \
#                                      --encoder_num 1                        \
#                                      --input_type 1                         \
#                                      --width_deduction_ratio 1.41

# python net3d_alllayers_extraction.py --resnet_folder /data/checkpoints_yehengz/2s2p_batchwise_rdra/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_prob_derivative0.75_prob_average0.25_width_deduc_ratio1.41_stem_deductFalse \
#                                      --resnet_name resnet2_epoch400.pth.tar \
#                                      --encoder_num 2                        \
#                                      --outfolder_root deduct_d75a25         \
#                                      --encoder_num 2                        \
#                                      --input_type 0                         \
#                                      --width_deduction_ratio 1.41

# python net3d_alllayers_extraction.py --resnet_folder /data/checkpoints_yehengz/2s2p_batchwise_rdra/ucf1.0_nce2s_r3d18/symTrue_bs64_lr4.8_wd1e-06_ds3_sl8_nw_randFalse_feature_size363_projection1452_proj_hidden1452_epochs400_seed233_operation_summation_prob_derivative0.75_prob_average0.25_width_deduc_ratio1.41_stem_deductFalse \
#                                      --resnet_name resnet2_epoch400.pth.tar \
#                                      --encoder_num 2                        \
#                                      --outfolder_root deduct_d75a25         \
#                                      --encoder_num 2                        \
#                                      --input_type 2                         \
#                                      --width_deduction_ratio 1.41