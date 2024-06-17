addpath('/ll3/home/choi574/git_libs/matlab/encoding/function_module');
addpath('/export/home/choi574/git_libs/matlab/encoding/function_module');


%% Process HW's features using efficient PCA
% I have already this data processed by old code, but I want to verify if
% my new code is working well. 
layername = {'/conv1';'/conv2';'/conv3';'/conv4';'/conv5';'/salmap';'/fc8'};

dataroot_read = '.../extract_features/';
dataroot_write = '.../process_features_newCode/';
fn_prefix_train = 'model_feature_maps_seg';
fn_prefix_test = 'model_feature_maps_test';
method_PCA = 3;
n_seg_test = 5;

feature_processing1_PCA(layername, dataroot_read, dataroot_write, method_PCA, fn_prefix_train);
feature_processing2_temporal_downsampling_train(layername, dataroot_read, dataroot_write, fn_prefix_train);
feature_processing2_temporal_downsampling_test(layername, dataroot_read, dataroot_write, fn_prefix_test, n_seg_test);


%% Feature processing for stream-wise encoding
dataroot_read = '.../process_features_newCode/';
dataroot_write = '.../process_features_newCode/';

method_PCA = 1;
post_fix = 'singleStream';
n_seg_test = 5;

layerAll_processing_PCA_train(layername, dataroot_read, dataroot_write, method_PCA, post_fix);
layerAll_processing_PCA_test(layername, dataroot_read, dataroot_write, method_PCA, post_fix, n_seg_test);



 

%% Encoding for stream-wise encoding
addpath('/ll3/home/choi574/git_libs/matlab/encoding/function_module');
addpath('/export/home/choi574/git_libs/matlab/encoding/function_module');

% Process HW's features using efficient PCA
dataroot_read = '.../process_features_newCode/';
dataroot_write = '.../encoding_layerAll/';
post_fix = 'singleStream';

%fmripath = '/ll3/home/choi574/research_mk/fMRI/modeling_brain/source_codes/source_data_copied_from_HW/video_fmri_dataset/subject1/fmri/';
fmripath = '/ll3/home/choi574/research_mk/fMRI/modeling_brain/KHan_data/fmri_data_processed_mk/subject1/'; % This file is processed by ME

lam1 = sqrt(1e-8);
lam2 = sqrt(1e-3);
nfold = 10;

layerAll_encoding_train(dataroot_read, dataroot_write, fmripath, lam1, lam2, nfold, post_fix);
Rmat = layerAll_encoding_test(dataroot_read, dataroot_write, fmripath, post_fix);
