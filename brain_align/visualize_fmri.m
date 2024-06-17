addpath('/export/code_libs/prj_mk/matlab/');
% region = randn(59412,1);
load("deduct_d75a25/adjust_encoder2_input2/corr_pcaFalse.mat")
plot_surface(test, 'flat', true) % Plot flat version
% plot_surface(test, 'inflated', true) % Plot flat version