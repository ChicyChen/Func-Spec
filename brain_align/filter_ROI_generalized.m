function [x_masked, x_score, mask] = filter_ROI_generalized(x, roi, isLibi3)
    % input roi directly represents the index of brain area. 
    % This function is different from 'filter_ROI.m' in that the input roi is not an simple predefined index anymore. 
    % if roi is 1, then the output will be corresponding to v1 voxels
    % if roi is 4, then the output will be corresponding to v2 voxels
    % see the variables rois_name, rois defined below. 
    %
    % Args:
    %   x: (1 x 59412) array
    %   roi: int indicating roi. 
    %        See https://docs.google.com/spreadsheets/d/1BMz62Rt_vdi1y-q43FSC5Avokd1cqsF158cRJ7DFQH8/edit#gid=1542825953

    % Visualize correlation

    %% Define data directory
    
    
    %addpath('/mnt/lls/home/choi574/git_libs/matlab/utils'); 
    %addpath('/mnt/lls/home/choi574/matlab/fieldtrip-20210128/external/afni/');    
    %addpath('/mnt/lls/code/matlab/0libi/hgwen/Matlab/matlab_HG/');

    %% Data region ROI
    if isLibi3
        addpath('/export/home/choi574/matlab/fieldtrip-20210128/fileio') % ft_read_cifti.m
        addpath('/export/home/choi574/research_mk/fMRI/libs/gifti/')
        addpath('/export/code/matlab/0libi/hgwen/Matlab/toolbox/fieldtrip/fileio'); % /ft_read_cifti.m
        addpath('/export/code/matlab/0libi/zmliu/eegfmritool/'); %amri_sig_detrend.m
        addpath('/export/code/matlab/0libi/hgwen/Matlab/matlab_HG');
        addpath('/export/code/matlab/0libi/khlu/cifti');
        addpath('/export/code/matlab/0libi/hgwen/Matlab/matlab_HG/');
        addpath('/export/home/choi574/matlab/fieldtrip-20210128/external/afni/');
        addpath('/export/home/choi574/git_libs/matlab/utils'); 
        fn = '/export/data/3/human/fMRI-VAE/rsfMRI-VAE/demo/v2/template/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
    else
        addpath('/ll3/home/choi574/matlab/fieldtrip-20210128/external/gifti/') % ft_read_cifti.m
        addpath('/ll3/home/choi574/matlab/fieldtrip-20210128/utilities/')
        addpath('/ll3/home/choi574/matlab/fieldtrip-20210128/fileio') % ft_read_cifti.m
        %addpath('/ll3/home/choi574/research_mk/fMRI/libs/gifti/@gifti')
        addpath('/ll3/code/matlab/0libi/hgwen/Matlab/toolbox/fieldtrip/fileio'); % /ft_read_cifti.m
        addpath('/ll3/code/matlab/0libi/zmliu/eegfmritool/'); %amri_sig_detrend.m
        addpath('/ll3/code/matlab/0libi/hgwen/Matlab/matlab_HG');
        addpath('/ll3/code/matlab/0libi/khlu/cifti');
        addpath('/ll3/home/choi574/research_mk/fMRI/modeling_brain/twilight/code_mk');
        addpath('/ll3/code/matlab/0libi/hgwen/Matlab/matlab_HG/');
        addpath('/ll3/home/choi574/matlab/fieldtrip-20210128/external/afni/');
        addpath('/ll3/home/choi574/git_libs/matlab/utils'); 
        fn = '/ll3/data/3/human/fMRI-VAE/rsfMRI-VAE/demo/v2/template/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
    end
    cf = ft_read_cifti(fn);
    voxel_idx = cf.x1;
    voxel_isnan = ~isnan(cf.x1);
    voxel_idx_nnan = zeros(59412,1);
    ii = 1;
    for i=1:size(voxel_idx)
        if voxel_isnan(i) == 1
            voxel_idx_nnan(ii,1) = voxel_idx(i,1);
            ii = ii+1;
        end
    end
    % voxel_idx_nnan: 59412 array including label index

    %rois_name = ["v1", "v2", "v3", "v4", "LO", "MT", "FFC", "PHA1", "PHA3", "TPOJ1", "PH"];
    %rois = [1, 4, 5, 6, 21, 23, 18, 126, 127, 139];
    %which copyfields

    R=x;
    % R: (1 x 59412) array 
    %R_nnan = isnan_mask .* R;
    % Replace NaN values with 0
    isnan_mask = isnan(R);
    R_nnan = R;
    R_nnan(isnan_mask) = 0;

    mask = (voxel_idx_nnan==roi)|(voxel_idx_nnan==roi+180);
    x_masked = mask .* R_nnan';

    x_score = sum(x_masked, 'all') / sum(mask, 'all');

end