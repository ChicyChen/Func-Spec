from nilearn import plotting as nlp 
import nibabel as nb
import numpy as np 
import random
import string
import torch
import math
import time 
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.interpolate import griddata
from scipy.io import loadmat

### cifti data reading and visualization

def _data_from_cifti(data, axis, surf_name):
    # modified from: nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  
        if name == surf_name:                                
            data = data.T[data_indices]             
            # model.vertex is the index-mapping from the gifti 64K space to the 59k space
            # data_indices is the index-mapping from the 91K (no medial wall) space to the 59k space
            return data, model.vertex
    raise ValueError(f"No structure named {surf_name}")

def amri_cifti_open(cii_path, rois):
    # inputs
    #   cii_in_path: input cifti file 
    #   roi: a list of roi names (e.g. ['CIFTI_STRUCTURE_CORTEX_LEFT']); See appendix

    # outputs
    #   results: data from each given roi in a list 
    #   indices: indice of each roi is a list, only to match gifti for display

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    cifti = nb.load(cii_path)
    cifti_data = cifti.get_fdata(dtype=np.float32)
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    # get data (surface data in 59412 space)
    results, indices = list(zip(*[_data_from_cifti(cifti_data, axes[1], roi) for roi in rois]))

    return results, indices

def _indices_from_cifti(axis, surf_name):
    # internal function of cifti_surf_modify()

    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  
        if name == surf_name:                                
            vtx_indices = model.vertex    # vtx_indices is the index-mapping from the 64K space to the 59k space                   
            return data_indices, vtx_indices # data_indices is the index_mapping from all brain_structures (91k) to the 59k space
    raise ValueError(f"No structure named {surf_name}")

    return

def amri_cifti_modify(cii_in_path, cii_out_path, data_list, rois, reset=False):
    # inputs
    #   cii_in_path:  input cifti file (to read template information)
    #   cii_out_path: output cifti file
    #   data_list:    list of data to fill 
    #   rois:         list of roi ro fill
    #   reset:        reset the cifti header (e.g. to change shape) 

    # outputs
    #   no outputs but save the cii_out_path

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    # load a cifti data, modify its surface fields, and save 
    cifti = nb.load(cii_in_path)
    cifti_data = cifti.get_fdata(dtype=np.float32)
    cifti_hdr = cifti.header
    axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
    # get indices
    data_indices, _ = list(zip(*[_indices_from_cifti(axes[1], roi) for roi in rois]))

    # check if reset the matrix 
    # reference: https://neurostars.org/t/alter-size-of-matrix-for-new-cifti-header-nibabel/20903
    if reset:
        TR = cifti.nifti_header.get_zooms()[3] # https://www.programcreek.com/python/?code=edickie%2Fciftify%2Fciftify-master%2Fciftify%2Fbin%2Fciftify_clean_img.py
        cifti_data = np.zeros((data_list[0].shape[1], cifti_data.shape[1]), dtype=np.float32) # TxN
        # Create new axes 0 to match new mat size and store orig axes 1
        ax_0 = nb.cifti2.SeriesAxis(start=0, step=TR, size=data_list[0].shape[1]) 
        ax_1 = cifti_hdr.get_axis(1)
        # Create new header and cifti object
        new_cifti_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))

    # assign values back to the cifti_data 
    for data, data_indice in zip(data_list, data_indices): 
        cifti_data.T[data_indice] = data
    # save the data
    new_img = nb.Cifti2Image(cifti_data, header=(new_cifti_header if reset else cifti.header), nifti_header=cifti.nifti_header)
    new_img.to_filename(cii_out_path)
    
    return

def amri_cifti_wbview(data_list, rois, surf='inflated', res=32, gii_dir='/data/3/human/HCP/HCP_fMRI/HCP_additional_file/', cifti_template_path='/data/3/human/HCP/HCP_fMRI/HCP_additional_file/template.dtseries.nii'):
    # inputs
    #   data_list:             list of roi-data to display; each roi-data has size: N x T
    #   rois:                  list of rois to display
    #   surf:                  'inflated' or 'flat'
    #   res:                   resolution = 32 or 59, usually 32 
    #   gii_dir:               the directory of gifti files for display
    #   cifti_template_path:   use the header and meta info from a cifti template

    # outputs
    #   display the cortical data with wb_view

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    # the python version similar to our display_surf in matlab, but is able to display the whole grayordinates
 
    gii_path_left = os.path.join(gii_dir, 'Q1-Q6_R440.L.'+surf+'.'+str(res)+'k_fs_LR.surf.gii')
    gii_path_right = os.path.join(gii_dir, 'Q1-Q6_R440.R.'+surf+'.'+str(res)+'k_fs_LR.surf.gii')
    if len(data_list[0].shape) == 1:
        data_list = [np.expand_dims(data, axis=1) for data in data_list]
    tmpdir =''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(10))
    print('Temporarily create ' + tmpdir) 
    os.system('mkdir '+tmpdir)
    command = 'wb_view '
    cii_out_path = os.path.join(tmpdir, 'display_grayordinates_temp.dtseries.nii')
    amri_cifti_modify(cifti_template_path, cii_out_path, data_list, rois, reset=True)
    command = command + cii_out_path + ' '
    command = command + gii_path_left + ' ' + gii_path_right + ' > /dev/null & '
    os.system(command)
    time.sleep(3) # delay for longer time if wb_view says "file does not exist"
    os.system('rm -r ' + tmpdir)

    return gii_path_left, gii_path_right # return the gifti files that are used by this function 

def amri_cifti_matchgifti(roi_data, roi_vtx_indice):
    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    #
 
    # match the vertex index of cifti-cortical data to the gifti data

    if len(roi_data.shape) == 1:
        roi_data = np.expand_dims(roi_data, axis=-1)
    roi_data_out = np.zeros((roi_vtx_indice.max()+1, ) + roi_data.shape[1:], dtype=roi_data.dtype)
    roi_data_out[roi_vtx_indice] = roi_data
         
    return roi_data_out
   
def _cart2sph(xyz):
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    elevation = np.arctan2(xyz[:,2], np.sqrt(xyz[:,0]**2 + xyz[:,1]**2))
    r = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2 + xyz[:,2]**2)

    return azimuth, elevation, r 

def _align_range(x, eps=1e-5):    
    length = max(max(x)-min(x), eps)
    return (2*x / length)

def _apply_grid_mapping(transformed_sin_elevation, transformed_azimuth, grid_size):
    # generate the grid points mapping and the inverse mapping
    grid_mapping = csr_matrix((grid_size**2, transformed_azimuth.shape[0]))
    inverse_mapping = csr_matrix((transformed_azimuth.shape[0], grid_size**2))
    # generate the transformed sphere coordinates and the grid coordinates 
    sphere_coords = np.concatenate((transformed_sin_elevation.reshape((-1, 1)), transformed_azimuth.reshape((-1, 1))), axis=1)
    grid_elevation, grid_azimuth = np.meshgrid(np.arange(-1, 1, 2.0/grid_size), np.arange(-1, 1, 2.0/grid_size))
    grid_coords = np.concatenate((grid_elevation.reshape(-1,1), grid_azimuth.reshape(-1,1)), axis=1)
    # generate the grid mapping and the inverse mapping: points, value, interpolate-to-which
    grid_interpolate_value = griddata(sphere_coords, np.arange(transformed_azimuth.shape[0]), grid_coords, method='nearest') # (grid_size**2, 1)
    inverse_interpolate_value = griddata(grid_coords, np.arange(grid_size**2), sphere_coords, method='nearest') # (69K, 1)
    # grid transformation and inverse transformation matrix
    for grid_idx in range(grid_size**2):
        grid_mapping[grid_idx, grid_interpolate_value[grid_idx]] = 1
    for sphere_idx in range(transformed_azimuth.shape[0]):
        inverse_mapping[sphere_idx, inverse_interpolate_value[sphere_idx]] = 1 
    
    return grid_mapping, inverse_mapping 

def amri_cifti_georeformat(sphere_coord, surf_indice, grid_size):
    # inputs
    #   sphere_coord:  x-y-z coordinates in the sphere space
    #   surf_indice:   indices of non-medial wall voxels
    #   grid_size:     size of grid

    # outputs
    #   transformed_sin_elevation: sin(elevation) aligned to the range of [-1, 1]
    #   transformed_azimuth:       azimuth aligned to the range of [-1, 1]
    #   grid_mask:                 the mask of non-medial-wall voxels
    #   grid_transformation:       transformation matrix from 59412 space to 2D grid_size^2 space 
    #   reverse_transformation:    reverse trasformation matrix from 2D grid_size^2 to 59412

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version based on the matlab code
    # 
   
    # to mirror images from hemispheres, flip the x-direction of R hemisphere
    azimuth, elevation, _ = _cart2sph(sphere_coord)
    sin_elevation = np.sin(elevation)
    # align range
    transformed_sin_elevation, transformed_azimuth = _align_range(sin_elevation), _align_range(azimuth)
    # generate the non-medial-wall mask first 
    grid_mapping_for_mask, _ = _apply_grid_mapping(transformed_sin_elevation, transformed_azimuth, grid_size) 
    grid_mask = grid_mapping_for_mask.dot(amri_cifti_matchgifti(np.ones_like(surf_indice), surf_indice))
    # the grid transformation matrix and the inverse transformation matrix
    # be consistent with the matlab version line-83,88 in geometric_reformatting.m
    grid_mapping, inverse_mapping = _apply_grid_mapping(transformed_sin_elevation[surf_indice], transformed_azimuth[surf_indice], grid_size) 
    
    return transformed_sin_elevation, transformed_azimuth, grid_mask, grid_mapping, inverse_mapping

# decode bert latent code to grid cortex data
def bert_2_grid(BERT_latent_code, model, VAEmodel):
    dec_output = model.dec_blocks(BERT_latent_code.unsqueeze(0))
    recon_L, recon_R = VAEmodel._decode(dec_output.squeeze(0))
    return recon_L.detach().cpu().numpy(), recon_R.detach().cpu().numpy()

# decode vae latent code to grid cortex data
def vae_2_grid(vae_latent_code, VAEmodel):
    recon_L, recon_R = VAEmodel._decode(vae_latent_code)
    return recon_L.detach().cpu().numpy(), recon_R.detach().cpu().numpy()

# convert grid cortex data to surf and visualize
def grid_2_surf(recon_L, recon_R):
    # load left and right inverse array
    gii_dir = '/data/3/human/HCP/HCP_fMRI/HCP_additional_file/'
    left_inverse_path = '/data/3/human/fMRI-VAE/Dataset_Preparation/Final_Transformation_Data/Final_Set/Left_fMRI2Grid_192_by_192_NN.mat'
    right_inverse_path = '/data/3/human/fMRI-VAE/Dataset_Preparation/Final_Transformation_Data/Final_Set/Right_fMRI2Grid_192_by_192_NN.mat'
    left_mat = loadmat(left_inverse_path)
    right_mat = loadmat(right_inverse_path)
    left_inverse_transformation = left_mat['inverse_transformation'].toarray()
    right_inverse_transformation = right_mat['inverse_transformation'].toarray()

    # convert to surf data
    num_frames = recon_L.shape[0]
    recon_L = recon_L.reshape(num_frames,-1).T
    recon_R = recon_R.reshape(num_frames,-1).T
    corticalrecon_L = left_inverse_transformation @ recon_L.astype('double')
    corticalrecon_R = right_inverse_transformation @ recon_R.astype('double')
    surf_tensors = [corticalrecon_L, corticalrecon_R]

    # visualize the surface
    rois = ['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']
    gii_left_path_used, gii_right_path_used = amri_cifti_wbview([surf_tensor for surf_tensor in surf_tensors], rois, surf='inflated', res=32, gii_dir=gii_dir)
    return

# visualize bert latent code to brain surface
def bert_2_vis(BERT_latent_code, model, VAEmodel):
    recon_L, recon_R = bert_2_grid(BERT_latent_code, model, VAEmodel)
    grid_2_surf(recon_L, recon_R)
    return

# visualize vae latent code to brain surface
def vae_2_vis(vae_latent_code, VAEmodel):
    recon_L, recon_R = vae_2_grid(vae_latent_code, VAEmodel)
    grid_2_surf(recon_L, recon_R)
    return
    
# Appendix: indices of each cifti ROI (load by nibabel)
# CIFTI_STRUCTURE_CORTEX_LEFT   slice(0, 29696, None)
# CIFTI_STRUCTURE_CORTEX_RIGHT   slice(29696, 59412, None)
# CIFTI_STRUCTURE_ACCUMBENS_LEFT   slice(59412, 59547, None)
# CIFTI_STRUCTURE_ACCUMBENS_RIGHT  slice(59547, 59687, None)
# CIFTI_STRUCTURE_AMYGDALA_LEFT   slice(59687, 60002, None)
# CIFTI_STRUCTURE_AMYGDALA_RIGHT   slice(60002, 60334, None)
# CIFTI_STRUCTURE_BRAIN_STEM   slice(60334, 63806, None)
# CIFTI_STRUCTURE_CAUDATE_LEFT   slice(63806, 64534, None)
# CIFTI_STRUCTURE_CAUDATE_RIGHT   slice(64534, 65289, None)
# CIFTI_STRUCTURE_CEREBELLUM_LEFT   slice(65289, 73998, None)
# CIFTI_STRUCTURE_CEREBELLUM_RIGHT   slice(73998, 83142, None)
# CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT   slice(83142, 83848, None)
# CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT   slice(83848, 84560, None)
# CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT   slice(84560, 85324, None)
# CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT   slice(85324, 86119, None)
# CIFTI_STRUCTURE_PALLIDUM_LEFT   slice(86119, 86416, None)
# CIFTI_STRUCTURE_PALLIDUM_RIGHT   slice(86416, 86676, None)
# CIFTI_STRUCTURE_PUTAMEN_LEFT   slice(86676, 87736, None)
# CIFTI_STRUCTURE_PUTAMEN_RIGHT   slice(87736, 88746, None)
# CIFTI_STRUCTURE_THALAMUS_LEFT   slice(88746, 90034, None)
# CIFTI_STRUCTURE_THALAMUS_RIGHT   slice(90034, None, None)

