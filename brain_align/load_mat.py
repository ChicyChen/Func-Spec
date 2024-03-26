import scipy.io as sio
import mat73
import numpy as np

# fmripath = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/training_fmri.mat'
fmripath = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/testing_fmri.mat'
# fmri = sio.loadmat(fmripath)
fmri = mat73.loadmat(fmripath)

# fmripath = '/data/3/human/Human_Visual_Experiments/fMRI_surface/fmri_surface_JY_train_section1-18_brain.mat'
# fmri = mat73.loadmat(fmripath)

test1 = np.mean(fmri['fmritest']['test1'], axis = 2)
print(test1.shape)

test = fmri['fmritest']
print(test.keys())

# print(fmri['hrf'])
# print(fmri['data1'].shape)
# print(fmri['data2'].shape)
# print(fmri['fmri'][0][0][0].shape) # (59412, 240, 18)
# print(fmri['fmri'][0][0][1].shape) # (59412, 240, 18)

# data1 = fmri['fmri'][0][0][0]
# data2 = fmri['fmri'][0][0][1]
# data_avg = (data1 + data2) / 2
# session1 = data_avg[:,:,0].T # (240, 59412): T, d
# print(session1.shape)


