from sklearn.linear_model import Ridge
import numpy as np
import scipy.io as sio
import torch
import mat73
import pickle
import copy
import os

fmripath = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/training_fmri.mat'
fmripath_test = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/testing_fmri.mat'
fmri = sio.loadmat(fmripath)
fmri_test = mat73.loadmat(fmripath_test)
with open("ventral", "rb") as fp:
    ventral_idx = pickle.load(fp)
with open("dorsal", "rb") as fp2:
    dorsal_idx = pickle.load(fp2)
with open("lateral", "rb") as fp3:
    lateral_idx = pickle.load(fp3)

data1 = fmri['fmri'][0][0][0]
data2 = fmri['fmri'][0][0][1]
data_avg = (data1 + data2) / 2
session1 = data_avg[:,:,0].T 
session1_sub = session1[:, ventral_idx] # (240, subnum): T, D

# features = torch.load("features_ds.pt") # (240, 367): T, d
# features_test = torch.load("features_ds_test.pt") # (240, 367): T, d
# load processed model features
folder = "encoder1"
features_path = os.path.join(folder, "ds_features_all.pt")
features_test_path = os.path.join(folder, "ds_features_test_all.pt")
features = np.load(features_path) # (240*18, 367): T, d
features_test = np.load(features_test_path) # (240*5, 367): T, d

clf = Ridge(alpha=10.0)
clf.fit(features, session1_sub)
score = clf.score(features, session1_sub)
print(score)

test1 = np.mean(fmri_test['fmritest']['test1'], axis = 2)
test1 = test1.T
test1_sub = test1[:, ventral_idx]
test1_sub_shuffle_time = copy.deepcopy(test1_sub)
np.random.shuffle(test1_sub_shuffle_time)
predict_test = clf.predict(features_test)  # (240, subnum)
predict_train = clf.predict(features) # (240, subnum)


ventral_train_arr = np.zeros(len(ventral_idx))
ventral_test_arr = np.zeros(len(ventral_idx))
ventral_testrand_arr = np.zeros(len(ventral_idx))

dorsal_train_arr = np.zeros(len(dorsal_idx))
dorsal_test_arr = np.zeros(len(dorsal_idx))

lateral_train_arr = np.zeros(len(lateral_idx))
lateral_test_arr = np.zeros(len(lateral_idx))


# for i in range(nd):
for i in range(len(ventral_idx)):
    tti = np.corrcoef(predict_train[:,i], session1_sub[:,i])[0,1]
    cci = np.corrcoef(predict_test[:,i], test1_sub[:,i])[0,1]
    rri = np.corrcoef(predict_test[:,i], test1_sub_shuffle_time[:,i])[0,1]
    ventral_train_arr[i] = tti
    ventral_test_arr[i] = cci
    ventral_testrand_arr[i] = rri
    # elif i in dorsal_idx:
    #     tti = np.corrcoef(predict_train[:,i], session1[:,i])[0,1]
    #     cci = np.corrcoef(predict_test[:,i], test1[:,i])[0,1]
    #     dorsal_train_arr[num2] = tti
    #     dorsal_test_arr[num2] = cci
    #     num2 += 1
    # elif i in lateral_idx:
    #     tti = np.corrcoef(predict_train[:,i], session1[:,i])[0,1]
    #     cci = np.corrcoef(predict_test[:,i], test1[:,i])[0,1]
    #     lateral_train_arr[num3] = tti
    #     lateral_test_arr[num3] = cci
    #     num3 += 1

print("Venral:", np.mean(ventral_train_arr), np.mean(ventral_test_arr), np.mean(ventral_testrand_arr))
print("Venral:", np.max(ventral_test_arr), np.min(ventral_test_arr))
# print("Dorsal:", np.mean(dorsal_train_arr), np.mean(dorsal_test_arr))
# print("Lateral:", np.mean(lateral_train_arr), np.mean(lateral_test_arr))

