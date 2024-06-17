from sklearn.linear_model import Ridge
import numpy as np
import scipy.io as sio
import torch
import mat73
import pickle
import copy
import os
from sklearn.model_selection import train_test_split

# python ridge_regression_sub.py

do_pca = False

print("Load signal")

fmripath = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/training_fmri.mat'
fmripath_test = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/testing_fmri.mat'
fmri = sio.loadmat(fmripath)
test = mat73.loadmat(fmripath_test)
with open("ventral", "rb") as fp:
    ventral_idx = pickle.load(fp)
with open("dorsal", "rb") as fp2:
    dorsal_idx = pickle.load(fp2)
with open("lateral", "rb") as fp3:
    lateral_idx = pickle.load(fp3)

data1 = fmri['fmri'][0][0][0]
data2 = fmri['fmri'][0][0][1]
data_avg = (data1 + data2) / 2

fmri_train = []
for i in range(18):
    fmri_train.append(data_avg[:,:,i].T)
fmri_train = np.concatenate(fmri_train) # (240*18, 59412): T, n
print("fmri data size:", fmri_train.shape)

fmri_test = []
for i in range(5):
    fmri_test.append(np.mean(test['fmritest'][f'test{i+1}'], axis = 2).T)
fmri_test = np.concatenate(fmri_test) # (240*5, 59412): T, n

fmri_train = fmri_train[:, ventral_idx]
fmri_test = fmri_test[:, ventral_idx]
fmri_test_shuffle_time = copy.deepcopy(fmri_test)
np.random.shuffle(fmri_test_shuffle_time)
print("selected fmri data size:", fmri_train.shape)

print("Load features")

# features = torch.load("features_ds.pt") # (240, 367): T, d
# features_test = torch.load("features_ds_test.pt") # (240, 367): T, d
# load processed model features
folder = "base/adjust_encoder_input0/together"
# features_path = os.path.join(folder, "ds_features_all.npy")
# features_test_path = os.path.join(folder, "ds_features_test_all.npy")
features_path = os.path.join(folder, f"ds_features_all_2pca{do_pca}.npy")
features_test_path = os.path.join(folder, f"ds_features_test_all_2pca{do_pca}.npy")
features = np.load(features_path) # (240*18, 367): T, d
features_test = np.load(features_test_path) # (240*5, 367): T, d

print(features.shape)


print("Cross validation")

# alpha_list = [1.0, 10.0, 100.0, 1000.0, 10000.0]
alpha_list = [100000.0]

acc_list = np.zeros(len(alpha_list))

for idx in range(len(alpha_list)):
    alpha = alpha_list[idx]


    X_train, X_test, y_train, y_test = train_test_split(features, fmri_train, test_size = 0.25, random_state = 42)

    clf = Ridge(alpha=alpha)
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    print("Score:", score)

    predict_test = clf.predict(X_test)  # (240, subnum)
    predict_train = clf.predict(X_train) # (240, subnum)

    ventral_train_arr = np.zeros(len(ventral_idx))
    ventral_test_arr = np.zeros(len(ventral_idx))

    # for i in range(nd):
    for i in range(len(ventral_idx)):
        tti = np.corrcoef(predict_train[:,i], y_train[:,i])[0,1]
        cci = np.corrcoef(predict_test[:,i], y_test[:,i])[0,1]
        ventral_train_arr[i] = tti
        ventral_test_arr[i] = cci

    acc_list[idx] = np.mean(ventral_test_arr)
    print("Venral:", np.mean(ventral_train_arr), np.mean(ventral_test_arr))
    print("Venral:", np.max(ventral_test_arr), np.min(ventral_test_arr))
    # print("Dorsal:", np.mean(dorsal_train_arr), np.mean(dorsal_test_arr))
    # print("Lateral:", np.mean(lateral_train_arr), np.mean(lateral_test_arr))

best_alpha = alpha_list[np.argmax(acc_list)]
print("Best alpha:", best_alpha)
print("Best acc:", np.max(acc_list))

print("Final pass")

clf = Ridge(alpha=best_alpha)
clf.fit(features, fmri_train)
score = clf.score(features, fmri_train)
print("Score:", score)


predict_test = clf.predict(features_test)  # (240, subnum)
predict_train = clf.predict(features) # (240, subnum)


ventral_train_arr = np.zeros(len(ventral_idx))
ventral_test_arr = np.zeros(len(ventral_idx))
ventral_testrand_arr = np.zeros(len(ventral_idx))


# for i in range(nd):
for i in range(len(ventral_idx)):
    tti = np.corrcoef(predict_train[:,i], fmri_train[:,i])[0,1]
    cci = np.corrcoef(predict_test[:,i], fmri_test[:,i])[0,1]
    rri = np.corrcoef(predict_test[:,i], fmri_test_shuffle_time[:,i])[0,1]
    ventral_train_arr[i] = tti
    ventral_test_arr[i] = cci
    ventral_testrand_arr[i] = rri

acc_list[idx] = np.mean(ventral_test_arr)
print("Venral:", np.mean(ventral_train_arr), np.mean(ventral_test_arr), np.mean(ventral_testrand_arr))
print("Venral:", np.max(ventral_test_arr), np.min(ventral_test_arr))