from sklearn.linear_model import Ridge, RidgeCV, Lasso
import numpy as np
import scipy.io as sio
import torch
import mat73
import pickle
import os
from scipy.io import savemat

import logging
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='dorsal_feature/adjust4_encoder2_input3_srate0.5', type=str)
parser.add_argument('--pca', action='store_true')

parser.add_argument('--layer_wise', action='store_true')
parser.add_argument('--layer_name', default='layer1.0', type=str)


args = parser.parse_args()


logging.basicConfig(filename=os.path.join(args.root, 'net3d_vic_train.log'), level=logging.INFO)
logging.info('Started')

# do_pca = True
do_pca = args.pca

print("Load fmri data")
# load train fmri data, average, and concatenate
fmripath = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/training_fmri.mat'
fmripath_test = '/data/3/human/Human_Visual_Experiments/video_fmri_dataset/subject1/fmri/testing_fmri.mat'
fmri = sio.loadmat(fmripath)
test = mat73.loadmat(fmripath_test)

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

root = args.root
folder = os.path.join(root,"together")
features_path = os.path.join(folder, f"ds_features_all_2pca{do_pca}.npy")
features_test_path = os.path.join(folder, f"ds_features_test_all_2pca{do_pca}.npy")

features = np.load(features_path) # (240*18, 367): T, d
features_test = np.load(features_test_path) # (240*5, 367): T, d

print("network data size:", features.shape)



# cross validation
kfold = 4
sub_train_size = features.shape[0] // kfold
myCViterator = []
for i in range(kfold):
    val_idx = list(range(i*sub_train_size, (i+1)*sub_train_size))
    train_idx = list(range(features.shape[0]))
    del train_idx[i*sub_train_size: (i+1)*sub_train_size]
    myCViterator.append((train_idx, val_idx))
alpha_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
clf = RidgeCV(alphas=alpha_list, cv=myCViterator).fit(features, fmri_train)
alpha = clf.alpha_
print("cross validation best alpha:", alpha)
logging.info(f"cross validation best alpha: {alpha}")


# final pass
clf = Ridge(alpha=alpha, max_iter=1000)
clf.fit(features, fmri_train)
predict_test = clf.predict(features_test)  # (240*5, 59412)
predict_train = clf.predict(features) # (240*18, 59412)
score_test = clf.score(features_test, fmri_test)
score_train = clf.score(features, fmri_train)
print("Scores:", score_train, score_test)
logging.info(f"Scores: {score_train} {score_test}")



# test all
nd = 59412
train_arr = np.zeros((nd,1))
test_arr = np.zeros((nd,1))
for i in range(nd):
    tti = np.corrcoef(predict_train[:,i], fmri_train[:,i])[0,1]
    cci = np.corrcoef(predict_test[:,i], fmri_test[:,i])[0,1]
    train_arr[i] = tti
    test_arr[i] = cci
print("All:", np.mean(train_arr), np.mean(test_arr))
logging.info(f"All: {np.mean(train_arr)} {np.mean(test_arr)}")
fm = os.path.join(root, f"corr_pca{do_pca}.mat")
fm_data = {"train": train_arr, "test": test_arr}
savemat(fm, fm_data)
best_acc = np.mean(test_arr)


# test ventral, dorsal, and lateral
with open("ventral", "rb") as fp:
    ventral_idx = pickle.load(fp)
with open("dorsal", "rb") as fp2:
    dorsal_idx = pickle.load(fp2)
# with open("lateral", "rb") as fp3:
#     lateral_idx = pickle.load(fp3)

ventral_train_arr = np.zeros(len(ventral_idx))
ventral_test_arr = np.zeros(len(ventral_idx))
num = 0
dorsal_train_arr = np.zeros(len(dorsal_idx))
dorsal_test_arr = np.zeros(len(dorsal_idx))
num2 = 0
# lateral_train_arr = np.zeros(len(lateral_idx))
# lateral_test_arr = np.zeros(len(lateral_idx))
# num3 = 0

for i in range(nd):
    if i in ventral_idx:
        tti = np.corrcoef(predict_train[:,i], fmri_train[:,i])[0,1]
        cci = np.corrcoef(predict_test[:,i], fmri_test[:,i])[0,1]
        ventral_train_arr[num] = tti
        ventral_test_arr[num] = cci
        num += 1

    if i in dorsal_idx:
        tti = np.corrcoef(predict_train[:,i], fmri_train[:,i])[0,1]
        cci = np.corrcoef(predict_test[:,i], fmri_test[:,i])[0,1]
        dorsal_train_arr[num2] = tti
        dorsal_test_arr[num2] = cci
        num2 += 1

    # if i in lateral_idx:
    #     tti = np.corrcoef(predict_train[:,i], fmri_train[:,i])[0,1]
    #     cci = np.corrcoef(predict_test[:,i], fmri_test[:,i])[0,1]
    #     lateral_train_arr[num3] = tti
    #     lateral_test_arr[num3] = cci
    #     num3 += 1

print("Venral:", np.mean(ventral_train_arr), np.mean(ventral_test_arr), np.max(ventral_test_arr))
print("Dorsal:", np.mean(dorsal_train_arr), np.mean(dorsal_test_arr), np.max(dorsal_test_arr))
# print("Lateral:", np.mean(lateral_train_arr), np.mean(lateral_test_arr), np.max(lateral_test_arr))

logging.info(f"Venral: {np.mean(ventral_train_arr)} {np.mean(ventral_test_arr)} {np.max(ventral_test_arr)}")
logging.info(f"Dorsal: {np.mean(dorsal_train_arr)} {np.mean(dorsal_test_arr)} {np.max(dorsal_test_arr)}")




