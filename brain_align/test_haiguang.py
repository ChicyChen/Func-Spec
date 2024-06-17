import h5py
import numpy as np
import scipy.io as sio
import mat73
from sklearn.linear_model import Ridge, RidgeCV, Lasso
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from scipy.io import savemat


filename = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/alexnet/AlexNet_feature_maps_pcareduced_concatenated.h5"
h5 = h5py.File(filename,'r')
all_keys = list(h5.keys())

features = []
for key in all_keys:
    feature_key = np.array(h5[key]['data'])
    features.append(feature_key)
features = np.vstack(features).T
print(features.shape)


features_test = []
for idx in range(5):
    test_list = []
    filename = f"/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/alexnet/AlexNet_feature_maps_pcareduced_test{idx+1}.h5"
    h5 = h5py.File(filename,'r')
    all_keys = list(h5.keys())
    for key in all_keys:
        test_key = np.array(h5[key]['data'])
        test_list.append(test_key)
    test_list = np.vstack(test_list).T
    features_test.append(test_list)
features_test = np.vstack(features_test) 
print(features_test.shape)



"""
print("Standardize")
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
features_test = scaler.transform(features_test)
"""

print("PCA")
n_components = 0.99
pca = PCA(n_components=n_components, svd_solver='full')
features = pca.fit_transform(features)
features_test = pca.transform(features_test) # use the same PCA as train
print(features.shape)
print(features_test.shape)



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


# cross validation
kfold = 4
sub_train_size = features.shape[0] // kfold
myCViterator = []
for i in range(kfold):
    val_idx = list(range(i*sub_train_size, (i+1)*sub_train_size))
    train_idx = list(range(features.shape[0]))
    del train_idx[i*sub_train_size: (i+1)*sub_train_size]
    myCViterator.append((train_idx, val_idx))
alpha_list = [0.01, 1.0, 100.0, 10000.0]
clf = RidgeCV(alphas=alpha_list, cv=myCViterator).fit(features, fmri_train)
alpha = clf.alpha_
print("cross validation best alpha:", alpha)



# final pass
clf = Ridge(alpha=alpha, max_iter=1000)
clf.fit(features, fmri_train)
predict_test = clf.predict(features_test)  # (240*5, 59412)
predict_train = clf.predict(features) # (240*18, 59412)
score_test = clf.score(features_test, fmri_test)
score_train = clf.score(features, fmri_train)
print("Scores:", score_train, score_test)



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
fm = "haiguang.mat"
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




# fmripath = "/data/3/human/Human_Visual_Experiments/video_fmri_dataset/stimuli/alexnet/AlexNet_feature_maps_processed_layer1_concatenated.mat"

# h5 = h5py.File(fmripath,'r')

# print(h5['lay_feat_concatenated'])

