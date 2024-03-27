import nibabel as nb
import numpy as np 
import pickle

ventral_list = [33, 35, 37, 41, 48, 56, 111, 163, 172]
ventral_list2 = [item + 180 for item in ventral_list]

dorsal_list = [34, 40, 54, 62, 95, 131, 142]
dorsal_list2 = [item + 180 for item in dorsal_list]

lateral_list = [20, 21, 42, 46, 48, 70, 76, 95, 159, 174]
lateral_list2 = [item + 180 for item in lateral_list]

path = 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
file = nb.load(path)

cifti_data = file.get_fdata(dtype=np.float32)
data = cifti_data[0].astype(np.int32)

ventral_idx = []
for i in range(data.shape[0]):
    if data[i] in ventral_list or data[i] in ventral_list2:
        ventral_idx.append(i)
print(len(ventral_idx))
with open("ventral", "wb") as fp:   #Pickling
    pickle.dump(ventral_idx, fp)

dorsal_idx = []
for i in range(data.shape[0]):
    if data[i] in dorsal_list or data[i] in dorsal_list2:
        dorsal_idx.append(i)
print(len(dorsal_idx))
with open("dorsal", "wb") as fp:   #Pickling
    pickle.dump(dorsal_idx, fp)

lateral_idx = []
for i in range(data.shape[0]):
    if data[i] in lateral_list or data[i] in lateral_list2:
        lateral_idx.append(i)
print(len(lateral_idx))
with open("lateral", "wb") as fp:   #Pickling
    pickle.dump(lateral_idx, fp)

# cifti_hdr = file.header
# axes = [cifti_hdr.get_axis(i) for i in range(file.ndim)]
# print(max(cifti_data[0]))
# print(min(cifti_data[0]))

# total_nan = 0
# for i in range(cifti_data[0].shape[0]):
#     if cifti_data[0][i] != cifti_data[0][i]:
#         total_nan += 1
# print(total_nan)

# print(axes[0])