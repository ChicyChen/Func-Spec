import torch.nn as nn
from resnet import r3d_18

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



resnet1 = r3d_18() # original
resnet2 = r3d_18(width_deduction_ratio = 1.41, stem_deduct = True) # change four blocks and stem
resnet3 = r3d_18(width_deduction_ratio = 2.0, stem_deduct = True) # change four blocks and stem

num_para1 = get_n_params(resnet1)
num_para2 = get_n_params(resnet2)
num_para3 = get_n_params(resnet3)

print(num_para1)
print(num_para2)
print(num_para3)


