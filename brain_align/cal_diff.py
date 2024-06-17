import torch

def RGB2Gray(video):
    # formula of RGB to gray: Y = 0.2125 R + 0.7154 G + 0.0721 B
    # convert a batch of rgb video frames augmented in 2 different ways into grayscale
    # shape of the input "video" is [B, N, C, T, H, W]
    # shape of the output "grayscle" is [B, N, T, H, W]
    gray = 0.2125*video[:,:,0,:,:,:] + 0.7154*video[:,:,1,:,:,:] + 0.0721*video[:,:,2,:,:,:]
    return gray

def padding(data):
    # padding H and W by 1 with replicated value
    # F.pad(input, (1,1,1,1,0,0), mode = 'replicate') supposed to do the same thing
    # but when running F.pad(...) there is a strange error
    pad_data_tmp = torch.cat((data[...,0:1], data, data[...,-1:]),4)
    pad_data = torch.cat((pad_data_tmp[...,0:1,:], pad_data_tmp, pad_data_tmp[...,-1:,:]),3)
    return pad_data

def poisson_blend(Ix,Iy,iteration=50):
    #Ix, Iy can only be np array...?
    #shape of Ix and Iy are now B, N, T, H, W
    device = Ix.device
    lap_blend = torch.zeros(Ix.shape,  device=device)
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.detach().clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[...,1:-1,2:] -  Iy[...,1:-1,1:-1]
                    + Iy[...,2:,1:-1] -  Ix[...,1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[...,2:,1:-1] + lap_blend_old[...,0:-2,1:-1]
                                 + lap_blend_old[...,1:-1,2:] + lap_blend_old[...,1:-1,0:-2])
        lap_blend[...,1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend

def vizDiff(d,thresh=0.24):
    # shape of input is B,N,T,H,W
    device = d.device
    diff = d.detach().clone()
    rgb_diff = 0
    B,N,T,H,W = diff.shape
    rgb_diff = torch.zeros([B,N,3,T,H,W], device=device) #background is zero
    diff[abs(diff)<thresh] = 0
    rgb_diff[:,:,0,...][diff>0] = diff[diff>0] # diff[diff>0]
    rgb_diff[:,:,1,...][diff>0] = diff[diff>0]
    rgb_diff[:,:,2,...][diff>0] = diff[diff>0]
    rgb_diff[:,:,0,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,1,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,2,...][diff<0] = diff[diff<0]
    return rgb_diff

def get_spatial_diff(data):
    # data is grayscale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get SD_x and SD_y, both with shape B,N,T,H,W
    # step2: use poisson blending to get SD_xy
    # step3: based on the value of SD_xy in the last two dimensions, convert it back to B, N, C, T,H,W
    padded_data = padding(data)
    SD_x = (padded_data[...,1:-1,:-2] - padded_data[...,1:-1, 2:])/2
    SD_y = (padded_data[...,:-2,1:-1] - padded_data[...,2:,1:-1])/2
    SD_xy = poisson_blend(SD_x, SD_y)
    SD_xy = vizDiff(SD_xy)
    return SD_xy

def get_temporal_diff(data):
    # data is grascale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get TD, with shape B,N,T-1,H,W
    # step2 convert TD back to B,N,C,T,H,W
    TDiff = data[:,:,1:,:,:] - data[:,:,:-1,:,:]
    TDiff = vizDiff(TDiff)
    return TDiff

def RGB2Gray(video):
    # formula of RGB to gray: Y = 0.2125 R + 0.7154 G + 0.0721 B
    # convert a batch of rgb video frames augmented in 2 different ways into grayscale
    # shape of the input "video" is [B, N, C, T, H, W]
    # shape of the output "grayscle" is [B, N, T, H, W]
    gray = 0.2125*video[:,:,0,:,:,:] + 0.7154*video[:,:,1,:,:,:] + 0.0721*video[:,:,2,:,:,:]
    return gray

def padding(data):
    # padding H and W by 1 with replicated value
    # F.pad(input, (1,1,1,1,0,0), mode = 'replicate') supposed to do the same thing
    # but when running F.pad(...) there is a strange error
    pad_data_tmp = torch.cat((data[...,0:1], data, data[...,-1:]),4)
    pad_data = torch.cat((pad_data_tmp[...,0:1,:], pad_data_tmp, pad_data_tmp[...,-1:,:]),3)
    return pad_data

def poisson_blend(Ix,Iy,iteration=50):
    #Ix, Iy can only be np array...?
    #shape of Ix and Iy are now B, N, T, H, W
    device = Ix.device
    lap_blend = torch.zeros(Ix.shape,  device=device)
    # Perform Poisson iteration
    for i in range(iteration):
        lap_blend_old = lap_blend.detach().clone()
        # Update the Laplacian values at each pixel
        grad = 1/4 * (Ix[...,1:-1,2:] -  Iy[...,1:-1,1:-1]
                    + Iy[...,2:,1:-1] -  Ix[...,1:-1,1:-1])
        lap_blend_old_tmp = 1/4 * (lap_blend_old[...,2:,1:-1] + lap_blend_old[...,0:-2,1:-1]
                                 + lap_blend_old[...,1:-1,2:] + lap_blend_old[...,1:-1,0:-2])
        lap_blend[...,1:-1,1:-1] = lap_blend_old_tmp + grad
        # Check for convergence
        if torch.sum(torch.abs(lap_blend - lap_blend_old)) < 0.1:
            #print("converged")
            break
    # Return the blended image
    return lap_blend

def vizDiff(d,thresh=0.24):
    # shape of input is B,N,T,H,W
    device = d.device
    diff = d.detach().clone()
    rgb_diff = 0
    B,N,T,H,W = diff.shape
    rgb_diff = torch.zeros([B,N,3,T,H,W], device=device) #background is zero
    diff[abs(diff)<thresh] = 0
    rgb_diff[:,:,0,...][diff>0] = diff[diff>0] # diff[diff>0]
    rgb_diff[:,:,1,...][diff>0] = diff[diff>0]
    rgb_diff[:,:,2,...][diff>0] = diff[diff>0]
    rgb_diff[:,:,0,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,1,...][diff<0] = diff[diff<0]
    rgb_diff[:,:,2,...][diff<0] = diff[diff<0]
    return rgb_diff

def get_spatial_diff(data):
    # data is grayscale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get SD_x and SD_y, both with shape B,N,T,H,W
    # step2: use poisson blending to get SD_xy
    # step3: based on the value of SD_xy in the last two dimensions, convert it back to B, N, C, T,H,W
    padded_data = padding(data)
    SD_x = (padded_data[...,1:-1,:-2] - padded_data[...,1:-1, 2:])/2
    SD_y = (padded_data[...,:-2,1:-1] - padded_data[...,2:,1:-1])/2
    SD_xy = poisson_blend(SD_x, SD_y)
    SD_xy = vizDiff(SD_xy)
    return SD_xy

def get_temporal_diff(data):
    # data is grascale-like and the shape of input data is B, N, T, H, W
    # TODO: complete the function without change the input
    # step1: get TD, with shape B,N,T-1,H,W
    # step2 convert TD back to B,N,C,T,H,W
    TDiff = data[:,:,1:,:,:] - data[:,:,:-1,:,:]
    TDiff = vizDiff(TDiff)
    return TDiff



# grayscale_video = RGB2Gray(video) # B,N,T,H,W
# video_sd = get_spatial_diff(grayscale_video)
# video_td = get_temporal_diff(grayscale_video)