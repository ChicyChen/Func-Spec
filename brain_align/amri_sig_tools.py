from nilearn import plotting as nlp 
import nibabel as nb
import numpy as np                    
import torch
import math
import time 
import os
 
### signal processing: amri_sig_utils

def amri_sig_detrend(its, polyorder=3):
    # inputs
    #   its: input time series, numpy tensor with size (N, T)
    #   polyorder: the order of polynomial detrend

    # outputs
    #   ots: the detrended tensor (the same format as input)

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    # Compatible with: numpy, pytorch

    # check if a pytorch tensor; if not transform to pytorch
    is_numpy = not torch.is_tensor(its)
    if is_numpy:
        its = torch.from_numpy(its)
    its.requires_grad = False

    # get predictors
    N, T = its.shape 
    line = torch.linspace(1.0, T, steps=T).to(its).unsqueeze(0).unsqueeze(2)  #[1, T, 1]
    power = torch.linspace(.0, polyorder, steps=polyorder+1).to(its).unsqueeze(0).unsqueeze(1) #[1, 1, polyorder+1]
    poly = line.pow(power) # broadcast to the same shape: [1, T, order+1]

    # demean and linear regression
    its = its - its.mean(1).unsqueeze(1)
    weight = torch.linalg.lstsq(poly, its.reshape(N, T, 1)).solution
    trend = poly.matmul(weight)
    ots = its - trend.squeeze(2)

    # if input is numpy, get numpy output
    if is_numpy:
        ots = ots.detach().numpy()

    return ots, trend

def amri_sig_filtfft(ts, fs, lowcut=None, highcut=None, revfilt=False, trans=0.15): 
    # inputs
    #  ts:      numpy or pytorch tensor time series with size (N, T)
    #  fs:      sampling frequency of the time series
    #  lowcut:  lowcutoff frequency (in Hz)
    #  highcut: highcutoff frequency (in Hz)
    #  revfilt: False:band-pass; True:band-stop {default: False}
    #  trans:   relative transition zone {default: 0.15}

    # outputs
    #  ts_new:  the filtered time series vector

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    # Compatible with: numpy, pytorch

    # nextpower2
    def nextpow2(x): return pow(2, math.ceil(math.log(x)/math.log(2)))

    # check if a pytorch tensor; if not transform to pytorch
    is_numpy = not torch.is_tensor(ts)
    if is_numpy:
        ts = torch.from_numpy(ts)
    ts.requires_grad = False

    # create the frequency domain response vector
    N, npts = ts.shape
    nfft = nextpow2(npts);            # number of frequency points 
    fv=fs/2*torch.linspace(0,1,nfft//2+1);   # even-sized frequency vector 
    fres=(fs/2)/(nfft/2);      # frequency domain resolution
    freq_response = torch.ones(1, nfft//2+1).to(ts);   # filter is a keyword, and therefor is changed from the matlab version    

    # design frequency domain filter
    if (lowcut is None) and (highcut is None): 
        print('amri_sig_filtfft(): No filtering parameter is provided');
        raise RuntimeError
    elif ((lowcut is not None)&(lowcut>0))&((highcut is None)|(highcut<=0)): # high pass
        #          lowcut
        #              ----------- 
        #             /
        #            /
        #           /
        #-----------
        #    lowcut*(1-trans)
        idxl = round(lowcut/fres)+1;
        idxlmt = round(lowcut*(1-trans)/fres)+1;
        idxlmt = max([idxlmt,1]);
        freq_response[:, :idxlmt] = 0  # filter(1:idxlmt)=0;
        freq_response[:, idxlmt-1:idxl] = 0.5*(1+torch.sin(-torch.pi/2+torch.linspace(0,torch.pi,idxl-idxlmt+1))).to(ts);
    
    elif ((lowcut is None)|(lowcut<=0))&((highcut is not None)&(highcut>0)): # loss pass
        #        highcut
        # ----------
        #           \
        #            \
        #             \
        #              -----------
        #              highcut*(1+trans)
        idxh=round(highcut/fres)+1;                                                              
        idxhpt = round(highcut*(1+trans)/fres)+1;                                   
        freq_response[:, idxh-1:idxhpt]=0.5*(1+torch.sin(torch.pi/2+torch.linspace(0,torch.pi,idxhpt-idxh+1))).to(ts);
        freq_response[:, idxhpt-1:]=0;
        
    elif (lowcut>0)&(highcut>0)&(highcut>lowcut):  
        if revfilt is False:                           # bandpass (revfilt==0)
            #         lowcut   highcut
            #             -------
            #            /       \     transition = (highcut-lowcut)/2*trans
            #           /         \    center = (lowcut+highcut)/2;
            #          /           \
            #   -------             -----------
            # lowcut-transition  highcut+transition
            transition = (highcut-lowcut)/2*trans;
            idxl   = round(lowcut/fres)+1;
            idxlmt = round((lowcut-transition)/fres)+1;
            idxh   = round(highcut/fres)+1;
            idxhpt = round((highcut+transition)/fres)+1;
            idxl = max([idxl,1]);
            idxlmt = max([idxlmt,1]);
            idxh = min([nfft/2, idxh]);
            idxhpt = min([nfft/2, idxhpt]);
            freq_response[:, :idxlmt]=0;
            freq_response[:, idxlmt-1:idxl]=0.5*(1+torch.sin(-torch.pi/2+torch.linspace(0,torch.pi,idxl-idxlmt+1))).to(ts);
            freq_response[:, idxh-1:idxhpt]=0.5*(1+torch.sin(torch.pi/2+torch.linspace(0,torch.pi,idxhpt-idxh+1))).to(ts);
            freq_response[:, idxhpt-1:]=0;
            
        else:                                    # bandstop (revfilt==1)
            # lowcut-transition  highcut+transition
            #   -------             -----------
            #          \           /  
            #           \         /    transition = (highcut-lowcut)/2*trans
            #            \       /     center = (lowcut+highcut)/2;
            #             -------
            #         lowcut   highcut
            transition = (highcut-lowcut)/2*trans;
            idxl   = round(lowcut/fres)+1;
            idxlmt = round((lowcut-transition)/fres)+1;
            idxh   = round(highcut/fres)+1;
            idxhpt = round((highcut+transition)/fres)+1;
            idxlmt = max([idxlmt,1]);
            idxh = min([nfft/2, idxh]);
            idxhpt = min([nfft/2, idxhpt]);
            freq_response[:, idxlmt-1:idxl]=0.5*(1+torch.sin(torch.pi/2+torch.linspace(0,torch.pi,idxl-idxlmt+1))).to(ts);
            freq_response[:, idxl-1:idxh]=0;
            freq_response[:, idxh-1:idxhpt]=0.5*(1+torch.sin(-torch.pi/2+torch.linspace(0,torch.pi,idxl-idxlmt+1))).to(ts);
            # filter(nfft-idxhpt+1:nfft-idxlmt+1)=filter(idxhpt:-1:idxlmt);
            
    else:
        print('amri_sig_filtfft(): error in lowcut and highcut setting');
        raise RuntimeError
    
    # rfft and apply the frequency response
    X=torch.fft.rfft(ts,nfft);                         # half side fft because the input is real
    ts_new = torch.fft.irfft(X*freq_response,nfft);    # ifft for real output
    ts_new = ts_new[:, :npts];                         # tranc

    # if input is numpy, get numpy output
    if is_numpy:
        ts_new = ts_new.detach().numpy()

    return ts_new

def amri_sig_std(X, percentage=False):
    # Inputs
    #   X:          N x T data to be standardized
    #   percentage: True: demean and calculate percentage of change, otherwise demean and std 
    
    # Outputs
    #   Y: standarized signal

    # history:
    #   time   |   modifier   |   comments
    # 20220714     Kuan Han     create the initial version
    # 

    # Compatible with: numpy, pytorch

    # check if a pytorch tensor; if not transform to pytorch
    is_numpy = not torch.is_tensor(X)
    if is_numpy:
        X = torch.from_numpy(X)
    X.requires_grad = False

    # 
    eps = 1e-4
    Xmean = X.mean(1).unsqueeze(1)
    X = X - Xmean
    Xstd = X.std(1).unsqueeze(1)
    Y = X.div(Xmean+(Xmean==0)) if percentage else X.div(Xstd.clamp(eps))

    # if input is numpy, get numpy output
    if is_numpy:
        Y = Y.detach().numpy()

    return Y
    
       
