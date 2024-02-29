from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math



def Alpha_Trim (img,s_size,d):
    h,w = img.shape
    s_size = int((s_size-1)/2)

    pad_img = np.pad(img,((s_size,s_size),(s_size,s_size)),'symmetric')

    for i in range(h):
        for j in range(w):
            S = pad_img[i:i+2*s_size+1,j:j+2*s_size+1]
            S = S.flatten()
            S = sorted(S)
            S = np.array(S[d//2:len(S)-d//2])
            img[i,j] = S.sum()/len(S)

    return img    



def Adaptive_Mean (img,s_size,smax):
    h,w = img.shape
    s_size = int((s_size-1)/2)
    s_max = int((smax-1)/2)
    pad_img = np.pad(img,((s_max,s_max),(s_max,s_max)),'symmetric')
    origin_s =s_size
    p_size = s_max
    for i in range(h):
        for j in range(w):
            S = pad_img[i-s_size+p_size:i+s_size+p_size+1,j-s_size+p_size:j+s_size+p_size+1]
            S = S.flatten()
            gmin = S.min()
            gmax = S.max()
            gmed = np.median(S)
            g = img[i,j]
            ## Stage A
            if(gmin<gmed<gmax):
                if (gmin<g<gmax):
                    img[i,j] = img[i,j]
                else :
                    img[i,j] = gmed
            else :  ## Stage B
                while s_size<=s_max:
                    s_size = s_size+1
                    S = pad_img[i-s_size+p_size:i+s_size+p_size+1,j-s_size+p_size:j+s_size+p_size+1]
                    S = S.flatten()
                    gmin = S.min()
                    gmax = S.max()
                    gmed = np.median(S)
                    g = img[i,j]
                    ### Stage A in Stage B ##
                    if(gmin<gmed<gmax):
                        if (gmin<g<gmax):
                            img[i,j] = img[i,j]
                            
                        else :
                            img[i,j] = gmed
                        s_size = origin_s
                        break
                    if(s_size==s_max):
                        img[i,j] = gmed
                        s_size= origin_s
                        break
    return img   
