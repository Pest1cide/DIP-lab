from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math


def Harmonic_Mean (img ,s_size):
    h,w = img.shape
    s_size = int((s_size-1)/2)

    pad_img = np.pad(img,((s_size,s_size),(s_size,s_size)),'symmetric')

    for i in range(h):
        for j in range(w):
            S = pad_img[i:i+2*s_size+1,j:j+2*s_size+1]
            S = S.flatten()
            S = 1/S
            img[i,j] = (2*s_size+1)*(2*s_size+1)/S.sum()


    return img

def ContraHarmonic_Mean (img,s_size,Q):
    h,w = img.shape
    s_size = int((s_size-1)/2)

    pad_img = np.pad(img,((s_size,s_size),(s_size,s_size)),'symmetric')

    for i in range(h):
        for j in range(w):
            S = pad_img[i:i+2*s_size+1,j:j+2*s_size+1]
            S = np.float32(S.flatten())
            S = S+0.000001
            S_n = pow(S,Q+1)    #numerator
            S_d = pow(S,Q)     #denumoninator

            img[i,j] = S_n.sum()/S_d.sum()


    return img
