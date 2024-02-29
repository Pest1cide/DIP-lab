from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math


def CLS_filter(img,gamma,a=0.1,b=0.1,T=1):
    h,w = img.shape
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)

    mid_u = int(h/2)
    mid_v = int(w/2)
    H = np.zeros_like(img)
    for x in range(h):
        for y in range(w):
            u = x - mid_u
            v = y - mid_v
            c = u*a+v*b
            if (c==0):
                H[x,y] = 1
                continue 
            H[x,y] = T*math.sin(math.pi*(c))*math.e**(-math.pi*(c)*1j)/(math.pi*(c))
    
    abs_H = abs(H)
    P = np.zeros_like(img)
    laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float)
    P[h // 2 - 1: h // 2 + 2, w // 2 - 1: w // 2 + 2] = laplace
    P = np.fft.fftshift(np.fft.fft2(P))
    img =  np.conj(H)/(abs_H**2+gamma*abs(P)**2)*img

    img = np.fft.ifft2(np.fft.ifftshift(img)).real
    img = (img-img.min())/(img.max()-img.min())*255




    return img


