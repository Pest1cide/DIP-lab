


from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math


def BLPF(image,d0,n):#巴特沃斯低通滤波器
    M,N = image.shape

    H = np.zeros((M,N))
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[x,y] = 1/(1+(d/d0)**(2*n))
    return H


def Inverse_Filtering(img,cut,k):
    h,w = img.shape
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)

    mid_u = int(h/2)
    mid_v = int(w/2)
    H = np.zeros_like(img)
    for u in range(h):
        for v in range(w):
            d = (u - mid_u) ** 2 + (v - mid_v) ** 2
            H[u,v] = math.exp(-k*d**(5/6))
    LP = BLPF(img,cut,10)
    img = img/H*LP

    img = np.fft.ifft2(np.fft.ifftshift(img))

    return img