from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def Conv(img,kernal):
    h,w = img.shape
    # pad_img = img
    pad_img = np.uint8(np.pad(img,((0,h),(0,w)),constant_values=0))
    pad_img = np.fft.fft2(pad_img)
    pad_img = np.fft.fftshift(pad_img)

    
    # a = np.real(pad_img)
    a = pad_img.copy()
    a = 10 * np.log(np.abs(a))
    a = np.real(a * kernal)
    
    # pad_img = 10 * np.log(np.abs(pad_img))
    # a = np.real(pad_img)
    pad_img = pad_img * kernal

    pad_img = np.fft.ifftshift(pad_img)
    pad_img = np.fft.ifft2(pad_img)
    pad_img = np.real(pad_img)
    # pad_img = abs(pad_img)

    return pad_img[:h,:w]

def BHPF(image,d0,n):#巴特沃斯低通滤波器
    H = BLPF(image,d0,n)
    H = 1 - H
    return H



def GLPF(image,d0):#高斯低通滤波器
    M,N = image.shape
    M = M*2
    N = N*2
    H = np.zeros((M,N))
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0, N):
            d = (x - mid_x)**2 + (y - mid_y)**2
            H[x, y] = np.exp(-d/(2*d0**2))
    return H

def GHPF(image,d0):#高斯低通滤波器
    H = GLPF(image,d0)
    H = 1 - H

    return H


def ILPF(image,d0):#理想低通滤波器
    M,N = image.shape
    M = M*2
    N = N*2
    H = np.zeros((M,N))
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0,N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            if d <= d0:
                H[x, y] = 1
    # H[mid_x-d0:mid_x+d0,mid_y-d0:mid_y+d0] = 1
    return H

           

def IHPF(image,d0):#理想低通滤波器
    
    H = ILPF(image,d0)
    H = 1-H

           
    return H