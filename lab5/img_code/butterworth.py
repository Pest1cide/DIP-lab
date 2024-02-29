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


    a = pad_img.copy()
    a = 10 * np.log(np.abs(a))
    a = np.real(a * kernal)
    

    pad_img = pad_img * kernal


    pad_img = np.fft.ifftshift(pad_img)
    pad_img = np.fft.ifft2(pad_img)
    pad_img = np.real(pad_img)


    return pad_img[:h,:w]

def BLPF(image,d0,n):#巴特沃斯低通滤波器
    M,N = image.shape
    M = M*2
    N = N*2
    H = np.zeros((M,N))
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[x,y] = 1/(1+(d/d0)**(2*n))
    return H

def BBPF(image,d0,n,u,v):#巴特沃斯低通滤波器
    M,N = image.shape
    M = M*2
    N = N*2
    H = np.zeros((M,N))
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0, N):
            d1 = np.sqrt((x - mid_x - v) ** 2 + (y - mid_y + u) ** 2)
            d2 = np.sqrt((x - mid_x - v) ** 2 + (y - mid_y - u) ** 2)
            d3 = np.sqrt((x - mid_x + v) ** 2 + (y - mid_y + u) ** 2)
            d4 = np.sqrt((x - mid_x + v) ** 2 + (y - mid_y - u) ** 2)
            H[x,y] = 1/(1+(d1/d0)**(2*n)) + 1/(1+(d2/d0)**(2*n)) + 1/(1+(d3/d0)**(2*n)) + 1/(1+(d4/d0)**(2*n))
    return H



def BHPF(image,d0,n):#巴特沃斯低通滤波器
    H = BLPF(image,d0,n)
    H = 1 - H
    return H
