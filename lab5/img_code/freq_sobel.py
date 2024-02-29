from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def freq_sobel (img):
    h,w = img.shape
    pad_img = np.pad(img,((0,h),(0,w)),constant_values=0)
    pad_img = np.fft.fft2(pad_img)
    pad_img = np.fft.fftshift(pad_img)
    
    
    dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # X方向
    dy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # Y方向

    dx = np.pad(dx,(((h*2-3)//2+1,(h*2-3)//2),((w*2-3)//2+1,(w*2-3)//2)),constant_values=0)
    dy = np.pad(dy,(((h*2-3)//2+1,(h*2-3)//2),((w*2-3)//2+1,(w*2-3)//2)),constant_values=0)

    # dx = np.fft.fftshift(np.fft.fft2(dx))
    # dy = np.fft.fftshift(np.fft.fft2(dy))
    dx = (np.fft.fft2(dx))
    dy = (np.fft.fft2(dy))
    
    # for i in range(h):
    #     for j in range(w):
    #         dx[i,j] *= (-1)**(i+j) 
    #         dy[i,j] *= (-1)**(i+j) 

    dx = np.fft.fftshift((dx))
    dy = np.fft.fftshift((dy))

    pad_img = 0.5*abs(np.fft.ifft2(np.fft.ifftshift(pad_img*dx)).real) + 0.5*abs(np.fft.ifft2(np.fft.ifftshift(pad_img*dy)).real)



    return pad_img
