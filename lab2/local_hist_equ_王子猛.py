from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math







def counter (input_image):
    h,w = input_image.shape
    dens = np.zeros(256)
    for i in range(h):
        for j in range(w):
            dens[input_image[i,j]] = dens[input_image[i,j]]+1
    dens = dens/(h*w)
    for i in range(1,256):
        dens[i] = dens[i] + dens[i-1]
      
    return dens



def local_hist_equ (input_image, m_size):
    h,w = input_image.shape
    m_size = int((m_size-1)/2)
    input_hist = counter(input_image)
    input_image = input_image.astype(np.uint8)
    pad_image = np.pad(input_image,((m_size,m_size),(m_size,m_size)),'symmetric')
    for i in range(h):
        for j in range(w):
            temp = pad_image[i:i+2*m_size+1,j:j+2*m_size+1]
            dens = counter(temp)
            # dens = (dens-dens.min())*255/(dens.max()-dens.min())
            input_image[i,j] = round(dens[input_image[i,j]]*255)

    output_hist = counter(input_image)

    return (input_image,output_hist,input_hist)