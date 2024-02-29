from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math

def hist_equ (input_image):
    h,w = input_image.shape
    dens = counter(input_image)
    for i in range(h):
        for j in range(w):
            input_image[i,j] = round(dens[input_image[i,j]]*255)

    return input_image


    
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
