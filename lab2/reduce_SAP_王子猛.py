from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math


def reduce_SAP (input_image, n_size):
    h,w = input_image.shape
    n_size = int((n_size-1)/2)
    # output_image = np.zeros((h,w),dtype=np.uint8)
    pad_image = np.pad(input_image,((n_size,n_size),(n_size,n_size)),'symmetric')


    for i in range(h):
        for j in range(w):
            temp = pad_image[i:i+2*n_size+1,j:j+2*n_size+1]
            temp = temp.flatten()
            temp = sorted(temp)
            # output_image[i,j] = temp[len(temp)//2]
            input_image[i,j] = temp[len(temp)//2]

    return input_image










