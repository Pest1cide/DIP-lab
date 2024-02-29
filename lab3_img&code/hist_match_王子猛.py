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



def hist_match (input_image, spec_hist):
    map = np.zeros(256,dtype=np.uint8)
    h,w = input_image.shape
    
    
    dens = counter(input_image)
    output_dens = np.zeros(256)
    j = 0

    for i in range(256):
        
        diff = spec_hist - dens[i]    
        
        if (diff[j]>=0):
            map[i] = j
            continue
        else :
            while(diff[j]<0 ):
                if(j == 255):
                    map[i] = j
                    break
                j +=1
                 
            if(-diff[j-1]>diff[j]):
                map[i] = j
                continue
            else:
                map[i] = j-1
                continue

    for i in range (h):
        for j in range(w):
            input_image[i,j] = map[input_image[i,j]]
    for i in range(256):
        output_dens[map[i]] += dens[i]    

    return input_image,output_dens,dens 
    





