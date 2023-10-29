from skimage import io
import numpy as np
from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
import math


a = np.uint32(270)

b = np.uint32(180)
b = a+b
print(b)
np.clip(b,0,255,out=b)
b = np.uint8(b)
print(b)