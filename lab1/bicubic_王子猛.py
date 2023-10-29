from skimage import io
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt







# znew = np.zeros((round(h*1.1),round(w*1.1)),dtype=np.uint8)


# print(znew)

def BiCubic (img,h,w):
    init_h,init_w = img.shape


    x = np.arange(init_h)
    y = np.arange(init_w)
    z = img
    f = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.arange(h)
    xnew = xnew/(h-1)*(init_h-1)
    ynew = np.arange(w)
    ynew = ynew/(w-1)*(init_w-1)
    znew = f(xnew, ynew)
    znew = np.asarray(znew,dtype = np.uint8)



    return znew

img = io.imread('D:\\Study!\\2023spring\\DIP\\rice.tif')
# img = io.imread('./rice.tif')
h,w = img.shape
znew = BiCubic(img,round(h*1.1),round(w*1.1))
io.imsave('D:\\Study!\\2023spring\\DIP\\enlarged_bicubic_王子猛.tif', znew) 
print(znew.shape)
znew1 = BiCubic(img,round(h*0.9),round(w*0.9))
io.imsave('D:\\Study!\\2023spring\\DIP\\shrunk_bicubic_王子猛.tif', znew1) 


########### below is comparing ###################### 

# import cv2 as cv

# img = cv.imread('automobile.png')

# # 放大图像,双三次插值
# enlarge = cv.resize(img, (0, 0), fx=1.1, fy=1.1, interpolation=cv.INTER_CUBIC)
# cv.imwrite('D:\\Study!\\2023spring\\DIP\\enlarged_bicubic_cv2.tif',enlarge)
# np.set_printoptions(threshold=np.inf)

# statis = np.zeros(50)
# substract = enlarge - znew 
# for i in range(282):
#     for j in range(282):
#         if(substract[i,j]>100):
#             substract[i,j] = 255 - substract[i,j]
#         statis[substract[i,j]] +=1        

# x = range(50)
# # print(statis)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(x=x, height=statis)
# ax.set_title("error_bicubic", fontsize=15)
# plt.show()
