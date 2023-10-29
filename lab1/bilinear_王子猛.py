from skimage import io
import numpy as np
import math
import cv2 as cv



img = io.imread('./rice.tif')
h,w = img.shape


def BiLinearInter (img,h,w):
    init_h ,init_w = img.shape
    new_img = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            x_deci,x_int = math.modf(i/(h-1)*(init_h-1))  
            y_deci,y_int = math.modf(j/(w-1)*(init_w-1))  
            x_int = (int)(x_int)
            y_int = (int)(y_int)
            new_img[i,j] = (1-y_deci)*(x_deci*img[x_int+(x_deci!=0),y_int]+(1-x_deci)*img[x_int,y_int])+y_deci*(x_deci*img[x_int+(x_deci!=0),y_int+(y_deci!=0)]+(1-x_deci)*img[x_int,y_int+(y_deci!=0)])
    return new_img

            
BLI_img = BiLinearInter(img,round(h*1.1),round(w*1.1))
# print(BLI_img)
io.imsave('D:\\Study!\\2023spring\\DIP\\enlarged_biliear_王子猛.tif', BLI_img)

BLI_img1 = BiLinearInter(img,round(h*0.9),round(w*0.9))
# print(BLI_img)
io.imsave('D:\\Study!\\2023spring\\DIP\\shrunk_biliear_王子猛.tif', BLI_img1)
# print(BLI_img.shape)








########### below is comparing ###################### 


# enlarge = cv.resize(img, (0, 0), fx=1.1, fy=1.1, interpolation=cv.INTER_LINEAR)
# cv.imwrite('D:\\Study!\\2023spring\\DIP\\enlarged_bilinear_cv2.tif',enlarge)           
            
# import matplotlib.pyplot as plt           
# statis = np.zeros(50)
# substract = enlarge - BLI_img 
# for i in range(282):
#     for j in range(282):
#         if(substract[i,j]>100):
#             substract[i,j] = 255 - substract[i,j]
#         statis[substract[i,j]] +=1        

# x = range(50)
# # print(statis)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(x=x, height=statis)
# ax.set_title("error_bilinear", fontsize=15)
# plt.show()            















