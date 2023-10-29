from skimage import io
import numpy as np
import math
import cv2 as cv



img = io.imread('./rice.tif')
h,w = img.shape


def NN_interpolation (img,h,w):
    init_h ,init_w = img.shape
    new_img = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            p_x = round(i/(h-1)*(init_h-1))
            p_y = round(j/(w-1)*(init_w-1))
            new_img[i,j] = img[p_x,p_y]
    return new_img 




new_img = NN_interpolation(img,round(h*1.1),round(w*1.1))
io.imsave('D:\\Study!\\2023spring\\DIP\\enlarged_nearest_王子猛.tif', new_img) 
new_img1 = NN_interpolation(img,round(h*0.9),round(w*0.9))
io.imsave('D:\\Study!\\2023spring\\DIP\\shrunk_nearest_王子猛.tif', new_img1) 


########### below is comparing ###################### 

# enlarge = cv.resize(img, (0, 0), fx=1.1, fy=1.1, interpolation=cv.INTER_NEAREST)
# cv.imwrite('D:\\Study!\\2023spring\\DIP\\enlarged_nearest_cv2.tif',enlarge)


# import matplotlib.pyplot as plt           
# statis = np.zeros(100)
# substract = enlarge - new_img 
# for i in range(282):
#     for j in range(282):
#         if(substract[i,j]>100):
#             substract[i,j] = 255 - substract[i,j]
#         statis[substract[i,j]] +=1        

# x = range(100)
# # print(statis)
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.bar(x=x, height=statis)
# ax.set_title("error_nearest", fontsize=15)
# plt.show()            




