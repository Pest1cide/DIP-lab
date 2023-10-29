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

def counter1 (input_image):
    h,w = input_image.shape
    dens = np.zeros(256)
    for i in range(h):
        for j in range(w):
            dens[input_image[i,j]] = dens[input_image[i,j]]+1
    
    for i in range(1,256):
        dens[i] = dens[i] + dens[i-1]
      
    return dens

def hist (input_image):
    h,w = input_image.shape
    dens = np.zeros(256)
    for i in range(h):
        for j in range(w):
            dens[input_image[i,j]] = dens[input_image[i,j]]+1
      
    return dens

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

def hist_match (input_image, spec_hist):
    map = np.zeros(256,dtype=np.uint8)
    h,w = input_image.shape
    
    
    dens = counter(input_image)
    output_dens = np.zeros(256)
    j = 0

    for i in range(256):
        
        diff = spec_hist - dens[i]    
        
        if (diff[j]==0):
            map[i] = j
            continue
        else :
            if(diff[j]>0):
                map[i] = j
                continue

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

def op_local_hist_equ (input_image, m_size):
    h,w = input_image.shape
    m_size = int((m_size-1)/2)
    input_hist = counter(input_image)
    input_image = input_image.astype(np.uint8)
    pad_image = np.pad(input_image,((m_size,m_size),(m_size,m_size)),'symmetric')
    

    
    for i in range(h):
        Beginner =  pad_image[i:i+2*m_size+1,0:2*m_size+1]
        dens = counter1(Beginner)
        old_col_dens = np.zeros(256)
        col_dens = np.zeros(256)
        for j in range(w):
            
            
            dens = dens- old_col_dens + col_dens
            input_image[i,j] = round(dens[input_image[i,j]]*255/(2*m_size+1)/(2*m_size+1))

            old_col = pad_image[i:i+2*m_size+1,j]
            old_col = old_col.reshape(-1,1)
            old_col_dens = counter1(old_col)
            if(j!=w-1):
                col = pad_image[i:i+2*m_size+1,j+2*m_size+1]
                col = col.reshape(-1,1)
                col_dens = counter1(col)
            

            

    output_hist = counter(input_image)
    return (input_image,output_hist,input_hist)



def sigmoid() :
    dens = np.zeros(256)
    for i in range(256):
        dens[i] = 1/(1+math.exp((-i+70)/23))


    return dens 





# np.set_printoptions(threshold=np.inf)
img = io.imread('D:\\Study!\\2023spring\\DIP\\env.jpg')
B,G,R = cv2.split(img) #get single 8-bits channel
EB=cv2.equalizeHist(B)
EG=cv2.equalizeHist(G)
ER=cv2.equalizeHist(R)
equal_test=cv2.merge((EB,EG,ER))  #merge it back
# cv2.imshow("test",test)
# cv2.imshow("equal_test",equal_test)



io.imsave('D:\\Study!\\2023spring\\DIP\\env1.jpg', equal_test)

# a = hist(img)
# new_img = hist_equ(img)
# io.imsave('D:\\Study!\\2023spring\\DIP\\hist_equ_match.tif', new_img)
# s = sigmoid()
# new_img1,c,d = hist_match(img,s)
# io.imsave('D:\\Study!\\2023spring\\DIP\\hist_match1.tif', new_img1)
# b = hist(new_img1)


# io.imsave('D:\\Study!\\2023spring\\DIP\\local3.tif', local)


##-------------------------------local hist match ----------------------
# new_img,a,b =local_hist_equ(img,3)

# io.imsave('D:\\Study!\\2023spring\\DIP\\new_img.tif', new_img)

# # img = cv2.imread('D:\\Study!\\2023spring\\DIP\\lab2\\Q3_1_1.tif')
##----------------------------------------------------------------------------




#---------------------------------------------plot col figure ---------

# plt.subplot(211)
# x= range(256)
# plt.plot(x,a)

# plt.subplot(212)
# plt.plot(x,b)






# plt.show()

#----------------------------------------------------------------------------





  
























