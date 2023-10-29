# -*- coding: utf-8 -*-

import  cv2
import numpy as np
import matplotlib.pyplot as plt


def is_inside(o,i):

    ox,oy,ow,oh = o
    ix,iy,iw,ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy+oh < iy+ih
    
def draw_person(img,person):

    x,y,w,h = person
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    

def detect_test(img):
    img = cv2.imread("C:\\Users\\31847\\Desktop\\pedes.jpg",)
    
    rows,cols = img.shape[:2]
    sacle = 1.0
    #print('img',img.shape)
    img = cv2.resize(img,dsize=(int(cols*sacle),int(rows*sacle)))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.float32(img)
    # img = img * 0.2
    # img = np.uint8(img)

    # cv2.imwrite("C:\\Users\\31847\\Desktop\\pedes_bl.jpg",img)
    # plt.imsave("C:\\Users\\31847\\Desktop\\pedes_bl.jpg",img)
    # print('img',img.shape)
    

    hog = cv2.HOGDescriptor()  
    #hist = hog.compute(img[0:128,0:64])   计算一个检测窗口的维度
    #print(hist.shape)
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()

    hog.setSVMDetector(detector)

    
    
    #多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
    found,w = hog.detectMultiScale(img)
    print(found)
    
    #过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
    found_filtered = []
    for ri,r in enumerate(found):
        for qi,q in enumerate(found):
            #r在q内？
            print(r)
            if ri != qi and is_inside(r,q):
                break
        else:
            found_filtered.append(r)
    # return found_filtered        
    for person in found_filtered:
        draw_person(img,person)
        
    cv2.imshow('img',img)
    cv2.imwrite("C:\\Users\\31847\\Desktop\\pedes_svm5.jpg",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    # while(True):
    #     ret, frame = cap.read()
    #     found_filtered  =detect_test(frame)
    #     if(len(found_filtered)>0):
    #         for person in found_filtered:
    #             x,y,w,h = person
    #             cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0, 255, 0), 2)
    #     cv2.imshow('frame',frame)
    #     if(cv2.waitKey(1) & 0xFF==ord('q')):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    detect_test(1)
        

