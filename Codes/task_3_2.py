# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:15:11 2018

@author: disha
"""

#Please read the comments

#Reference: https://stackoverflow.com/questions/43339287/

import cv2
import numpy as np
import glob

def task3_2(img_path,temp_path):    

    #List to store template images
    template_data=[]
    
    #Listing of the template images with the extension .png. I saved all the 3 different template images in .png
    templates= glob.glob(temp_path +'\\*.png')
    
    for i in templates:
        temp = cv2.imread(i,0)
        template_data.append(temp)
    
    test_image=cv2.imread(img_path)    
    
    img_gray= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)        
    
    blur_img = cv2.GaussianBlur(img_gray,(3,3),0)
        
    img_input=cv2.Laplacian(blur_img,cv2.CV_32F)
    
    #loop for matching
    for template in template_data:
        h = template.shape[0]
        w = template.shape[1]
        
        cv2.imshow("Template", template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        img_temp=cv2.Laplacian(template,cv2.CV_32F)
        
        result = cv2.matchTemplate(img_input, img_temp, cv2.TM_CCORR_NORMED)
        
        min_val, max_val, min_location, max_location = cv2.minMaxLoc(result)
        
        print(max_val)
        if max_val>0.4:
            top_left = max_location
            bottom_right = (top_left[0] + h, top_left[1] + w)
            cv2.rectangle(test_image,top_left, bottom_right,(0,255,255),2)
    
    cv2.imshow('Result',test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()