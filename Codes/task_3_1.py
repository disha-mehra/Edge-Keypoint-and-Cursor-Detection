# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:26:28 2018

@author: disha
"""

#Please read the comments for the explanation of why the specific code is put in

# Reference :https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html and TA suggestions

import numpy as np
import cv2

def cursor_detection(img_path,template_path):
    img = cv2.imread(img_path)
    #Read the image from the path
    
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur_img = cv2.GaussianBlur(img_gray,(3,3),0)
    #We blur the image to improvise our template matching technique   
    
    img_input=cv2.Laplacian(blur_img,cv2.CV_32F)    
    template = cv2.imread(template_path)
    
    h= template.shape[0]
    w= template.shape[1]
    
    blur_template= cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_temp=cv2.Laplacian(blur_template,cv2.CV_32F)
    
    result_img=cv2.matchTemplate(img_input,img_temp,cv2.TM_CCORR_NORMED)
    #We see that the result image has a bright spot and the bright spot is the cursor in most cases. 
    
    cv2.namedWindow('Result image', cv2.WINDOW_NORMAL)
    cv2.imshow('Result image',result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    min_val, max_val, min_location, max_location = cv2.minMaxLoc(result_img)
    
   # We set a threshold limit for the max_val because if dont consider it the code will come up with the max_val 
   #in every image and show the detector the which is not correct. 
    print(max_val)
    if max_val>0.31:
        top_left_corner = max_location
        right_bottom_corner = (top_left_corner[0] + h, top_left_corner[1] + w)
        cv2.rectangle(img,top_left_corner, right_bottom_corner,(0,255,255),2)
        
    cv2.namedWindow('Detected image', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
