# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:15:35 2018

@author: disha
"""

import cv2
import numpy as np

def octave(img_path):
    
    img= cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)  
        
    def gaussian_kernel(size,sigma_val):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)    
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_val**2))
        return kernel / np.sum(kernel)
    
    first_oct_sigma_values=[(1/(np.sqrt(2))),1,np.sqrt(2),2,2*(np.sqrt(2))]
    sec_oct_sigma_values=[np.sqrt(2),2,2*(np.sqrt(2)),4,4*(np.sqrt(2))]
    third_oct_sigma_values=[2*(np.sqrt(2)),4,4*(np.sqrt(2)),8,8*(np.sqrt(2))]
    fourth_oct_sigma_values=[4,4*(np.sqrt(2)),8,8*(np.sqrt(2)),16,16*(np.sqrt(2))]

    def compute(image,kernel):
        img_h=image.shape[0]
        img_w=image.shape[1]
        
        kernel_h=kernel.shape[0]
        kernel_w=kernel.shape[1]
        
        k_h=kernel_h//2
        k_w=kernel_w//2
        
        img_new=image.copy()
        for i in range(k_h,img_h-k_h):
            for j in range(k_w,img_w-k_w):
                val1=0
                for m in range(-k_h,(k_h+1)):
                    for n in range(-k_w,(k_w+1)):
                        val1=val1+image[i+m][j+n]*kernel[3+m][3+n]
                img_new[i,j]=val1
        return img_new
    
    #For first octave:
    
    kerns_1=[]
    gauss_conv_1=[]
    DoG_1_oct=[]
        
    for sigma_1 in first_oct_sigma_values:
        print(sigma_1)
        kerns_1.append(gaussian_kernel(7,sigma_1))
    
    for i in range(len(kerns_1)):
        img_out_1=img.copy()
        gauss_conv_1.append(compute(img_out_1,kerns_1[i])) 
    
    for i in range(len(gauss_conv_1)-1):
        DoG_1_oct.append(gauss_conv_1[i]-gauss_conv_1[i+1])
    
    
    #For second octave:
    
    kerns_2=[]
    gauss_conv_2=[]
    DoG_2_oct=[]
    res_gauss_2=[]

    for sigma_2 in sec_oct_sigma_values:
        print(sigma_2)
        kerns_2.append(gaussian_kernel(7,sigma_2))
    
    for i in range(len(kerns_2)):
        img_out_2=img_out_1[::2,::2]
        gauss_conv_2.append(compute(img_out_2,kerns_2[i])) 
        res_gauss_2.append(gauss_conv_2[i].shape)
    print('Resolution of Octave 2')
    print(res_gauss_2[1])
    
    for i in range(len(gauss_conv_2)-1):
        DoG_2_oct.append(gauss_conv_2[i]-gauss_conv_2[i+1])
      
    #For third octave:
    
    kerns_3=[]
    gauss_conv_3=[]
    DoG_3_oct=[]
    res_gauss_3=[]

    for sigma_3 in third_oct_sigma_values:
      #  print(sigma_3)
        kerns_3.append(gaussian_kernel(7,sigma_3))
    
    for i in range(len(kerns_3)):
        img_out_3=img_out_2[::2,::2]
        gauss_conv_3.append(compute(img_out_3,kerns_3[i])) 
        res_gauss_3.append(gauss_conv_3[i].shape)
    print('Resolution of Octave 3')
    print(res_gauss_3[1])
    
    for i in range(len(gauss_conv_3)-1):
        DoG_3_oct.append(gauss_conv_3[i]-gauss_conv_3[i+1])
    
    #For fourth octave:
     
    kerns_4=[]
    gauss_conv_4=[]
    DoG_4_oct=[]
    
    
    for sigma_4 in fourth_oct_sigma_values:
      #  print(sigma_4)
        kerns_4.append(gaussian_kernel(7,sigma_4))
    
    for i in range(len(kerns_4)):
        img_out_4=img_out_3[::2,::2]
        gauss_conv_4.append(compute(img_out_4,kerns_4[i])) 
        
    
    for i in range(len(gauss_conv_4)-1):
        DoG_4_oct.append(gauss_conv_4[i]-gauss_conv_4[i+1])
    
    
    for i in range(5):
        cv2.imshow('Octave 2',gauss_conv_2[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    
    for i in range(5):
        cv2.imshow('Octave 3',gauss_conv_3[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    for i in range(4):
        cv2.imshow('DoG_2',DoG_2_oct[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
        
    for i in range(4):
        cv2.imshow('DoG_3',DoG_3_oct[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
