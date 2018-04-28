# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:42:23 2018

@author: firebug
"""
import numpy as np
import cv2

arcMatrix=np.zeros((200,400), dtyps="vals")
oddKernels=np.zeros(36)
evenKernels=np.zeros(36)

def computeVpScore(filePath):
    
    #Mat image_origin = imread(filePath);
    #Mat img_gray, img_float;
    #cvtColor(image_origin, img_gray, CV_RGB2GRAY);
    #img_gray.convertTo(img_float, CV_32F);
    image_origin=cv2.imread(filePath)
    img_gray=cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    img_float=np.float(img_gray)
    
    ow,oh=image_origin.shape
    n_theta = 36;
    width = 128;
    scale_factor = width/oh;
     
    gw,gh=img_gray.shape
                                        
	#Mat image(img_gray.rows * scale_factor, width, CV_32F);
    image=np.zeros((gw*scale_factor,width), np.float32)
	#resize(img_float, image, image.size());
    image=cv2.resize(img_float,image.shape,interpolation=cv2.INTER_AREA)
     #int m = image.rows, n = image.cols;
    m,n=image.shape
 
     #float scores[85][128];
    scores=np.zeros((m,n),np.float32)
	#float ***gabors = new float**[m];
    gabors = np.zeros((m,n,n_theta),np.float32)
	#for_each(gabors, gabors + m, [n](float** &x) {x = new float*[n]; });
	#for_each(gabors, gabors + m, [n, n_theta](float** x) {for_each(x, x + n, [&, n_theta](float* &y) { y = new float[n_theta]; }); });

#Mat oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);
    # oddfiltered=np.zeros((m,n),np.float32)
    # evenfiltered=np.zeros((m,n),np.float32)
   
    oddfiltered=cv2.filter2D(image,-1,oddKernels)
    evenfiltered=cv2.filter2D(image,-1,evenKernels)
   
    for t in range(n_theta):
        for i in range (m):
            for j in range(n):
                gabors[i][j][t]=np.power(oddfiltered[i][j],2.0)+np.power(evenfiltered[i][j],2.0)
     
    directions=np.zeros((m,n),np.float32)
    for i in range(m):
        for j in range(n):
            directions[i][j]=np.max(gabors[i][j])-gabors[i][j]
    
    thresh=2.0*180/n_theta
    #r=(m+n)/7
    #r_dia=np.sqrt(m*m+n*n)
    gamma=0
    c=0
    for i in range(m):
        for j in range(n):
            tempScore=0
            for i1 in range(i+1,m):
                for j1 in range(n):
                    c=directions[i1][j1]/n_theta*180.0
                    gamma=arcMatrix[i1-1][j1-j+200].arc
                    if(np.abs(c-gamma)<thresh):
                        tempScore +=1
            scores[i][j]=tempScore    
    
    score_min,score_max,p_min,p_max=cv2.minMaxLoc(scores)
   # scale=score_max/255.0
    x=p_max.x/scale_factor
    y=p_max.y/scale_factor
    # 	cv::circle(image_origin, cvPoint(p_max.x / scale_factor, p_max.y / scale_factor), 10, Scalar(0), 5, 8, 0);
    cv2.circle(image_origin,(x,y),10,(0,0,0),5,8,0)
    return image_origin     
     
