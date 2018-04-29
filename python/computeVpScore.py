# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:42:23 2018

@author: firebug
"""
import numpy as np
import cv2
from math import cos, sin, exp, pi, acos, ceil
import time
import os

vals = np.dtype([("arc", np.float32), ("d", np.float32)])
arcMatrix = np.zeros((200, 400), dtype=vals)
oddKernels = list()
evenKernels = list()

# , psi, gamma, ktype
def myGarborKernel(ksize, sigma, theta, Lambda):
    k = int((ksize - 1) / 2)
    odd_kernel = np.zeros((ksize, ksize), np.float32)
    even_kernel = np.zeros((ksize, ksize), np.float32)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            a = i * cos(theta) + j * sin(theta)
            b = j * cos(theta) - i * sin(theta)
            odd_resp = exp(-1.0 / 8.0 / sigma / sigma * (4.0 * a * a + b * b)) * sin(2.0 * pi * a / Lambda)
            even_resp = exp(-1.0 / 8.0 / sigma / sigma * (4.0 * a * a + b * b)) * cos(2.0 * pi * a / Lambda)
            odd_kernel[i + k, j + k] = odd_resp
            even_kernel[i + k, j + k] = even_resp

    odd_mean = odd_kernel.mean()
    even_mean = even_kernel.mean()
    odd_kernel = odd_kernel - odd_mean
    even_kernel = even_kernel - even_mean

    odd_square_sum = np.sum(odd_kernel ** 2)
    even_square_sum = np.sum(even_kernel ** 2)
    l2_sum_odd = odd_square_sum / (17.0 * 17.0)
    l2_sum_even = even_square_sum / (17.0 * 17.0)

    odd_kernel = odd_kernel / l2_sum_odd
    even_kernel = even_kernel / l2_sum_even
    kernel = list()
    kernel.append(odd_kernel)
    kernel.append(even_kernel)
    return kernel


def computeVpScore(filePath):
    # Mat image_origin = imread(filePath);
    # Mat img_gray, img_float;
    # cvtColor(image_origin, img_gray, CV_RGB2GRAY);
    # img_gray.convertTo(img_float, CV_32F);
    image_origin = cv2.imread(filePath)
    img_gray = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
    img_float = np.float_(img_gray)
    # print(image_origin.shape)
    origin_row, origin_col, origin_dep = image_origin.shape
    n_theta = 36
    width = 128
    scale_factor = width / origin_col

    gray_row, gray_col = img_gray.shape

    # Mat image(img_gray.rows * scale_factor, width, CV_32F);
    image = np.zeros((ceil(gray_row * scale_factor), width), np.float32)
    # resize(img_float, image, image.size());
    image = cv2.resize(img_float, image.shape, interpolation=cv2.INTER_AREA)
    # int m = image.rows, n = image.cols;
    m, n = image.shape

    # float scores[85][128];
    scores = np.zeros((m, n), np.float32)
    # float ***gabors = new float**[m];
    gabors = np.zeros((m, n, n_theta), np.float32)
    # for_each(gabors, gabors + m, [n](float** &x) {x = new float*[n]; });
    # for_each(gabors, gabors + m, [n, n_theta](float** x) {for_each(x, x + n, [&, n_theta](float* &y) { y = new float[n_theta]; }); });

    # Mat oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);
    # oddfiltered=np.zeros((m,n),np.float32)
    # evenfiltered=np.zeros((m,n),np.float32)

    for t in range(n_theta):
        oddfiltered = cv2.filter2D(image, -1, oddKernels[t])
        evenfiltered = cv2.filter2D(image, -1, evenKernels[t])
        for i in range(m):
            for j in range(n):
                gabors[i][j][t] = np.power(oddfiltered[i][j], 2.0) + np.power(evenfiltered[i][j], 2.0)
    ######## why hard coded
    # directions = np.zeros((85, 128), np.uint8)
    directions = np.zeros((m, n), np.uint8)
    for i in range(m):
        for j in range(n):
            directions[i][j] = np.max(gabors[i][j]) - gabors[i][j][0]

    thresh = 2.0 * 180 / n_theta
    # r=(m+n)/7
    # r_dia=np.sqrt(m*m+n*n)
    gamma = 0
    c = 0
    for i in range(m):
        for j in range(n):
            tempScore = 0
            for i1 in range(i + 1, m):
                for j1 in range(n):
                    c = directions[i1][j1] / n_theta * 180.0
                    gamma = arcMatrix[i1 - 1][j1 - j + 200]["arc"]
                    if (np.abs(c - gamma) < thresh):
                        tempScore += 1
            scores[i][j] = tempScore

    score_min, score_max, p_min, p_max = cv2.minMaxLoc(scores)
    # scale=score_max/255.0
    x = p_max.x / scale_factor
    y = p_max.y / scale_factor
    # 	cv::circle(image_origin, cvPoint(p_max.x / scale_factor, p_max.y / scale_factor), 10, Scalar(0), 5, 8, 0);
    cv2.circle(image_origin, (x, y), 10, (0, 0, 0), 5, 8, 0)
    return image_origin


def visit(directory_in_str):
    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename:
            # input_path = directory_in_str + "/" + filename
            input_path = directory_in_str + "\\" + filename
            print("Processing {}".format(input_path))
            start = time.time()
            result_mat = computeVpScore(input_path)
            end = time.time()
            print("Use time: {0:f} sec", (end - start))
            # output_path = directory_in_str + "/results" + filename
            output_path = directory_in_str + "\\results" + filename
            cv2.imwrite(output_path, result_mat)
    if not filename:
        print("No valid input file.")


for j in range(200):
    for i in range(400):
        tmp = np.sqrt(np.power(i - 200.0, 2.0) + np.power(j, 2.0))
        arcMatrix[j][i]["arc"] = acos((200.0 - float(i)) / tmp) / pi * 180.0
        arcMatrix[j][i]["d"] = tmp
n_theta = 36
for t in range(n_theta):
    theta = pi * t / n_theta
    kernel = myGarborKernel(17, 17/9.0, theta, 5.0)
    oddKernels.append(kernel[0])
    evenKernels.append(kernel[1])
visit("img")
