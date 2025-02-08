# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:56:17 2024

@author: cqsfdu
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

path0 = "G://newbeads/6umbeads_test/6um_for_con"
path1 = "G://newbeads/6um_for_con"

# path2 = "D://shanxi normal university/9um_for_con/"
# file1 = open("F://fudan2024_2/202410/cycle_network_fixing/newbeads_gradient3.txt", 'a')
tiffiles = os.listdir(path0)
edge_gradient1 = np.zeros(len(tiffiles))
edge_gradient2 = np.zeros(len(tiffiles))

# kernel_post = np.array([[0.5*np.sqrt(2),1,0.5*np.sqrt(2)],[1, -4-2*np.sqrt(2),1],[0.5*np.sqrt(2),1,0.5*np.sqrt(2)]])
sigma = 1
size = int(2*np.round(3*sigma))+1
x, y = np.meshgrid(np.arange(-size/2+1, size/2+1), np.arange(-size/2+1, size/2+1))
norm2 = np.power(x, 2) + np.power(y, 2)
sigma2, sigma4 = np.power(sigma, 2), np.power(sigma, 4)
kernelLoG = ((norm2 - (2.0 * sigma2)) / sigma4) * np.exp(- norm2 / (2.0 * sigma2))
# print(kernelLoG)
edge_growth = np.zeros(len(tiffiles))
for i in range(len(tiffiles)):
    filename = path0 + '/' + tiffiles[i]
    print(filename)
    if not os.path.isdir(filename): 
        img0 = cv2.imread(filename, 0)
        img1 = cv2.imread(path1 +'/'+ tiffiles[i], 0)
        # img2 = cv2.imread(path2 + tiffiles[i], 0)
        # img_width = img2.shape[1] // 2
        # img_height = img2.shape[0] // 2
        img_width = 32
        img_height = 32
        img0_worthy = img0[128-img_height:128+img_height,128-img_width:128+img_width]
        img1_worthy = img1[128-img_height:128+img_height,128-img_width:128+img_width]
        eg_0 = cv2.filter2D(img0_worthy, -1, kernel=kernelLoG)
        # eg_0 = cv2.Laplacian(img0_worthy,cv2.CV_64F)
        # img_sobel_x=cv2.Sobel(img0_worthy,cv2.CV_64F,1,0,ksize=3)
        # img_sobel_y=cv2.Sobel(img0_worthy,cv2.CV_64F,1,0,ksize=3)
        # sobel_img_x_abs=cv2.convertScaleAbs(img_sobel_x)
        # sobel_img_y_abs=cv2.convertScaleAbs(img_sobel_y)
        # eg_0=cv2.addWeighted(sobel_img_x_abs,0.5,sobel_img_y_abs,0.5,0)
        edge_gradient1[i] = np.sum(eg_0)/eg_0.size
        eg_1 = cv2.filter2D(img1_worthy, -1, kernel=kernelLoG)
        # eg_1 = cv2.Laplacian(img1_worthy,cv2.CV_64F)
        # img_sobel_x=cv2.Sobel(img1_worthy,cv2.CV_64F,1,0,ksize=3)
        # img_sobel_y=cv2.Sobel(img1_worthy,cv2.CV_64F,1,0,ksize=3)
        # sobel_img_x_abs=cv2.convertScaleAbs(img_sobel_x)
        # sobel_img_y_abs=cv2.convertScaleAbs(img_sobel_y)
        # eg_1=cv2.addWeighted(sobel_img_x_abs,0.5,sobel_img_y_abs,0.5,0)
        edge_gradient2[i] = np.sum(eg_1)/eg_1.size
        # file1.write(tiffiles[i] + ' ' + str(edge_gradient1[i]) + ' ' + str(edge_gradient2[i]) + '\n')
        if(edge_gradient1[i] > edge_gradient2[i]):
            edge_growth[i] = 1
# np.savetxt('F://fudan2024_2/202410/cycle_network_fixing/newbeads_gradient.txt',edge_gradient1)
# np.savetxt('F://fudan2024_2/202410/cycle_network_fixing/newbeadsfix_gradient.txt',edge_gradient2)
# file1.close()
print(np.sum(edge_growth)/len(tiffiles))
plt.hist(edge_gradient1,bins=200)
# plt.hist((edge_gradient1-edge_gradient2),bins=200)
plt.xlim([-2.0,42.0])
plt.grid()
plt.title('edge gradient conting ori', fontsize=16)
plt.xlabel('Value (a.u.)')
plt.ylabel('Frequency')
plt.show()