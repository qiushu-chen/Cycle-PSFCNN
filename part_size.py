# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:49:15 2024

@author: cqsfdu
"""

import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

# img_bkg_ori = cv2.imread("F://fudan2024_1/202406/newset/cal_image_000001.tif",0)
# img_bkg = img_bkg_ori[876:1132, 320:576]

tif_path = "G://newbeads/5umbeads_fixed1/5um_run3/"
tif_path1 = "G://newbeads/5um_for_con/"
# tif_path = "G://newbeads/6umbeads_test/6um_for_con/"
# tif_path1 = "G://newbeads/6um_for_con/"
tifdatanames = os.listdir(tif_path)
num_partles = np.zeros(len(tifdatanames))
area_partles = []
dia_partles = []
dia_partles2 = []
dia_error = []
dia_diff = []
num_counter = 0
num_particles = 0
sigma = 1.0
size = int(2*np.round(3*sigma))+1
x, y = np.meshgrid(np.arange(-size/2+1, size/2+1), np.arange(-size/2+1, size/2+1))
norm2 = np.power(x, 2) + np.power(y, 2)
sigma2, sigma4 = np.power(sigma, 2), np.power(sigma, 4)
kernelLoG = ((norm2 - (2.0 * sigma2)) / sigma4) * np.exp(- norm2 / (2.0 * sigma2))
edge_gradient = []
cont_fixed = 0
for tifdataname in tifdatanames:
    # print(tifdataname)
    tif1 = cv2.imread(tif_path + tifdataname, 0)
    tif2 = cv2.imread(tif_path1 + tifdataname, 0)
    # tif_diff = img_bkg.astype(np.int16) - tif1.astype(np.int16) 
    # tif_diff_final = np.uint8(tif_diff - np.min(tif_diff))
    # print(tif_diff_final)
    # thresh, result_binarization = cv2.threshold(tif_diff_final,20,255,cv2.THRESH_BINARY) 
    th3 = cv2.adaptiveThreshold(tif1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,49,5)
    ret, thresh1u = cv2.threshold(th3,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th4 = cv2.adaptiveThreshold(tif2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,49,5)
    ret, thresh2u = cv2.threshold(th4,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel_dil1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # drop_imd = cv2.morphologyEx(result_binarization, cv2.MORPH_CLOSE, kernel_dil1)
    drop_imd = cv2.morphologyEx(thresh1u, cv2.MORPH_CLOSE, kernel_dil1)
    drop_imd1 = cv2.morphologyEx(thresh2u, cv2.MORPH_CLOSE, kernel_dil1)
    #kernel_dil2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #drop_imd = cv2.dilate(drop_imd, kernel_dil2)
    # cv2.imwrite(tif_path1 + tifdataname, drop_imd)
    contours, hierarchy = cv2.findContours(drop_imd,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours1, hierarchy = cv2.findContours(drop_imd1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    area_image = []
    dia_image = []
    # dia_error = []
    dia_image2 = []
    # dia_image_diff = []
    pi_inv = 1/np.pi
    pi_rt_inv = 1/np.sqrt(np.pi)
    bounding_id = np.zeros((len(contours),4))
    
    seq_img_cont = 0
    for cont in contours:
        peri_cont = cv2.arcLength(cont, True)
        area_cont = cv2.contourArea(cont)
        dia = (np.sqrt(area_cont*pi_inv))*0.586*2.0
        # dia = (np.sqrt(area_cont*pi_inv)-1)*0.586*2.0
        # print(dia)
        circul = 2*np.sqrt(np.pi*area_cont )/ peri_cont
        if (circul > 0.1 and area_cont>20 and area_cont<1000):
            # print(tifdataname, dia)
            area_image.append(area_cont)
            dia_partles.append(dia)
            error = (np.sqrt(area_cont)-np.sqrt(area_cont-peri_cont))*2*0.586*pi_rt_inv
            # print(tifdataname,error)
            if area_cont>peri_cont:
                dia_error.append(error)
                bounding_id[seq_img_cont, :] = cv2.boundingRect(cont)
                seq_img_cont += 1
            dia_image.append(dia)
            num_partles[num_counter] += 1
    for i in range(bounding_id.shape[0]):
        boundid = bounding_id[i]
        if np.max(boundid) == 0:
            continue
        #print(boundid[2],boundid[3])
        x_temp = int(boundid[0]-5)
        y_temp = int(boundid[1]-5)
        w_temp = int(boundid[2]+10)
        h_temp = int(boundid[3]+10)
        # print(x_temp,y_temp)
        if x_temp<0:
            w_temp = w_temp + x_temp
            x_temp = 0
        if y_temp<0:
            h_temp = h_temp + y_temp
            y_temp = 0
        if (x_temp+w_temp)>255:
            w_temp = 255-x_temp
        if (y_temp+h_temp)>255:
            h_temp = 255-y_temp
        img_temp = tif1[y_temp:y_temp+h_temp,x_temp:x_temp+w_temp]
        # print(x_temp, img_gray.shape)
        eg_0 = cv2.filter2D(img_temp, -1, kernel=kernelLoG)
        edge_grad = np.sum(eg_0)/eg_0.size
        if (edge_grad < 2.5489):
            cont_fixed += 1
        # print(np.min(eg_0),np.max(eg_0),edge_grad)
        # if edge_grad<0.1:
        #     cv2.imshow('contours',img_temp)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        edge_gradient.append(edge_grad)
        num_particles += 1
    for cont2 in contours1:
        peri_cont = cv2.arcLength(cont2, True)
        area_cont = cv2.contourArea(cont2)
        # dia = (np.sqrt(area_cont*pi_inv)-1.75)*0.725*2.0
        dia = np.sqrt(area_cont*pi_inv)*0.586*2.0
        circul = 2*np.sqrt(np.pi*area_cont )/ peri_cont
        if (circul > 0.1 and area_cont>20 and area_cont<1000):
            dia_partles2.append(dia)
            dia_image2.append(dia)
    if (len(dia_image)==len(dia_image2) and len(dia_image)!=0):
        print(tifdataname, len(dia_image))
        for k in range(len(dia_image)):
            dia_diff.append(dia_image2[k]-dia_image[k])
            
    area_partles.append(area_image)
    # dia_partles.append(dia_image)
    num_counter += 1
#print(num_partles)
#print(num_partles.sum())
dia_array = np.array(dia_partles)
# print(dia_error)
dia_error = np.array(dia_error)
edge_gradient = np.array(edge_gradient)
# print(np.mean(dia_array))
# print(np.mean(dia_error),np.var(dia_error))
# print(np.min(edge_gradient), cont_fixed, num_particles)
# dia_array2 = np.array(dia_partles2)
# print(np.mean(dia_array2))
dia_differ = np.array(dia_diff)
print(np.mean(dia_diff))
# plt.hist(edge_gradient,bins=200)
# plt.hist(dia_array,bins=250)
plt.hist(dia_diff,bins=200)
# plt.xlim([3.0,14.0])
plt.xlim([-2.0,4.0])
plt.grid()
# plt.title('histogram of diameters diff of 6μm con images')
plt.title('5um con mode differ', fontsize=14)
# plt.xlabel('Value (a.u.)', fontsize=12)
plt.xlabel('Value (μm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()