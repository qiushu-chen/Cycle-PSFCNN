# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:16:25 2024

@author: Lenovo
"""

import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

tif_path = "F://fudan2024_2/202411/cont4/cont4_fix/group_fix/306/group1_Image306_23_3_2.tif"
img_gray = cv2.imread(tif_path,0)
print(img_gray.shape)
th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 49, 5)
ret, thresh1 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
denoised= cv2.fastNlMeansDenoising(thresh1)
kernel_dil1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
drop_imd = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel_dil1)
# drop_imd = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, denoised)
cv2.imshow('thresh',drop_imd)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, hierarchy = cv2.findContours(drop_imd,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
area_image = []
    # dia_image = []
cont_num2 = len(contours)
num_counter = 0
bounding_id = np.zeros((cont_num2,4))
pi_inv = 1/np.pi
for cont in contours:
    peri_cont = cv2.arcLength(cont, True)
    area_cont = cv2.contourArea(cont)
    dia = np.sqrt(area_cont*pi_inv)*0.65*2.0
    circul = 2*np.sqrt(np.pi*area_cont )/ peri_cont
    if (circul > 0.1 and area_cont>15 and area_cont<1000):
        area_image.append(area_cont)
        bounding_id[num_counter, :] = cv2.boundingRect(cont)
        num_counter += 1
        # dia_partles.append(dia)
        # num_partles[num_counter] += 1
print(area_image)
for i in range(bounding_id.shape[0]):
    boundid = bounding_id[i]
    if np.max(boundid) == 0:
        continue
    x_temp = int(boundid[0]-10)
    y_temp = int(boundid[1]-5)
    w_temp = int(boundid[2]+20)
    h_temp = int(boundid[3]+10)
    start_point = (x_temp,y_temp)
    end_point = (x_temp+w_temp, y_temp+h_temp)
    color = (0,255,0)
    cv2.rectangle(img_gray, start_point, end_point, color, 2)
    # dia_partles.append(dia_image)
# num_counter += 1
cv2.imshow('thresh',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("F://fudan2024_2/202411/cont4/cont4_fix/group1_fix/306/group1_Image306_23_3_2_c.tif",img_gray)