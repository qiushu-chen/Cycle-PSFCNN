# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:10:02 2024

@author: cqsfdu
"""

import numpy as np
import cv2
import os

# path0 = "D://shanxi normal university/10um for PS&MO/"
path0 = "F://fudan2024_2/202412/to_cqs/4K-5_run/"
path1 = "G://newbeads/5um_for_con1/"

# img0 = cv2.imread("G:\newbeads\6um 20241009 for Con", 0)
# img1 = cv2.imread("D://FlowCam_Test_Report/Henlius_repeatablity_Thermo_Duke_0.5ml_20191025_3_deselected/Henlius_repeatablity_Thermo_Duke_0.5ml_20191025_3_deselected_000001.tif", 1)
img2 = cv2.imread("F://fudan2024_1/202406/newset/cal_image_000001.tif", 1)
# retval, img_label = cv2.threshold(img0, 100, 255, cv2.THRESH_BINARY)
# path2 = "D://FlowCam_Test_Report/Henlius_20190808/HLX13F4/"
# cv2.imshow('origin',img_label)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def edge_smmothing(col_index, row_index, height, width, img_target):
    if col_index > 3:
        #print(np.mean(np.int16(img_target[col_index-3, row_index:row_index+width])-np.int16(img_target[col_index+1, row_index:row_index+width])))
        img_target[col_index-2, row_index:row_index+width] = np.uint8(0.8*img_target[col_index-3, row_index:row_index+width] 
                                                                      + 0.2*img_target[col_index+2, row_index:row_index+width])
        img_target[col_index-1, row_index:row_index+width] = np.uint8(0.6*img_target[col_index-3, row_index:row_index+width] 
                                                                      + 0.4*img_target[col_index+2, row_index:row_index+width])
        img_target[col_index, row_index:row_index+width] = np.uint8(0.4*img_target[col_index-3, row_index:row_index+width] 
                                                                      + 0.6*img_target[col_index+2, row_index:row_index+width])
        img_target[col_index+1, row_index:row_index+width] = np.uint8(0.2*img_target[col_index-3, row_index:row_index+width] 
                                                                    + 0.8*img_target[col_index+2, row_index:row_index+width])
    if row_index > 3:
        img_target[col_index:col_index+height, row_index-2] = np.uint8(0.8*img_target[col_index:col_index+height, row_index-3] 
                                                                      + 0.2*img_target[col_index:col_index+height, row_index+2])
        img_target[col_index:col_index+height, row_index-1] = np.uint8(0.6*img_target[col_index:col_index+height, row_index-3] 
                                                                      + 0.4*img_target[col_index:col_index+height, row_index+2])
        img_target[col_index:col_index+height, row_index] = np.uint8(0.4*img_target[col_index:col_index+height, row_index-3] 
                                                                      + 0.6*img_target[col_index:col_index+height, row_index+2])
        img_target[col_index:col_index+height, row_index+1] = np.uint8(0.2*img_target[col_index:col_index+height, row_index-3] 
                                                                    + 0.8*img_target[col_index:col_index+height, row_index+2])
    if (col_index + height) < 254:
        img_target[col_index+height+1, row_index:row_index+width] = np.uint8(0.8*img_target[col_index+height+2, row_index:row_index+width] 
                                                                             + 0.2*img_target[col_index+height-3, row_index:row_index+width])
        img_target[col_index+height, row_index:row_index+width] = np.uint8(0.6*img_target[col_index+height+2, row_index:row_index+width] 
                                                                             + 0.4*img_target[col_index+height-3, row_index:row_index+width])
        img_target[col_index+height-1, row_index:row_index+width] = np.uint8(0.4*img_target[col_index+height+2, row_index:row_index+width] 
                                                                             + 0.6*img_target[col_index+height-3, row_index:row_index+width])
        img_target[col_index+height-2, row_index:row_index+width] = np.uint8(0.2*img_target[col_index+height+2, row_index:row_index+width] 
                                                                           + 0.8*img_target[col_index+height-3, row_index:row_index+width])
    if (row_index + width) < 254:
        img_target[col_index:col_index+height, row_index+width+1] = np.uint8(0.8*img_target[col_index:col_index+height, row_index+width+2] 
                                                                             + 0.2*img_target[col_index:col_index+height, row_index+width-3])
        img_target[col_index:col_index+height, row_index+width] = np.uint8(0.6*img_target[col_index:col_index+height, row_index+width+2] 
                                                                             + 0.4*img_target[col_index:col_index+height, row_index+width-3])
        img_target[col_index:col_index+height, row_index+width-1] = np.uint8(0.4*img_target[col_index:col_index+height, row_index+width+2] 
                                                                             + 0.6*img_target[col_index:col_index+height, row_index+width-3])
        img_target[col_index:col_index+height, row_index+width-2] = np.uint8(0.2*img_target[col_index:col_index+height, row_index+width+2] 
                                                                           + 0.8*img_target[col_index:col_index+height, row_index+width-3])        
        

def partle_pre_bkg(img_bkg, img, start_point0, start_point1):
    # start_point0 = np.int(np.floor((128 - img_size0)/2))
    # start_point1 = np.int(np.floor((128 - img_size1)/2))
    img_size0 = img.shape[0]
    img_size1 = img.shape[1]
    img_new = img_bkg
    #print(img_new.shape)
    img_new[start_point0:start_point0+img_size0, start_point1:start_point1+img_size1]=img
    # print(start_point0)
    edge_smmothing(start_point0, start_point1, img_size0, img_size1, img_new)  
    return img_new

bkg_ori = np.int16(img2[876:1132, 320:576])
new_imgbk = np.uint8(np.copy(bkg_ori) + np.int16(np.random.randn(256,256,3)*1))
new_imgbko = np.int16(new_imgbk)
num_part = 0
for i in range(5):
    tiffilename = path0 + "4K-5 run1_" + str(i+17) + '.tif'
    # print(tiffilename)
    img0 = cv2.imread(tiffilename, 0)
    # print(img0)
    img1 = cv2.imread(tiffilename, 1)
    retval, img_label = cv2.threshold(img0, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1,contours,-1,(120,0,0),2)
# cv2.imshow('origin',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    for cont in contours:
    # print(cont.shape)
        cont0 = cont[:,0,:]
        start_y = np.min(cont0[:,1])+2
        start_x = np.min(cont0[:,0])+2
        end_y = np.max(cont0[:,1])-1
        end_x = np.max(cont0[:,0])-1
        if (end_y>start_y and end_x>start_x):
            img_tif = img1[start_y:end_y,start_x:end_x]
        else:
            continue
    # print(start_y,end_y,start_x,end_x,np.max(img_tif),np.min(img_tif))
        print(np.mean(img_tif[0,:,:]),np.mean(img_tif[:,0,:]))
        height_start = np.int(128-(end_y-start_y)/2)
        width_start = np.int(128-(end_x-start_x)/2)
    # img1_p = img_tif[height_start[j]:height_end[j], width_start[j]:width_end[j]]
    # mask = np.zeros((end_y-start_y,end_x-start_x,3))
    # mask_buf = (np.mean(img_tif[0,:,:])-np.mean(img_tif[-1,:,:]))/mask.shape[0]
    # mask_buf1 = (np.mean(img_tif[:,0,:])-np.mean(img_tif[:,-1,:]))/mask.shape[1]
    # for k in range(mask.shape[0]):
    #     mask[k,:,:] = np.uint8(mask_buf*k+5)
    # for k1 in range(mask.shape[1]):
    #     mask[:,k1,:] += np.uint8(mask_buf1*k1+2)
    # # print(mask)
    # img_tif = img_tif - mask
        img_buf3 = np.int16((np.mean(img_tif[0,:,:]) + np.mean(img_tif[:,0,:]))/2-160)
        print(img_buf3)
        new_imgbko3 = np.uint8(new_imgbko + img_buf3)
        if len(cont0)>=40 and cv2.contourArea(cont0)>100:
            new_imgbko3 = partle_pre_bkg(new_imgbko3, img_tif, height_start, width_start)
            new_imgbko3 = cv2.convertScaleAbs(new_imgbko3, alpha=1.0, beta=-img_buf3)
            num_idx = str(num_part).zfill(6)
            cv2.imwrite(path1+num_idx+'.tif',new_imgbko3)
            num_part += 1