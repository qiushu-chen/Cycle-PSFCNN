# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:53:19 2024

@author: cqsfdu
"""

import numpy as np
import cv2
import os
import random
import mmcv 
from mmcls.datasets import PIPELINES

rand_increasing_policies = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    ]
aug_cfg = dict(
    type='RandAugment',
    policies = rand_increasing_policies,
    num_policies=3,
    total_level=10,
    magnitude_level=9,
    magnitude_std=0.5,
    )
aug = PIPELINES.build(aug_cfg)

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

img_bkg_ori = cv2.imread("./newset/cal_image_000001.tif",1)
bkg_ori = np.int16(img_bkg_ori[876:1132, 320:576])
new_imgbk = np.uint8(np.copy(bkg_ori) + np.int16(np.random.randn(256,256,3)*1))

path0 = "../../fudan2024_2/202408/glass/test/glassseg_new/Image"
path1 = "../../fudan2024_2/202408/glass/out/glassnew/Image"
path2 = "../../fudan2024_2/202408/glass/test_gt/glassseg50/Image"
path3 = "../../fudan2024_2/202408/glass/out/glassnew_gt/Image"

img_start = 457
img_end = img_start + 50
img_mid = 482
width_start = np.array([150]).astype(np.int)
height_start = np.array([5]).astype(np.int)
width_end = np.array([210]).astype(np.int)
height_end = np.array([65]).astype(np.int)
for i in range(25): 
    img_idx1 = img_mid-i-1
    img_idx2 = img_mid+i+1
    if img_idx1 >= img_start:
        img_name1 = path0 + str(img_idx1) + '_33.tif'
        imggt_name1 = path2 + str(img_idx1) + '_33.tif'
        img1 = cv2.imread(img_name1,1)
        imggt1 = cv2.imread(imggt_name1,1)
        new_imgbko = np.int16(new_imgbk)
        for j in range(len(width_start)):
            # print(img_idx1)
            img1_p = img1[height_start[j]:height_end[j], width_start[j]:width_end[j]]
            imggt1_p = imggt1[height_start[j]:height_end[j], width_start[j]:width_end[j]]
            mask = np.zeros((height_end[j]-height_start[j],width_end[j]-width_start[j],3))
            mask_buf = (np.mean(img1_p[0,:,:])-np.mean(img1_p[-1,:,:]))/mask.shape[0]
            mask_buf1 = (np.mean(img1_p[:,0,:])-np.mean(img1_p[:,-1,:]))/mask.shape[1]
            for k in range(mask.shape[0]):
                mask[k,:,:] = np.uint8(mask_buf*k+10)
            for k1 in range(mask.shape[1]):
                mask[:,k1,:] += np.uint8(mask_buf1*k1)
            img1_p = img1_p + mask
            imggt1_p = imggt1_p + mask
            img_buf3 = np.int16((np.mean(img1_p[0,:,:]) + np.mean(img1_p[:,-1,:]))/2-160)
            # img_buf3 = np.int16(np.mean(img1_p[0:-50,0,:])-160)
            if j==0:
                new_imgbko3 = np.uint8(new_imgbko + img_buf3)
            else:
                new_imgbko3 = np.uint8(np.int16(new_imgbko3) + img_buf3)
            new_imgbko3 = partle_pre_bkg(new_imgbko3, img1_p, height_start[j], width_start[j])
            new_imgbko3 = cv2.convertScaleAbs(new_imgbko3, alpha=1.0, beta=-img_buf3)
            # cv2.imshow('newing_shown',new_imgbko3)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img_buf4 = np.int16((np.mean(imggt1_p[0,:,:]) + np.mean(imggt1_p[:,-1,:]))/2-160)
            # img_buf4 = np.int16(np.mean(imggt1_p[0:-50,0,:])-160)
            if j==0:
                new_imgbko4 = np.uint8(new_imgbko + img_buf4)
            else:
                new_imgbko4 = np.uint8(np.int16(new_imgbko4) + img_buf4)
            new_imgbko4 = partle_pre_bkg(new_imgbko4, imggt1_p, height_start[j], width_start[j])
            new_imgbko4 = cv2.convertScaleAbs(new_imgbko4, alpha=1.0, beta=-img_buf4)
            # cv2.imshow('newing_shown',new_imgbko4)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        cv2.imwrite(path1 + str(img_idx1) + '_33.tif', new_imgbko3+10)
        cv2.imwrite(path3 + str(img_idx1) + '_33.tif', new_imgbko4+10)
    if img_idx2 <= img_end:
        img_name2 = path0 + str(img_idx2) + '_33.tif'
        imggt_name2 = path2 + str(img_idx2) + '_33.tif'
        img2 = cv2.imread(img_name2,1)
        imggt2 = cv2.imread(imggt_name2,1)
        new_imgbko = np.int16(new_imgbk)
        for j in range(len(width_start)):
            img2_p = img2[height_start[j]:height_end[j], width_start[j]:width_end[j]]
            imggt2_p = imggt2[height_start[j]:height_end[j], width_start[j]:width_end[j]]
            mask = np.zeros((height_end[j]-height_start[j],width_end[j]-width_start[j],3))
            mask_buf = (np.mean(img2_p[0,:,:])-np.mean(img2_p[-1,:,:]))/mask.shape[0]
            mask_buf1 = (np.mean(img2_p[:,0,:])-np.mean(img2_p[:,-1,:]))/mask.shape[1]
            for k in range(mask.shape[0]):
                mask[k,:,:] = np.uint8(mask_buf*k+10)
            for k1 in range(mask.shape[1]):
                mask[:,k1,:] += np.uint8(mask_buf1*k1)
            img2_p = img2_p + mask
            imggt2_p = imggt2_p + mask
            img_buf1 = np.int16((np.mean(img2_p[0,:,:]) + np.mean(img2_p[:,-1,:]))/2-160)
            # img_buf1 = np.int16(np.mean(img2_p[0:-50,0,:])-160)
            if j==0:
                new_imgbko1 = np.uint8(new_imgbko + img_buf1)
            else:
                new_imgbko1 = np.uint8(np.int16(new_imgbko1) + img_buf1)
            new_imgbko1 = partle_pre_bkg(new_imgbko1, img2_p, height_start[j], width_start[j])
            new_imgbko1 = cv2.convertScaleAbs(new_imgbko1, alpha=1.0, beta=-img_buf1)
            img_buf2 = np.int16((np.mean(imggt2_p[0,:,:]) + np.mean(imggt2_p[:,-1,:]))/2-160)
            # img_buf2 = np.int16(np.mean(imggt2_p[0:-50,0,:])-160)
            if j==0:
                new_imgbko2 = np.uint8(new_imgbko + img_buf2)
            else:
                new_imgbko2 = np.uint8(np.int16(new_imgbko2) + img_buf2)
            new_imgbko2 = partle_pre_bkg(new_imgbko2, imggt2_p, height_start[j], width_start[j])
            new_imgbko2 = cv2.convertScaleAbs(new_imgbko2, alpha=1.0, beta=-img_buf2)
        cv2.imwrite(path1 + str(img_idx2) + '_33.tif', new_imgbko1+10)
        cv2.imwrite(path3 + str(img_idx2) + '_33.tif', new_imgbko2+10)