# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:51:54 2024

@author: Lenovo
"""

import os
import shutil
import numpy as np

f1 = open('./lxaset2blur.txt','r')
f2 = open('./lxaset2blur_train.txt','a')
f3 = open('./lxaset2blur_val.txt','a')
contents = f1.readlines()
f1.close()
path0 = "./lxaset2blur_test/lxaset2blur/"
path0_dst = "./lxaset2blur_test/val/"
path1 = "./lxaset2blur_test/lxaset2blur_gt/"
path1_dst = "./lxaset2blur_test/val_gt/"
path2 = "./lxaset2blur_test/lxaset2blur_rl/"
path2_dst = "./lxaset2blur_test/val_rl/"

for content in contents:
    filename_read = content.split(' ')
    filename = filename_read[0]
    rand_judg = np.random.rand()
    if rand_judg <= 0.1:
        shutil.move(path0+filename, path0_dst+filename)
        shutil.move(path1+filename, path1_dst+filename)
        shutil.move(path2+filename, path2_dst+filename)
        f3.write(content)
    else:
        f2.write(content)
f2.close()
f3.close()