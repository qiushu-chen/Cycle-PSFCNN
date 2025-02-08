# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:56:21 2024

@author: Lenovo
"""

import numpy as np
import os

path0 = "G://newbeads/10um_beads/"
f1 = open("G://newbeads/10um.txt",'a')
temp_label = np.zeros(7)
tifdirnames = os.listdir(path0)
for tiffilename in tifdirnames:
    tifdir = path0 + tiffilename
    tifdirpaths = os.listdir(tifdir)
    for tifdirpath in tifdirpaths:
        str_2_write = tiffilename + '/' + tifdirpath + ' '
        # str_2_write = tifdirpath + ' '
        for i in range(6):
            str_2_write = str_2_write + str(temp_label[i]) + ','
        str_2_write = str_2_write +str(temp_label[6])+ '\n'
        f1.write(str_2_write)
# for tifdirname in tifdirnames:
#     tifdir = path0 + tifdirname
#     tiffilenames = os.listdir(tifdir)
#     for tiffilename in tiffilenames:
#         str_2_write = tifdirname + '/' + tiffilename + ' '
#         for i in range(6):
#             str_2_write = str_2_write + str(temp_label[i]) + ','
#         str_2_write = str_2_write +str(temp_label[6])+ '\n'
#         f1.write(str_2_write)
f1.close()