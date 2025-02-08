# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:56:38 2024

@author: cqsfdu
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
import ast
import pandas as pd

with open("G://newbeads/temp/fix_6umbeads_psmo.json", 'r') as f1:
    contents = f1.readlines()
f1.close()
with open("G://newbeads/6umpsmo.txt", 'r') as f2:
    contents_label = f2.readlines()
f2.close()
# f3 = open("F://fudan2024_2/202411/cont4/cont4_defocus.txt", 'a')

contents1 = ast.literal_eval(contents[0])
psf_perameter =np.array(contents1['class_scores'])
psf_defocus = psf_perameter[:,1]
print(len(psf_defocus))
# psf_defocus_fl = [[],[],[],[]]
psf_defocus_fl = []
for i in range(len(contents_label)):
    label_z_split = contents_label[i].split(' ')
    filename = label_z_split[0]
    label_z = label_z_split[1]
    defocus_split = label_z.split(',')
    # defocus = np.abs(np.float32(defocus_split[1]))
    defocus_reason = np.float32(psf_defocus[i])*10.0/0.951292
    psf_defocus_fl.append(defocus_reason)
    # f3.write(filename + ' ' + str(defocus_reason) + '\n')
# f3.close()
    # label_z = label_z_split[1]
    # defocus_split = label_z.split(',')
    # defocus = np.abs(np.float32(defocus_split[1]))
    # if defocus<0.98:
    #     psf_defocus_fl[0].append(psf_defocus[i])
    # elif 0.98 <= defocus <1.94:
    #     psf_defocus_fl[1].append(psf_defocus[i])
    # elif 1.94 <= defocus <2.88:
    #     psf_defocus_fl[2].append(psf_defocus[i])
    # else:
    #     psf_defocus_fl[3].append(psf_defocus[i])
# f3.close()
# psf_defocus_fl0 = np.abs(psf_defocus_fl[0]) *10.0/0.951292
plt.hist(np.abs(psf_defocus_fl),bins=100)
# plt.xlim([0.0,40.0])
plt.grid()
plt.title('Histogram of defocus of fixed 6 μm beads psmo')
plt.xlabel('defocus (μm)')
plt.ylabel('Frequency')
plt.show()