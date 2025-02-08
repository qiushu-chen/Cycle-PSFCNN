import os
import shutil

file1 = open("./meta/lxaset2blur_val1.txt",'r')
path0 = "./val/"
path1 = "./val_rl/"
path2 = "./val_gt/"
path0_dst = "./val1/"
path1_dst = "./val1_rl/"
path2_dst = "./val1_gt/"
contents = file1.readlines()
file1.close()
for content in contents:
    filesplit = content.split(' ')
    filename = filesplit[0]
    shutil.move(path0+filename, path0_dst+filename)
    shutil.move(path1+filename, path1_dst+filename)
    shutil.move(path2+filename, path2_dst+filename)