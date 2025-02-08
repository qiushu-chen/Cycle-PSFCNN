# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:42:52 2023

@author: cqsfdu
"""

# import deepdish as dd
import numpy as np
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)

cfg = Config.fromfile('./configs/resnest/resnest50_mydataset.py')
cfg.work_dir = './work_dirs/resnest_psfcycle_0914'
cfg.save_dir = './work_dirs/resnest_psfcycle_0914'
model = build_classifier(cfg.model)
model_weight_path = '../mmediting/tutorial_exps/cycle_esrgan_1025/iter_8000.pth'
pretrained_dict1 = torch.load(model_weight_path)
new_state_dict = {}
new_state_dict['meta'] = pretrained_dict1['meta']
new_state_dict['meta']['epoch'] = 0
new_state_dict['meta']['iter'] = 0
print(new_state_dict['meta'])
new_state_dict['state_dict'] = {}
for k,v in model.state_dict().items():
    k_mean = k.split('.')
    # if(k_mean[1] == 'fc1' or k_mean[1] == 'fc2' or k_mean[1] == 'fc3'):
    if(k_mean[0]=='head' or k_mean[0]=='backbone'):
        keq = k.replace(k_mean[0], 'psfcnn')    
        new_state_dict['state_dict'][k] = pretrained_dict1['state_dict'][keq]
    else:
        print(k)
        new_state_dict['state_dict'][k] = model.state_dict()[k]
torch.save(new_state_dict, "./work_dirs/resnest_psfcycle_0914/iter_8000.pth")