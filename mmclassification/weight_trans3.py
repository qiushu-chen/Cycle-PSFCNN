# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:42:52 2023

@author: Lenovo
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
from mmcv.runner import get_dist_info, init_dist

from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)

cfg = Config.fromfile('./configs/resnest/resnest50_mydataset.py')
cfg.work_dir = './work_dirs/resnest_psfcycle_0921'
cfg.save_dir = './work_dirs/resnest_psfcycle_0930'
model = build_classifier(cfg.model)
#print(model)
model_weight_path = './work_dirs/resnest_psfcycle_0921/epoch_150.pth'
pretrained_dict1 = torch.load(model_weight_path)
new_state_dict = {}
new_state_dict['meta'] = pretrained_dict1['meta']
new_state_dict['meta']['epoch'] = 0
new_state_dict['meta']['iter'] = 0
print(new_state_dict['meta'])
new_state_dict['state_dict'] = {}

for k,v in model.state_dict().items():
    print(k)
    new_state_dict['state_dict'][k] = pretrained_dict1['state_dict'][k]
    # index_weight_ori = k.split('.')
    # if index_weight_ori[1]=='features':
    #     new_state_dict['state_dict'][k] = pretrained_dict1['state_dict'][k]
    # else:
    #     print(k)
    #     new_state_dict['state_dict'][k] = model.state_dict()[k]
torch.save(new_state_dict, "./work_dirs/resnest_psfcycle_0930/epoch_0_fixing.pth")