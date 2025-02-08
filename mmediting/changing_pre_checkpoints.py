# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:41:40 2023

@author: gywbc
"""

import torch, torchvision
import mmedit
import os
import os.path as osp

from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model

import mmcv
from mmcv.runner import init_dist, set_random_seed
from mmcv import Config

cfg = Config.fromfile('./configs/restorers/real_esrgan/cycle_realesrgan_psfnet.py')
cfg.model.generator.type = 'RRDABNet'
cfg.data.train_dataloader.samples_per_gpu = 1
if cfg.evaluation.get('gpu_collect', None):
    cfg.evaluation.pop('gpu_collect')
#cfg.load_from = 'F://mmlab/mmediting/checkpoint/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth'
cfg.work_dir = './tutorial_exps/cycle_esrgan_1025'
cfg.save_dir = './tutorial_exps/cycle_esrgan_1025'

cfg.seed = 10
set_random_seed(0, deterministic=True)
cfg.gpus = 1

datasets = [build_dataset(cfg.data.train)]
# 构建模型
model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# print(model)
# 创建工作路径
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# 额外信息
meta = dict()
if cfg.get('exp_name', None) is None:
    cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
meta['exp_name'] = cfg.exp_name
meta['mmedit Version'] = mmedit.__version__
meta['seed'] = 0

#print(model.state_dict().items())

model_weight_path = "./tutorial_exps/real_esrgan_0919_comp/iter_24000_1002.pth"
model_weight_path1 = "../mmclassification/work_dirs/resnest_psfcycle_1024/epoch_30.pth"
model_weight_path2 = "./tutorial_exps/cycle_esrgan_0914/cycle_0930_fix2.pth"
pre_state_dict = torch.load(model_weight_path)
pre_state_dict1 = torch.load(model_weight_path1)
#print(pre_state_dict['state_dict']['perceptual_loss.vgg.vgg_layers.0.weight'].shape)
pre_state_dict2 = torch.load(model_weight_path2)
# print(pre_state_dict2['params'].keys())
# for k, v in pre_state_dict['state_dict'].items():
#     print(k)
#print(pre_state_dict['state_dict'])
new_state_dict = {}
new_state_dict['meta'] = pre_state_dict['meta']
new_state_dict['meta']['epoch']=0
new_state_dict['meta']['iter']=0
new_state_dict['meta']['hook_msgs']={'last_ckpt': None}
print(new_state_dict['meta'])
new_state_dict['state_dict'] = {}

#print(pre_state_dict['state_dict'].keys())
#print(pre_state_dict['state_dict']['generator_ema.conv_last.bias'])
for k, v in model.state_dict().items():
    k_mean = k.split('.')
    # print(k_mean[0])
    
    if(k_mean[0]=='psfcnn'):
        # print(k)
        if(k_mean[1] == 'fc1' or k_mean[1] == 'fc2' or k_mean[1] == 'fc3'):
            keq = k.replace(k_mean[0], 'head')
            new_state_dict['state_dict'][k] = pre_state_dict1['state_dict'][keq]
        else:
            keq = k.replace(k_mean[0], 'backbone')
            new_state_dict['state_dict'][k] = pre_state_dict1['state_dict'][keq]
        # print(keq)
        # print(pre_state_dict1['state_dict'][keq])
    # keq = k.replace(k_mean[0], 'generator')
    elif k in pre_state_dict['state_dict'].keys():
        # print(k)
        new_state_dict['state_dict'][k] = pre_state_dict['state_dict'][k]
    # elif k_mean[0]=='task_classifier_backbone':
    #     keq1 = k.replace(k_mean[0], 'backbone')
    #     new_state_dict['state_dict'][k] = pre_state_dict1['state_dict'][keq1]
    # elif k_mean[0]=='task_classifier_head':
    #     keq2 = k.replace(k_mean[0], 'head')
    #     new_state_dict['state_dict'][k] = pre_state_dict1['state_dict'][keq2]
    # elif (k_mean[0]=='generator_ema' and (keq in pre_state_dict['state_dict'].keys())):
    #     new_state_dict['state_dict'][k] = pre_state_dict['state_dict'][keq]
    # elif k_mean[0]!='generator_ema' :
    #     print('rest: ', k)
    #     new_state_dict['state_dict'][k] = model.state_dict()[k]
    # elif krest in pre_state_dict2['params'].keys():
        #print(k)
    #     new_state_dict['state_dict'][k] = pre_state_dict2['params'][krest]
    # elif k_adj == 'conv_body_att':
        #print(k)
    #     kresttemp = krest.replace(k_adj, 'conv_body.1')
     #    new_state_dict['state_dict'][k] = pre_state_dict2['params'][kresttemp]
    else:
        print(k)
        new_state_dict['state_dict'][k] = model.state_dict()[k]
        #keq = k.replace(k_mean[0], 'generator')
        #rint(k, model.state_dict()[k])
        # new_state_dict['state_dict'][k] = new_state_dict['state_dict'][keq]
torch.save(new_state_dict, "./tutorial_exps/cycle_esrgan_1025/cycle_1025_fix.pth")