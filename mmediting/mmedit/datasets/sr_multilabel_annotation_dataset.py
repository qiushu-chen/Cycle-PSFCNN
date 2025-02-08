# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:18:22 2024

@author: cqsfdu
"""

import os.path as osp
import numpy as np
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

@DATASETS.register_module()
class SRMultiAnnotationDataset(BaseSRDataset):
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 ann_file,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_info = line.split(' ')
                gt_name = line_info[0]
                psf_label0 = line_info[1]
                psf_float_split = psf_label0.split(',')
                psf_label = np.zeros(len(psf_float_split))
                for i in range(len(psf_float_split)):
                    psf_label[i] = np.array(psf_float_split[i], dtype=np.float32)
                basename, ext = osp.splitext(osp.basename(gt_name))
                lq_name = gt_name
                #lq_name = f'{self.filename_tmpl.format(basename)}{ext}'
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.lq_folder, lq_name),
                        gt_path=osp.join(self.gt_folder, gt_name),
                        gt_label = psf_label))
        return data_infos