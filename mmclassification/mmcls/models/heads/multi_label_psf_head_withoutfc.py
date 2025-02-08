# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:27:35 2024

@author: cqsfdu
"""

import torch
import torch.nn as nn

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead

@HEADS.register_module()
class MultiLabelL2PSFHead_WithoutFC(MultiLabelClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='PSFL2Loss',
                     reduction='none',
                     loss_weight=1.0),    
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelL2PSFHead_WithoutFC, self).__init__(
            loss=loss, init_cfg=init_cfg)
        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        # self.fc = nn.Linear(self.in_channels, self.num_classes)
        
    # def pre_logits(self, x):
    #     if isinstance(x, tuple):
    #         x = x[-1]
    #     return x

    def forward_train(self, x, gt_label, **kwargs):
        # x = self.pre_logits(x)
        gt_label = gt_label.type_as(x)
        # cls_score = self.fc(x)
        cls_score = x
        losses = self.loss(cls_score, gt_label, **kwargs)
        return cls_score, losses
    
    def simple_test(self, x, sigmoid=False, post_process=False):
        # x = self.pre_logits(x)
        # cls_score = self.fc(x)
        cls_score = x
        if sigmoid:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = cls_score
        if post_process:
            return self.post_process(pred)
        else:
            return pred