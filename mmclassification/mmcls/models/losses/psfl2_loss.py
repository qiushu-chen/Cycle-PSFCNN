# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:25:51 2024

@author: cqsfdu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def l2loss(outputs, labels):
    zero_or_one = 1.0 - labels[:,0]
    loss_flag =  torch.pow((outputs[:,0] - labels[:,0]),2).mean()
    outputs_w = outputs[:,1:]*torch.tensor([0.5,1.0,0.32,1.0,1.0,1.0]).cuda()
    labels_w = labels[:,1:]*torch.tensor([0.5,1.0,0.32,1.0,1.0,1.0]).cuda()
    loss_parameters = torch.pow((outputs_w - labels_w),2).mean(1)
    loss_defocus = torch.pow(torch.sqrt(torch.pow(2.0*outputs[:,1], 2))-torch.sqrt(torch.pow(2.0*labels[:,1],2)),2)
    # loss_defocus = torch.sqrt(loss_defocuso)
    # print(loss_defocus)
    loss = torch.mul(zero_or_one,loss_parameters).mean() + loss_flag + torch.mul(zero_or_one,loss_defocus).mean()
    return loss

@LOSSES.register_module()
class PSFL2Loss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='none'):
        super(PSFL2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cls_criterion = l2loss
       
    def forward(self,
                cls_score,
                label,
                weight=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label)
        return loss_cls