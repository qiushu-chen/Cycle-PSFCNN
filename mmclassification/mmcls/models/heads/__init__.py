# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .linear_extend_head import LinearExtendClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_label_psf_head import MultiLabelL2PSFHead
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .multi_label_psf_head_withoutfc import MultiLabelL2PSFHead_WithoutFC

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead', 'LinearExtendClsHead', 'MultiLabelL2PSFHead', 'MultiLabelL2PSFHead_WithoutFC'
]
