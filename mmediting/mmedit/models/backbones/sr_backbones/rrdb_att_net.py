# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmedit.models.common import (default_init_weights, make_layer,
                                  pixel_unshuffle)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from einops import rearrange

# class SALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SALayer, self).__init__()
#         self.sam = nn.Sequential(
#             nn.Conv2d(channel, channel//reduction, 3, 1, 1),
#             #nn.ReLU(inplace=True),
#             nn.GELU(),
#             nn.Conv2d(channel//reduction, channel, 3, 1, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         y = self.sam(x)
#         return x * y
    
    
class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv1(y))
        return x * y
# class FixedSALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(FixedSALayer, self).__init__()
#         self.conv3 = nn.Conv2d(
#             in_channels=channel // 2,
#             out_channels=channel,
#             kernel_size=1,
#             padding=0,
#             stride=1,
#             groups=1,
#             bias=True)
#         self.sam = nn.Sequential(
#             nn.Conv2d(channel, channel//reduction, 3, 1, 1),
#             #nn.ReLU(inplace=True),
#             nn.GELU(),
#             nn.Conv2d(channel//reduction, channel, 3, 1, 1),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x = self.conv3(x)
#         y = self.sam(x)
#         return x * y  

    
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             #nn.ReLU(inplace=True),
#             nn.GELU(),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        z = self.max_pool(x).view(b, c)
        z = self.fc2(z)
        y = self.sigmoid(y+z).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SimplifiedSELayer1(nn.Module):
    def __init__(self, channel):
        super(SimplifiedSELayer1, self).__init__()
        # self.sg = SimpleGate()
        self.fc = nn.Sequential(
            nn.BatchNorm2d(channel // 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=channel // 2,
                out_channels=channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True)
        )
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y = x1 * x2
        y = x * self.fc(y)
        return y
    
class SimplifiedSELayer2(nn.Module):
    def __init__(self, channel):
        super(SimplifiedSELayer2, self).__init__()
        # self.sg = SimpleGate()
        self.fc = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True)
        )
    def forward(self, x): 
        y = x * self.fc(x)
        return y
    
# class SimplifiedSELayer(nn.Module):
#     def __init__(self, channel):
#         super(SimplifiedSELayer, self).__init__()
#         # self.sg = SimpleGate()
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(
#                 in_channels=channel,
#                 out_channels=channel // 2,
#                 kernel_size=1,
#                 padding=0,
#                 stride=1,
#                 groups=1,
#                 bias=True), nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 in_channels=channel // 2,
#                 out_channels=channel,
#                 kernel_size=1,
#                 padding=0,
#                 stride=1,
#                 groups=1,
#                 bias=True), nn.Sigmoid()
#         )
#     def forward(self, x):
#         # x1, x2 = x.chunk(2, dim=1)
#         # y = x1 * x2
#         # y = y*self.fc(y)
#         y = x*self.fc(x)
#         return y
    
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels=64, growth_channels=32):
        super().__init__()
        #self.bias = bias
        for i in range(5):
            out_channels = mid_channels if i == 4 else growth_channels
            self.add_module(
                f'conv{i+1}',
                nn.Conv2d(mid_channels + i * growth_channels, out_channels, 3,
                          1, 1))
        #self.conv3_att = SELayer(growth_channels, 8)
        #self.conv3_sam = SALayer(growth_channels, 8)
        self.conv4_att = SELayer(growth_channels, 8)
        # self.conv4_sam = SALayer(growth_channels, 8)
        self.conv4_sam = SALayer(3)
        self.conv5_att = SELayer(mid_channels, 8)
        #self.conv5_sam = SALayer(mid_channels, 8)
        self.conv5_sam = SALayer(3)
        #self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.gelu = nn.GELU()
        
        self.init_weights()

    def init_weights(self):
        """Init weights for ResidualDenseBlock.

        Use smaller std for better stability and performance. We empirically
        use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
        Generative Adversarial Networks"
        """
        for i in range(5):
            default_init_weights(getattr(self, f'conv{i+1}'), 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x1 = self.gelu(self.conv1(x))
        x2 = self.gelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.gelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x3 = self.conv3_att(x3)
        # x3 = self.conv3_sam(x3)
        x4 = self.gelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = self.conv4_att(x4)
        x4 = self.conv4_sam(x4)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.conv5_att(x5)
        x5 = self.conv5_sam(x5)
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDAB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        mid_channels (int): Channel number of intermediate features.
        growth_channels (int): Channels for each growth.
    """

    def __init__(self, mid_channels, growth_channels=32):
        super().__init__()
        #self.bias = bias
        self.rdb1 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(mid_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(mid_channels, growth_channels)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@BACKBONES.register_module(force=True)
class RRDABNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data. # noqa: E501
    Currently, it supports [x1/x2/x4] upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Defaults: 23
        growth_channels (int): Channels for each growth. Default: 32.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
            Default: 4.
    """
    _supported_upscale_factors = [1, 2, 4]

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 growth_channels=32,
                 upscale_factor=4):
        super().__init__()
        if upscale_factor in self._supported_upscale_factors:
            #in_channels = in_channels * ((4 // upscale_factor)**2)
            in_channels = in_channels
        else:
            raise ValueError(f'Unsupported scale factor {upscale_factor}. '
                             f'Currently supported ones are '
                             f'{self._supported_upscale_factors}.')

        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDAB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        # layers = []
        # for i in range(3):
        #     layers.append(RRDAB(mid_channels=mid_channels,growth_channels=growth_channels))
        # for i in range(4):
        #     layers.append(RRDAB(1,mid_channels=mid_channels,growth_channels=growth_channels,bias=False))
        # for i in range(10):
        #     layers.append(RRDAB(1,mid_channels=mid_channels,growth_channels=growth_channels,bias=False))
        # for i in range(6):
        #     layers.append(RRDAB(1,mid_channels=mid_channels,growth_channels=growth_channels,bias=False))
        # self.body = nn.Sequential(*layers)
        # self.conv_body = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
        #                                SELayer(mid_channels,8))
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_body_att = SELayer(mid_channels, 8)
        # upsample
        #self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)

        #self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.upscale_factor in [1, 2]:
            #feat = pixel_unshuffle(x, scale=4 // self.upscale_factor)
            feat = x
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        body_featatt = self.conv_body_att(body_feat)
        feat = feat + body_featatt

        # upsample
        #feat = self.lrelu(
        feat = self.gelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            #self.conv_up1(F.interpolate(feat, scale_factor=1, mode='nearest')))
            #self.conv_up1(feat))
        #feat = self.lrelu(
        feat = self.gelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
            #self.conv_up2(F.interpolate(feat, scale_factor=1, mode='nearest')))
            #self.conv_up2(feat))

        #out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        out = self.conv_last(self.gelu(self.conv_hr(feat)))
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            # Use smaller std for better stability and performance. We
            # use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
            # Generative Adversarial Networks"
            for m in [
                    self.conv_first, self.conv_body, self.conv_body_att, self.conv_up1,
                    self.conv_up2, self.conv_hr, self.conv_last
            ]:
                default_init_weights(m, 0.1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
