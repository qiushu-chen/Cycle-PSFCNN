# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:56:36 2024

@author: cqsfdu
"""

import numbers
import os.path as osp
from copy import deepcopy

import mmcv
import torch
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper, MMDistributedDataParallel
from mmcv.runner import auto_fp16
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy.fft import rfft2, irfft2
import scipy
from functools import reduce

from mmedit.core import tensor2img
# from ..base import BaseModel
# from .srgan import SRGAN
from .basic_restorer import BasicRestorer
from ..builder import build_backbone, build_component, build_loss
from ..common import GANImageBuffer, set_requires_grad
from ..registry import MODELS

from mmcls.models.builder import build_backbone as mmcls_backbone
from mmcls.models.builder import build_head as mmcls_head
from mmcls.models.builder import build_neck as mmcls_neck

def to_radial(x, y):
    return x ** 2 + y ** 2

def to_radial_torch(x, y):
    return torch.pow(x,2) + torch.pow(y,2)

def to_radian(x):
    return float(x) * np.pi / 180.

def to_radian_torch(x):
    return x * np.pi / 180.

class Params:
    def __init__(self):
        # self.tilt = 0.
        # self.tilt_angle = to_radian(0.)
        # self.focus = 0.
        # self.coma = 0.
        # self.coma_angle = to_radian(0.)
        # self.ast = 0.
        # self.ast_angle = to_radian(0.)
        # self.sph = 0.
        self.size = 63. # px
        self.wavelength=520. # nm
        self.tubelength=180. # mm
        self.na = 0.3
        self.n = 1.0 #refraction
        self.magnification = 10.
        self.pixelsize=7.3 # um

def unpad(img, npad):
    '''
    Revert the np.pad command
    '''
    return img[npad:-npad, npad:-npad]

def scale(v):
    '''
    Normalize a 2D matrix with a maximum of 1 per pixel
    :param v:
    :return: normalized vector
    '''
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    out = out * (1 / np.max(np.abs(out)))
    if np.all(np.isfinite(out)):
        return out
    else:
        print('Error, image is not finite (dividing by infinity on norm).')
        return np.zeros(v.shape)
    
def scale_torch(v):
    norm = torch.linalg.norm(v, ord=1)
    if norm==0:
        norm = torch.finfo(v.dtype).eps
    out = v / norm
    out = out * (1/torch.max(torch.abs(out)))
    if torch.all(torch.isfinite(out)):
        return out
    else:
        print('Error, image is not finite (dividing by infinity on norm).')
        return torch.zeros(v.shape, device="cuda:0")
        

def get_wavefront(x,y,params):
    x = 2.*x/params.size
    y = 2.*y/params.size
    r2 = to_radial(x, y)

    aberration = params.sph * r2**2 + params.focus * r2 + params.ast * (x*np.cos(params.ast_angle) + y*np.sin(params.ast_angle))**2 + \
            params.coma * ( (x*r2)*np.cos(params.coma_angle) + (y*r2)*np.sin(params.coma_angle)) + \
            params.tilt*(x*np.cos(params.tilt_angle) + y*np.sin(params.tilt_angle))

    wavefront = np.exp(2*1j*np.pi*aberration)
    return wavefront

def get_wavefront_torch(x,y,params,cnn_pa):
    x = torch.tensor(2.*x/params.size,device="cuda:0")
    y = torch.tensor(2.*y/params.size,device="cuda:0")
    r2 = to_radial_torch(x, y) 
    aberration = cnn_pa[4]*torch.pow(r2,2) + cnn_pa[1]*r2 + cnn_pa[2]*torch.pow((x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3])),2) + \
            cnn_pa[6] * ((x*r2)*torch.cos(cnn_pa[3]) + (y*r2)*torch.sin(cnn_pa[3])) + \
           cnn_pa[5] * (x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3]))
    wavefront = torch.exp(2*1j*np.pi*aberration)
    return wavefront

# def get_psf(params, centered = True):
def get_psf(params, cnn_pa, centered = True):
    datapoints = int(params.size)
    padding = int(np.ceil(datapoints/2))
    totalpoints = datapoints + 2*padding
    center_point = int(np.floor(totalpoints/2))
    wavelength = params.wavelength * float(1e-9) #wavelength in m
    pupil_diameter = 2.0 * params.tubelength * params.na / (params.magnification * params.n)
    D = pupil_diameter*1e-3 # diameter in m
    d = 1.8*1e-2 # distance btw pupil plane and object
    PRw = D / (2 * wavelength * d) # unit = 1/m
    NT = params.size//2
    x = np.linspace(-NT, NT, datapoints)
    y = np.linspace(-NT, NT, datapoints)
    xx, yy = np.meshgrid(x, y)
    sums = torch.tensor(np.power(xx,2) + np.power(yy,2))
    # wavefront = get_wavefront(xx, yy, params)
    wavefront = get_wavefront_torch(xx, yy, params, cnn_pa)
    pixel_limit = PRw*params.size*params.pixelsize*1e-6
    wavefront0 = torch.ones(wavefront.shape, device="cuda:0")
    wavefront0[sums > pixel_limit] = 0.0
    wavefront_fix = wavefront*wavefront0
    wavefront_padded = torch.nn.functional.pad(wavefront_fix, (padding,padding,padding,padding), "constant", 0.0)
    # wavefront_padded = np.pad(wavefront, ((padding,padding),(padding,padding)), mode='constant',constant_values=(0))
    # psf = np.power(np.abs(np.fft.fft2(wavefront_padded, norm='ortho')),2)
    # psf = np.roll(psf, center_point, axis = (0,1))
    psf = torch.pow(torch.abs(torch.fft.fft2(wavefront_padded, norm='ortho')),2)
    psf_new1 = torch.roll(psf, center_point, dims=0)
    psf_new2 = torch.roll(psf_new1, center_point, dims=1)   
    normalisation = torch.pow(torch.abs(wavefront).sum()/float(totalpoints),2)
    # normalisation = np.power(np.sum(np.abs(wavefront)) / float(totalpoints),2)
    psf_final0 = unpad(psf_new2, padding) / normalisation
    psf_final = scale_torch(torch.fliplr(psf_final0)).type(torch.float32)
    # print("psf device: ",psf_final.device, params)
    # psf = scale(np.fliplr(psf)).astype(np.float32)
    return psf_final, wavefront, pupil_diameter

def _centered(arr, newshape):
    """
    Return the center newshape portion of the array.
    """
    newshape = np.asarray(newshape)
    # currshape = np.array(arr.shape)
    currshape = np.asarray(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def div0( a, b ):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def div0_torch(a,b):
    #old_err_mode = torch.get_err_mode()
    #old_err_msg = torch.get_num_threads()
    #torch.set_err(torch.ErrMode.IGNORE)
    c = torch.true_divide(a,b)
    # print(c)
    c[ ~ torch.isfinite( c )] = 0
    #torch.set_err(old_err_mode)
    #torch.set_num_threads(old_err_msg)
    return c
    
def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return reduce(np.add,np.gradient(F))

def div_torch(F):
    return reduce(torch.add,torch.gradient(F)) 

def rl_deconvolution(img_list, psf_list, iterations=20, lbd=0.2):   #temp np type, try the application to tensor
    min_value = []
    for img_idx, img in enumerate(img_list):
        img_list[img_idx] = np.pad(img_list[img_idx], np.max(psf_list[0].shape), mode='reflect')
        min_value.append(np.min(img))
        img_list[img_idx] = img_list[img_idx] - np.min(img)
    size = np.array(np.array(img_list[0].shape) + np.array(psf_list[0].shape)) - 1
    fsize = [scipy.fftpack.helper.next_fast_len(int(d)) for d in size]
    fslice = tuple([slice(0, int(sz)) for sz in size])
    
    latent_estimate = img_list.copy()
    error_estimate = img_list.copy()

    psf_f = []
    psf_flipped_f = []
    for img_idx, img in enumerate(latent_estimate):
        psf_f.append(rfft2(psf_list[img_idx], fsize))
        _psf_flipped = np.flip(psf_list[img_idx], axis=0)
        _psf_flipped = np.flip(_psf_flipped, axis=1)
        psf_flipped_f.append(np.fft.rfft2(_psf_flipped, fsize))
        
    for i in range(iterations):
        regularization = np.ones(img_list[0].shape)
        for img_idx, img in enumerate(latent_estimate):
            estimate_convolved = irfft2(np.multiply(psf_f[img_idx], rfft2(latent_estimate[img_idx], fsize)))[fslice].real
            estimate_convolved = _centered(estimate_convolved, img.shape)
            relative_blur = div0(img_list[img_idx], estimate_convolved)
            error_estimate[img_idx] = irfft2(np.multiply(psf_flipped_f[img_idx], rfft2(relative_blur, fsize)), fsize)[fslice].real
            error_estimate[img_idx] = _centered(error_estimate[img_idx], img.shape)
            regularization += 1.0 - (lbd * divergence(latent_estimate[img_idx] / np.linalg.norm(latent_estimate[img_idx], ord=1)))
            latent_estimate[img_idx] = np.multiply(latent_estimate[img_idx], error_estimate[img_idx])
            
        for img_idx, img in enumerate(img_list):
            latent_estimate[img_idx] = np.divide(latent_estimate[img_idx], regularization/float(len(img_list)))
    for img_idx, img in enumerate(latent_estimate):
        latent_estimate[img_idx] += min_value[img_idx]
        latent_estimate[img_idx] = unpad(latent_estimate[img_idx], np.max(psf_list[0].shape))
    return latent_estimate

def rl_deconvolution_torch(img_list0, psf_list, iterations=25, lbd=0.2):
    img_list = img_list0*255.0
    padding = np.max(psf_list[0].shape)
    # print(padding)
    if img_list.dim() == 3:
        img_list = img_list.unsqueeze(0)
    img_list_pad = torch.nn.functional.pad(img_list, (padding, padding, padding, padding), "reflect")
    min_value_temp = torch.zeros(len(img_list), device="cuda:0")
    max_value_temp = torch.zeros(len(img_list), device="cuda:0")
    for k in range(len(img_list)):
        max_value_temp[k] = max_value_temp[k] + torch.max(img_list[k])
        min_value_temp[k] = min_value_temp[k] + torch.min(img_list[k])
        # img_list_pad[k] = img_list_pad[k] - min_value_temp[k]
        img_list_pad[k] = max_value_temp[k] - img_list_pad[k]
    # print(torch.max(img_list[k]), torch.min(img_list[k]))
    size = np.array(np.array(img_list_pad[0, 0].shape) + np.array(psf_list[0].shape)) - 1
    fsize = [scipy.fftpack.helper.next_fast_len(int(d)) for d in size]
    latent_estimate = img_list_pad.clone()
    tens = dict()
    for i in range(iterations):
        for k in range(len(img_list)):
            tens[f'latent_temp{i}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
            tens[f'latent_final_estimate{i}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
            tens[f'error_estimate{i}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
            tens[f'estimate_convolved{i}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
            tens[f'relative_blur{i}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
            tens[f'regularization{i}_{k}'] = torch.ones(img_list_pad[0].shape, device="cuda:0")
    for k in range(len(img_list)):
        tens[f'latent_temp{iterations}_{k}'] = torch.zeros(latent_estimate[0].shape, device="cuda:0")
    size_ext = [3]
    fsize_extend = tuple(sz for sz in fsize)
    size_ext = np.hstack([size_ext, size])
    fslice = tuple([slice(0, int(sz)) for sz in size_ext])
    # print(fsize_extend, psf_list.shape)
    psf_f = torch.fft.rfft2(psf_list, fsize_extend)
    # print(psf_f.shape)
    _psf_flipped = torch.flip(psf_list, dims=[1])
    _psf_flipped = torch.flip(_psf_flipped, dims=[2])
    psf_flipped_f = torch.fft.rfft2(_psf_flipped, s=fsize_extend)
    for k in range(len(img_list)): 
        tens[f'latent_temp0_{k}'] = latent_estimate[k].clone()
    for i in range(iterations):
        for k in range(len(img_list)):            
            tens[f'estimate_convolved{i}_{k}'] = tens[f'estimate_convolved{i}_{k}'] + _centered(torch.fft.irfft2(torch.mul(psf_f[k],torch.fft.rfft2(tens[f'latent_temp{i}_{k}'],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            tens[f'relative_blur{i}_{k}'] = tens[f'relative_blur{i}_{k}'] + div0_torch(img_list_pad[k], tens[f'estimate_convolved{i}_{k}'])
            tens[f'error_estimate{i}_{k}'] = tens[f'error_estimate{i}_{k}'] + _centered(torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(tens[f'relative_blur{i}_{k}'],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            tens[f'regularization{i}_{k}'] = tens[f'regularization{i}_{k}'] + 1.0 - (lbd*div_torch(tens[f'latent_temp{i}_{k}']/torch.linalg.norm(tens[f'latent_temp{i}_{k}'][0], ord=1)))
            tens[f'latent_final_estimate{i}_{k}'] = tens[f'latent_final_estimate{i}_{k}'] + tens[f'latent_temp{i}_{k}']*tens[f'error_estimate{i}_{k}']
            tens[f'latent_temp{i+1}_{k}'] = torch.divide(tens[f'latent_final_estimate{i}_{k}'], tens[f'regularization{i}_{k}'])
            # print(k, torch.max(tens[f'latent_temp{i+1}_{k}']), torch.min(tens[f'latent_temp{i+1}_{k}']))
    latent_final_estimate1 = torch.stack([tens[f'latent_temp{iterations}_{k}'] for k in range(len(img_list))],dim=0)
    for k in range(len(img_list)):
        latent_final_estimate1[k] =  max_value_temp[k] - 1.5*latent_final_estimate1[k]
        # print(max_value_temp[k], torch.max(latent_estimate[k]))
    latent_final_estimate1 = latent_final_estimate1[:, :, padding:-padding, padding:-padding]
    # for k in range(len(latent_final_estimate1)):
    #     print(k, torch.max(latent_final_estimate1[k]), torch.min(latent_final_estimate1[k]))
    return latent_final_estimate1/255.0
        
def forwardtest_rl_deconvolution(img_list, psf_list, iterations=20, lbd=0.2):
    latent_temp = rl_deconvolution(img_list, psf_list, iterations, lbd)
    return np.sum(latent_temp, axis=0)      

def normalize(v):
    '''
    Normalize a 2D matrix with a sum of 1
    :param v:
    :return: normalized vector
    '''
    norm = v.sum()
    if norm == 0:
        # norm = np.finfo(v.dtype).eps
        norm = torch.finfo(v.dtype).eps
    return v / norm

def convolve(input, psf, padding = 'constant'):
    '''
    Convolve an image with a psf using FFT
    :param padding: replicate, reflect, constant
    :return: output image
    '''
    psf = normalize(psf)
    npad = np.max(psf.shape)

    if len(input.shape) != len(psf.shape):
        #print("Warning, input has shape : {}, psf has shape : {}".format(input.shape, psf.shape))
        input = input[:,:,0]
        #print("New input shape : {}".format(input.shape))

    input = np.pad(input, pad_width=npad, mode=padding)

    try:
        out = scipy.signal.fftconvolve(input, psf, mode='same')
    except:
        print("Exception: FFT cannot be made on image !")
        out = np.zeros(input.shape)

    out = unpad(out, npad)
    return out

def conv_with_torch(img_list0, psf_list0):
    img_list = img_list0*255.0
    # print("for conv:",torch.max(img_list), torch.min(img_list))
    # print(psf_list0.shape)
    psf_list = torch.zeros(psf_list0.shape, device="cuda:0")
    # psf_list = normalize(psf_list0)
    npad = np.max(psf_list.shape)
    paddingsize = npad//2
    img_list_pad = torch.nn.functional.pad(img_list, (paddingsize, paddingsize, paddingsize, paddingsize), "reflect")
    # print(img_list_pad.shape)
    tensc = dict()
    out_list = torch.zeros(img_list.shape, device="cuda:0")
    for step in range(len(psf_list)):
        psf_list[step] = psf_list[step] + normalize(psf_list0[step])
        tensc[f'psf_w{step}'] = psf_list[step].clone()
        tensc[f'psf_w{step}'] = tensc[f'psf_w{step}'].unsqueeze(0)
        tensc[f'psf_w{step}'] = tensc[f'psf_w{step}'].unsqueeze(0)
        for image_channel in range(3):
            tensc[f'img_w{step}_{image_channel}'] = img_list_pad[step, image_channel]
            tensc[f'img_w{step}_{image_channel}'] = tensc[f'img_w{step}_{image_channel}'].unsqueeze(0)
            tensc[f'img_w{step}_{image_channel}'] = tensc[f'img_w{step}_{image_channel}'].unsqueeze(0)
            tensc[f'out_{step}_{image_channel}'] = torch.nn.functional.conv2d(tensc[f'img_w{step}_{image_channel}'], tensc[f'psf_w{step}'], padding=0)
            tensc[f'out_{step}_{image_channel}'] = tensc[f'out_{step}_{image_channel}'].squeeze()
            tensc[f'out_{step}_{image_channel}'] = tensc[f'out_{step}_{image_channel}'].squeeze()
            # print(tensc[f'out_{step}_{image_channel}'].shape)
            out_list[step, image_channel] = out_list[step, image_channel] + tensc[f'out_{step}_{image_channel}']
        # print("after conv:", step, torch.max(tensc[f'out_{step}_{image_channel}']), torch.min(tensc[f'out_{step}_{image_channel}']))
    out_list0 = out_list/255.0
    return out_list0

f2 = open("./test_psf_ori1.txt", 'r')
psflines = f2.readlines()
f2.close()
psf_list_focus = np.zeros((63,63))
for i in range(len(psflines)-1):
    test_pst_line = psflines[i][2:]
    test_psf_line = test_pst_line.split('  ')
    for j in range(len(test_psf_line)-1):
        psf_list_focus[i,j] = float(test_psf_line[j+1][:-1])
# print(psf_list)
psf_list_focus = torch.tensor(psf_list_focus, device="cuda:0", dtype=torch.float32)
# print(psf_list_focus.dtype)
psf_list_focus = psf_list_focus.unsqueeze(0)
psf_list_focus8 = torch.cat((psf_list_focus,psf_list_focus,psf_list_focus,psf_list_focus,psf_list_focus,psf_list_focus,psf_list_focus,psf_list_focus),0)

@MODELS.register_module()
class PSFCycleGAN(BasicRestorer):
    def __init__(self,
                 generator,
                 discriminator,
                 psfcnn,            # the new cycle
                 # psfcnnneck,        # the new cycle
                 psfcnnhead,        # the new cycle
                 gan_loss,
                 cycle_loss,
                 cycle_loss2=None,
#                  psf_loss,          # the new cycle
                 perceptual_loss=None,    # the new cycle
                 pixel_loss=None,         # the new cycle
                 pixel_loss2=None,
                 id_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # identity loss only works when input and output images have the same
        # number of channels
        if id_loss is not None and id_loss.get('loss_weight') > 0.0:
            assert generator.get('in_channels') == generator.get(
                'out_channels')

        # generators in old cyclegan
        # self.generators = nn.ModuleDict()
        # self.generators['a'] = build_backbone(generator)
        # self.generators['b'] = build_backbone(generator)
        
        # generators in psf cyclegan
        self.generator = build_backbone(generator)
        self.psfcnn = mmcls_backbone(psfcnn)
        # self.psfcnnneck = mmcls_neck(psfcnnneck)
        self.psfcnnhead = mmcls_head(psfcnnhead)

        # discriminators in old cyclegan
        # self.discriminators = nn.ModuleDict()
        # self.discriminators['a'] = build_component(discriminator)
        # self.discriminators['b'] = build_component(discriminator)
        
        # discriminators in psf cyclegan
        self.discriminator = build_component(discriminator)

        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        self.image_buffers['lq'] = GANImageBuffer(self.buffer_size)
        self.image_buffers['gt'] = GANImageBuffer(self.buffer_size)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_loss(gan_loss)
        assert cycle_loss is not None  # cycle loss cannot be None
        self.cycle_loss = build_loss(cycle_loss)
        self.id_loss = build_loss(id_loss) if id_loss else None
#         assert psf_loss is not None    # psf loss in psf cyclegan
#         self.psf_loss = build_loss(psf_loss)
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.pixel_loss2 = build_loss(pixel_loss2) if pixel_loss2 else None
        self.cycle_loss2 = build_loss(cycle_loss2) if cycle_loss2 else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        # others
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        # if self.train_cfg is None:
        #     self.direction = ('a2b' if self.test_cfg is None else
        #                       self.test_cfg.get('direction', 'a2b'))
        # else:
        #     self.direction = self.train_cfg.get('direction', 'a2b')
        self.step_counter = 0  # counting training steps
        # self.register_buffer('step_counter', torch.zeros(1))
        self.show_input = (False if self.test_cfg is None else
                           self.test_cfg.get('show_input', False))
        # In CycleGAN, if not showing input, we can decide the translation
        # direction in the test mode, i.e., whether to output fake_b or fake_a
        # if not self.show_input:
        #     self.test_direction = ('a2b' if self.test_cfg is None else
        #                            self.test_cfg.get('test_direction', 'a2b'))
        #     if self.direction == 'b2a':
        #         self.test_direction = ('b2a' if self.test_direction == 'a2b'
        #                                else 'a2b')

        # support fp16
        self.fp16_enabled = False
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        # self.generators['a'].init_weights(pretrained=pretrained)
        # self.generators['b'].init_weights(pretrained=pretrained)
        # self.discriminators['a'].init_weights(pretrained=pretrained)
        # self.discriminators['b'].init_weights(pretrained=pretrained)
        self.generator.init_weights(pretrained=pretrained)
        self.discriminator.init_weights(pretrained=pretrained)
        
    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module

        return module
    
    @auto_fp16(apply_to=('lq', 'lq2'))
    def forward(self, lq, lq2, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if test_mode:
            return self.forward_test(lq, lq2, gt, **kwargs)

        raise ValueError(
            'SRGAN model does not support `forward_train` function.') 
    
    
    def train_step(self, data_batch, optimizer):
        lq = data_batch['lq']
        lq2 = data_batch['lq2']
        
        # img_size = lq2.size()
        # img_fft_tensor = torch.zeros((img_size[0],2,img_size[2],img_size[3]),device='cuda:0')
        # for i in range(img_size[0]):
        #     fft_temp = torch.fft.fft2(lq2[i,1,:,:], dim=(-2,-1))
        #     fft_temp = torch.stack((fft_temp.real, fft_temp.imag), 0)
        #     img_fft_tensor[i,:,:,:] = fft_temp
        # lq2_withfft = torch.cat((lq2,img_fft_tensor),1)
        gt = data_batch['gt']
        psf_label = data_batch['gt_label']
        # print("input: ",torch.max(lq), torch.min(lq),torch.max(gt),torch.min(gt))
        losses=dict()
        gt_pixel, gt_percep, gt_gan, gt_cycle = gt.clone(), gt.clone(), gt.clone(), gt.clone()
        lq_cycle, lq_pixel, lq_gan = lq.clone(), lq.clone(), lq.clone()
        # psf_lq_x = self.psfcnn(lq2_withfft)
        psf_lq_x = self.psfcnn(lq2)
        # psf_lq_x = self.psfcnnneck(psf_lq_x)
        # psf_lq_x = self.psfcnnhead.pre_logits(psf_lq_x)
        psf_lq_parax, psf_lossdict = self.psfcnnhead.forward_train(psf_lq_x, psf_label)
        # psf_loss = psf_lossdict['loss']
        # need to deal with tensor to img to tensor
        psf_lq_para = Params()
        # print(psf_lq_parax, psf_lossdict, psf_label)
        if psf_lq_parax.dim()==1:
            psf_lq_temp, wavefront_lq, pupil_diameter_lq = get_psf(psf_lq_para, psf_lq_parax)
            psf_lq = torch.unsqueeze(psf_lq_temp,0)
        elif psf_lq_parax.dim()==2:
            psf_lq_dict = dict()
            for i in range(len(psf_lq_parax)):
                psf_lq_dict[f'psf_w{i}'], wavefront_lq, pupil_diameter_lq = get_psf(psf_lq_para, psf_lq_parax[i])
            psf_lq = torch.stack([psf_lq_dict[f'psf_w{i}'] for i in range(len(psf_lq_parax))], dim=0)
        # print(psf_lq)
        lq_cycle_list = lq_cycle
        psf_lq_1 = psf_lq.detach()
        #lq_cycle_list = tensor2img(lq_cycle)
        # psf_lq_para = Params()
        # psf_lq_para.focus = psf_lq_parax[0,1]
        # psf_lq_para.ast = psf_lq_parax[0,2]
        # psf_lq_para.ast_angle = psf_lq_parax[0,3]
        # psf_lq, wavefront_lq, pupil_diameter_lq = get_psf(psf_lq_para)        
        # fake_gt_pre = torch.from_numpy(fake_gt_pre_list.transpose(0, 3, 1, 2))
        # end of the img to tensor process
        fake_gt_pre_ori1 = rl_deconvolution_torch(lq_cycle_list, psf_lq_1)
        # print("psfcnn out: ",torch.max(fake_gt_pre_ori1), torch.min(fake_gt_pre_ori1))
        fake_gt_pre_ori2 = F.avg_pool2d(fake_gt_pre_ori1, kernel_size=3, stride=2, padding=1)
        fake_gt_pre = F.avg_pool2d(fake_gt_pre_ori2, kernel_size=3, stride=2, padding=1)
        fake_gt = self.generator(fake_gt_pre)
        # print("fake gt: ",torch.max(fake_gt), torch.min(fake_gt))
        rec_lq = conv_with_torch(fake_gt, psf_lq)
        # print("rec lq: ",torch.max(rec_lq), torch.min(rec_lq))
        # print(rec_lq.shape)
        
        gt_defocus = rl_deconvolution_torch(gt_cycle, psf_list_focus8)
        fake_lq = conv_with_torch(gt_defocus, psf_lq)
        # print(fake_lq.shape)
        rec_gt_pre_ori1 = rl_deconvolution_torch(fake_lq.detach(), psf_lq_1)
        rec_gt_pre_ori2 = F.avg_pool2d(rec_gt_pre_ori1, kernel_size=3, stride=2, padding=1)
        rec_gt_pre = F.avg_pool2d(rec_gt_pre_ori2, kernel_size=3, stride=2, padding=1)
        rec_gt = self.generator(rec_gt_pre)
        
        losses = dict()
        log_vars = dict()
        losses_p = dict()
        # log_vars_p0 = dict()
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)
        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix_gt'] = self.pixel_loss(fake_gt, gt_pixel) + self.pixel_loss2(fake_gt, gt_pixel)
                losses['loss_pix_lq'] = self.pixel_loss(fake_lq, lq_pixel) + self.pixel_loss2(fake_lq, lq_pixel)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_gt, gt_percep)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
            if self.gan_loss:
                real_d_pred = self.discriminator(gt).detach()
                fake_g_pred = self.discriminator(fake_gt)
                loss_gan_fake = self.gan_loss(
                    fake_g_pred - torch.mean(real_d_pred),
                    target_is_real=True,
                    is_disc=False)
                loss_gan_real = self.gan_loss(
                    real_d_pred - torch.mean(fake_g_pred),
                    target_is_real=False,
                    is_disc=False)
                losses['loss_gan'] = (loss_gan_fake + loss_gan_real) / 2
            if self.cycle_loss:
                losses['loss_cycle_lq'] = self.cycle_loss(rec_lq, lq_gan) + self.cycle_loss2(rec_lq, lq_gan)
                losses['loss_cycle_gt'] = self.cycle_loss(rec_gt, gt_gan) + self.cycle_loss2(rec_gt, gt_gan)
            losses['loss_psfcnn'] = psf_lossdict['loss']            
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)
            # loss_p, log_vars_p = self.parse_losses(psf_lossdict)
            # log_vars.update(log_vars_p)
            optimizer['generator'].zero_grad()
            optimizer['psfcnn'].zero_grad()
            # optimizer['psfcnnhead'].zero_grad()
            loss_g.backward()
            # loss_p.backward()
            optimizer['generator'].step()
            optimizer['psfcnn'].step()
            # optimizer['psfcnnhead'].step()
        
        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            fake_d_pred = self.discriminator(fake_gt).detach()  #added
            real_d_pred = self.discriminator(gt_gan)
            loss_d_real = self.gan_loss(
                real_d_pred - torch.mean(fake_d_pred),
                target_is_real=True,
                is_disc=True
            ) * 0.5
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)
            # fake
            fake_d_pred = self.discriminator(fake_gt.detach())
            loss_d_fake = self.gan_loss(
                fake_d_pred - torch.mean(real_d_pred.detach()),
                target_is_real=False,
                is_disc=True
            ) * 0.5
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)
            optimizer['discriminator'].step()
            
        self.step_counter += 1
        log_vars.pop('loss')  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(
                real_lq=lq.cpu(),
                fake_gt=fake_gt.cpu(),
                real_gt=gt.cpu(),
                fake_lq=fake_lq.cpu()))
        
        return results

    def forward_test(self,
                     lq,
                     lq2,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        _model_p = self.psfcnn
        # _model_pn = self.psfcnnneck
        _model_ph = self.psfcnnhead
        _model_g = self.generator
        
        # img_size = lq2.size()
        # img_fft_tensor = torch.zeros((img_size[0],2,img_size[2],img_size[3]),device='cuda:0')
        # for i in range(img_size[0]):
        #     fft_temp = torch.fft.fft2(lq2[i,1,:,:], dim=(-2,-1))
        #     fft_temp = torch.stack((fft_temp.real, fft_temp.imag), 0)
        #     img_fft_tensor[i,:,:,:] = fft_temp
        # lq2_withfft = torch.cat((lq2,img_fft_tensor),1)
        
        # psf_xt = _model_p(lq2_withfft)
        psf_xt = _model_p(lq2)
        # psf_xt = _model_pn(psf_xt)
        # psf_xt = _model_ph.pre_logits(psf_xt)
        psf_xt_parax = _model_ph.simple_test(psf_xt, post_process=False)  # not a list any more
        psf_xt_para = Params()
        if psf_xt_parax.dim()==1:
            psf_xt_temp, wavefront_xt, pupil_diameter_xt = get_psf(psf_xt_para, psf_xt_parax)
            psf_test = torch.unsqueeze(psf_xt_temp,0)
        elif psf_xt_parax.dim()==2:
            psf_xt = dict()
            for i in range(len(psf_xt_parax)):
                psf_xt[f'psf_w{i}'], wavefront_xt, pupil_diameter_xt = get_psf(psf_xt_para, psf_xt_parax[i])
            psf_test = torch.stack([psf_xt[f'psf_w{i}'] for i in range(len(psf_xt_parax))], dim=0)
        # psf_xt_para.focus = psf_xt_parax[0,1]
        # psf_xt_para.ast = psf_xt_parax[0,2]
        # psf_xt_para.ast_angle = psf_xt_parax[0,3]
        # psf_test, wavefront_test, pupil_diameter_test = get_psf(psf_xt_para)        
        output_pre_ori1 = rl_deconvolution_torch(lq, psf_test)
        output_pre_ori2 = F.avg_pool2d(output_pre_ori1, kernel_size=3, stride=2, padding=1)
        output_pre = F.avg_pool2d(output_pre_ori2, kernel_size=3, stride=2, padding=1)
        output = _model_g(output_pre)
        
        if self.test_cfg is not None and self.test_cfg.get(
                'metrics', None) and gt is not None:
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
        
        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name0 = lq_path.split('/')
            folder_name1 = folder_name0[-2] + '_' + osp.splitext(osp.basename(lq_path))[0]
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name1,
                                     f'{folder_name}-{iteration + 1:06d}.tif')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name1}.tif')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results