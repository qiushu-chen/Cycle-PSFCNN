# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:59:08 2024

@author: cqsfdu
"""
import numbers
import os
import os.path as osp
from copy import deepcopy
import cv2

import mmcv
import torch
import numpy as np
import scipy
from functools import reduce

def to_radial_torch(x, y):
    return torch.pow(x,2) + torch.pow(y,2)

def to_radian(x):
    return float(x) * np.pi / 180.

def rand_float(a, b, size=1):
    '''
    Return random floats in the half-open interval [a, b).
    '''
    return (b - a) * np.random.random_sample(size) + a


def rand_int(a, b, size=1):
    '''
    Return random integers in the half-open interval [a, b).
    '''
    return np.floor((b - a) * np.random.random_sample(size) + a).astype(dtype=np.int16)

class Params:
    def __init__(self):
        self.size = 63. # px
        self.wavelength=520. # nm
        self.tubelength=180. # mm
        self.na = 0.3
        self.n = 1 #refraction
        self.magnification = 10.
        self.pixelsize = 6.5 # um
        
def unpad(img, npad):
    '''
    Revert the np.pad command
    '''
    return img[npad:-npad, npad:-npad]

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
        return torch.zeros(v.shape)
    
def get_wavefront_torch(x,y,params,cnn_pa):
    x = torch.tensor(2.*x/params.size)
    y = torch.tensor(2.*y/params.size)
    r2 = to_radial_torch(x, y)
    aberration = cnn_pa[4]*torch.pow(r2,2) + cnn_pa[1]*r2 + cnn_pa[2]*torch.pow((x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3])),2) + \
            cnn_pa[6] * ((x*r2)*torch.cos(cnn_pa[3]) + (y*r2)*torch.sin(cnn_pa[3])) + \
            cnn_pa[5]*(x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3]))
    wavefront = torch.exp(2*1j*np.pi*aberration)
    return wavefront

def get_psf(params, cnn_pa, centered = True):
    datapoints = int(params.size)
    padding = int(np.ceil(datapoints/2))
    totalpoints = datapoints + 2*padding
    center_point = int(np.floor(totalpoints/2))

    wavelength = params.wavelength * float(1e-9) #wavelength in m
    pupil_diameter = 2.0 * params.tubelength * params.na / (params.magnification * params.n)
    D = pupil_diameter*1e-3 # diameter in m
    d = 1.8*1e-2 # distance btw pupil plane and object
    PRw = D / (2 * wavelength * d) # unit = 1/m, 
    NT = params.size//2
    x = np.linspace(-NT, NT, datapoints)
    y = np.linspace(-NT, NT, datapoints)
    xx, yy = np.meshgrid(x, y)
    sums = torch.tensor(np.power(xx,2) + np.power(yy,2))
    wavefront = get_wavefront_torch(xx, yy, params, cnn_pa)
    pixel_limit = PRw*params.size*params.pixelsize*1e-6
    wavefront0 = torch.ones(wavefront.shape)
    wavefront0[sums > pixel_limit] = 0.0
    wavefront_fix = wavefront*wavefront0
    wavefront_padded = torch.nn.functional.pad(wavefront_fix, (padding,padding,padding,padding), "constant", 0.0)
    psf = torch.pow(torch.abs(torch.fft.fft2(wavefront_padded, norm='ortho')),2)
    psf_new1 = torch.roll(psf, center_point, dims=0)
    psf_new2 = torch.roll(psf_new1, center_point, dims=1)
    
    normalisation = torch.pow(torch.abs(wavefront).sum()/float(totalpoints),2)
    psf_final0 = unpad(psf_new2, padding) / normalisation
    psf_final = scale_torch(torch.fliplr(psf_final0)).type(torch.float32)   
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
    # print(myslice)
    return arr[tuple(myslice)]

def div0_torch(a,b):
    c = torch.true_divide(a,b)
    c[ ~ torch.isfinite( c )] = 0
    return c

def div_torch(F):
    return reduce(torch.add,torch.gradient(F)) 

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

def conv_with_torch(img_list, psf_list):
    psf_list = normalize(psf_list)
    npad = np.max(psf_list.shape)
    paddingsize = npad//2
    img_list_pad = torch.nn.functional.pad(img_list, (paddingsize, paddingsize, paddingsize, paddingsize), "reflect")
    print(img_list_pad.shape)
    tensc = dict()
    out_list_ori = torch.zeros(img_list_pad.shape)
    out_list = torch.zeros(img_list.shape)
    for step in range(len(psf_list)):
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
    return out_list

def rl_deconvolution_torch(img_list, psf_list, iterations=30, lbd=0.2):
    padding = np.max(psf_list[0].shape)
    if img_list.dim() == 3:
        img_list = img_list.unsqueeze(0)
    img_list_pad = torch.nn.functional.pad(img_list, (padding, padding, padding, padding), "reflect")
    min_value_temp = torch.zeros(len(img_list))
    max_value_temp = torch.zeros(len(img_list))
    for k in range(len(img_list)):
        max_value_temp[k] = max_value_temp[k] + torch.max(img_list[k])
        min_value_temp[k] = min_value_temp[k] + torch.min(img_list[k])
        img_list_pad[k] = max_value_temp[k] - img_list_pad[k]
    size = np.array(np.array(img_list_pad[0, 0].shape) + np.array(psf_list[0].shape)) - 1
    fsize = [scipy.fftpack.helper.next_fast_len(int(d)) for d in size]   
    latent_estimate = img_list_pad.clone()
    error_estimate = img_list_pad.clone()
    num_iter = np.hstack((iterations,latent_estimate.shape))
    num_iter1 = tuple(nm for nm in num_iter)
    num_iterre = np.hstack((iterations,img_list_pad[0].shape))
    num_iter2 = tuple(nm for nm in num_iterre)
    tens = dict()
    for i in range(iterations):
        tens[f'latent_temp{i}'] = torch.zeros(latent_estimate.shape)
        # tens[f'latent_final_estimate{i}'] = torch.zeros(latent_estimate.shape)
        tens[f'error_estimate{i}'] = torch.zeros(latent_estimate.shape)
        tens[f'estimate_convolved{i}'] = torch.zeros(latent_estimate.shape)
        tens[f'relative_blur{i}'] = torch.zeros(latent_estimate.shape)
        tens[f'regularization{i}'] = torch.ones(img_list_pad[0].shape)
    tens[f'latent_temp{iterations}'] = torch.zeros(latent_estimate.shape)
    size_ext = [3]
    fsize_extend = tuple(sz for sz in fsize)
    size_ext = np.hstack([size_ext, size])
    fslice = tuple([slice(0, int(sz)) for sz in size_ext])
    psf_f = torch.fft.rfft2(psf_list, fsize_extend)
    _psf_flipped = torch.flip(psf_list, dims=[1])
    _psf_flipped = torch.flip(_psf_flipped, dims=[2])
    psf_flipped_f = torch.fft.rfft2(_psf_flipped, s=fsize_extend)
    tens['latent_temp0'] = latent_estimate.clone()
    for i in range(iterations):
        for k in range(len(img_list)):            
            tens[f'estimate_convolved{i}'][k] = tens[f'estimate_convolved{i}'][k] + _centered(torch.fft.irfft2(torch.mul(psf_f[k],torch.fft.rfft2(tens[f'latent_temp{i}'][k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            tens[f'relative_blur{i}'][k] = tens[f'relative_blur{i}'][k] + div0_torch(img_list_pad[k], tens[f'estimate_convolved{i}'][k])
            tens[f'error_estimate{i}'][k] = tens[f'error_estimate{i}'][k] + _centered(torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(tens[f'relative_blur{i}'][k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            for k1 in range(3):
                tens[f'regularization{i}'][k1] = tens[f'regularization{i}'][k1] + 1.0 - (lbd*div_torch(tens[f'latent_temp{i}'][k,k1]/torch.linalg.norm(tens[f'latent_temp{i}'][k,k1], ord=1)))
            tens[f'latent_temp{i+1}'][k] = tens[f'latent_temp{i+1}'][k] + tens[f'latent_temp{i}'][k]*tens[f'error_estimate{i}'][k]
        tens[f'latent_temp{i+1}'] = torch.divide(tens[f'latent_temp{i+1}'], tens[f'regularization{i}']/float(len(img_list)))
    latent_final_estimate1 = tens[f'latent_temp{iterations}'].clone()
    for k in range(len(img_list)):
        # latent_final_estimate1 =  latent_final_estimate1 + min_value_temp[k]
        latent_final_estimate1 =  max_value_temp[k] - 2.0*latent_final_estimate1
        # print(max_value_temp[k], torch.max(latent_estimate[k]))
    latent_final_estimate1 = latent_final_estimate1[:, :, padding:-padding, padding:-padding]
    # print('final shape', latent_estimate.shape)
    return latent_final_estimate1

torch.autograd.set_detect_anomaly(True)
# path = "../cqs_psf_fixed/train/"
# para_list_file = open("../cqs_psf_fixed/meta/train00.txt",'r')
path = "../lxaset2blur_test/lxaset2blur1/"
para_list_file = open("../lxaset2blur_test/meta/train1_psf.txt",'r')
para_list = para_list_file.readlines()
para_list_file.close()
step = 0
min_max=(0, 255)
para_sys = Params()
for para_line in para_list:
    para_line_split = para_line.split(' ')
    filename = path + para_line_split[0]
    img = mmcv.imread(filename)
    img_tensor = torch.from_numpy(img.transpose(2,0,1))
    img_tensor = img_tensor.type(torch.float32)
    para_focus_split = para_line_split[1].split(',')
    para_focus = np.zeros(7)
    for i in range(7):
        para_focus[i] = np.float32(para_focus_split[i])
    # print(para_focus)
    psf_array = torch.tensor(para_focus).requires_grad_(True)
    psf_kernel, wavefront_lq, pupil_diameter_lq = get_psf(para_sys, psf_array)
    psf_list = psf_kernel.unsqueeze(0)
    img_list = img_tensor.unsqueeze(0)
    img_focus_list = rl_deconvolution_torch(img_list, psf_list)
    img_focus = img_focus_list.squeeze(0).squeeze(0)
    img_outnp = img_focus.detach().numpy()
    img_outnp = np.transpose(img_outnp, (1, 2, 0)).astype(np.uint8)
    save_path = "../lxaset2blur_test/train1_hqrl/" + para_line_split[0]
    mmcv.imwrite(img_outnp, save_path)