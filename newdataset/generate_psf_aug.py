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
# from mmcv.parallel import is_module_wrapper, MMDistributedDataParallel
# from mmcv.runner import auto_fp16
# from torchvision import transforms
# from PIL import Image
import numpy as np
# from numpy.fft import rfft2, irfft2
import scipy
from functools import reduce
# import torchaudio
# from .datasets.pipelines import ToTensor, ImageToTensor
# from torchaudio.functional import fftconvolve


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
        self.n = 1 #refraction
        self.magnification = 10.
        self.pixelsize = 7.3 # um

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
    # x = torch.tensor(2.*params.pixelsize*x/(params.size*params.magnification))
    # y = torch.tensor(2.*params.pixelsize*y/(params.size*params.magnification))
    r2 = to_radial_torch(x, y) 
    # print(x, y, r2)
    # print(cnn_pa[1], cnn_pa[2], cnn_pa[3])
    aberration = cnn_pa[4]*torch.pow(r2,2) + cnn_pa[1]*r2 + cnn_pa[2]*torch.pow((x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3])),2) + \
            cnn_pa[6] * ((x*r2)*torch.cos(cnn_pa[3]) + (y*r2)*torch.sin(cnn_pa[3])) + \
            cnn_pa[5]*(x*torch.cos(cnn_pa[3])+ y*torch.sin(cnn_pa[3]))
    wavefront = torch.exp(2*1j*np.pi*aberration)
    # print(wavefront)
    return wavefront

def get_psf(params, cnn_pa, centered = True):
    datapoints = int(params.size)
    # print(datapoints)
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
    # print(xx, yy)
    sums = torch.tensor(np.power(xx,2) + np.power(yy,2))
    # wavefront = get_wavefront(xx, yy, params)
    wavefront = get_wavefront_torch(xx, yy, params, cnn_pa)
    pixel_limit = PRw*params.size*params.pixelsize*1e-6
    # print(pixel_limit, sums)
    wavefront0 = torch.ones(wavefront.shape)
    wavefront0[sums > pixel_limit] = 0.0
    wavefront_fix = wavefront*wavefront0
    # print(wavefront, wavefront_fix, wavefront[32,32], wavefront_fix[32,32])
    # print(wavefront.sum())
    # wavefront4d = wavefront.unsqueeze(0)
    # wavefront4d = wavefront4d.unsqueeze(0)
    wavefront_padded = torch.nn.functional.pad(wavefront_fix, (padding,padding,padding,padding), "constant", 0.0)
    # wavefront_padded = np.pad(wavefront, ((padding,padding),(padding,padding)), mode='constant',constant_values=(0))
    # psf = np.power(np.abs(np.fft.fft2(wavefront_padded, norm='ortho')),2)
    # psf = np.roll(psf, center_point, axis = (0,1))
    
    psf = torch.pow(torch.abs(torch.fft.fft2(wavefront_padded, norm='ortho')),2)
    # print(center_point)
    psf_new1 = torch.roll(psf, center_point, dims=0)
    psf_new2 = torch.roll(psf_new1, center_point, dims=1)
    
    normalisation = torch.pow(torch.abs(wavefront).sum()/float(totalpoints),2)
    # normalisation = np.power(np.sum(np.abs(wavefront)) / float(totalpoints),2)
    psf_final0 = unpad(psf_new2, padding) / normalisation
    psf_final = scale_torch(torch.fliplr(psf_final0)).type(torch.float32)   
    # print(psf)
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

# def conv_fft_torch(img_list, psf_list):
#     psf_list = normalize(psf_list)
#     npad = np.max(psf_list.shape)
#     img_list = torch.nn.functional.pad(img_list, (npad,npad,npad,npad), "constant", 0.0)
#     out_list = img_list.clone()
#     for step in range(len(psf_list)):
#         psf_w = psf_list[step]
#         psf_w = psf_w.unsqueeze(0)
#         out_list[step] = torchaudio.functional.fftconvolve(img_list[step], psf_w, mode='same')
#     out_list = out_list[:, :, npad:-npad, npad:-npad]
#     return out_list        
    
def conv_with_torch(img_list, psf_list):
    psf_list = normalize(psf_list)
    npad = np.max(psf_list.shape)
    paddingsize = npad//2
    #print(paddingsize)
    img_list_pad = torch.nn.functional.pad(img_list, (paddingsize, paddingsize, paddingsize, paddingsize), "reflect")
    print(img_list_pad.shape)
    # size = np.array(np.array(img_list_pad[0, 0].shape) + np.array(psf_list[0].shape)) - 1
    # fsize = [scipy.fftpack.helper.next_fast_len(int(d)) for d in size]
    # fsize_extend = tuple(sz for sz in fsize)
    # psf_f = torch.fft.rfft2(psf_list, fsize_extend)
    # size_ext = [3]
    # size_ext = np.hstack([size_ext, size])
    # fslice = tuple([slice(0, int(sz)) for sz in size_ext])
    tensc = dict()
    # img_list_f = torch.fft.rfft2(img_list_pad, s=fsize_extend, dim=(-2,-1))
    out_list_ori = torch.zeros(img_list_pad.shape)
    out_list = torch.zeros(img_list.shape)
       # conv_new = torch.nn.Conv2d(1, 1, kernel_size = npad, padding=(paddingsize, paddingsize))
    # conv_new = torch.nn.Conv2d(1, 1, kernel_size = npad, padding=(paddingsize, paddingsize), padding_mode='reflect')
    for step in range(len(psf_list)):
        tensc[f'psf_w{step}'] = psf_list[step].clone()
        tensc[f'psf_w{step}'] = tensc[f'psf_w{step}'].unsqueeze(0)
        tensc[f'psf_w{step}'] = tensc[f'psf_w{step}'].unsqueeze(0)
    #     conv_new.weight.data = tensc[f'psf_w{step}']
    #     conv_new.bias.data.zero_()
    #     print(conv_new.weight, conv_new.bias)
        for image_channel in range(3):
            tensc[f'img_w{step}_{image_channel}'] = img_list_pad[step, image_channel]
            tensc[f'img_w{step}_{image_channel}'] = tensc[f'img_w{step}_{image_channel}'].unsqueeze(0)
            tensc[f'img_w{step}_{image_channel}'] = tensc[f'img_w{step}_{image_channel}'].unsqueeze(0)
            tensc[f'out_{step}_{image_channel}'] = torch.nn.functional.conv2d(tensc[f'img_w{step}_{image_channel}'], tensc[f'psf_w{step}'], padding=0)
            tensc[f'out_{step}_{image_channel}'] = tensc[f'out_{step}_{image_channel}'].squeeze()
            tensc[f'out_{step}_{image_channel}'] = tensc[f'out_{step}_{image_channel}'].squeeze()
            # print(tensc[f'out_{step}_{image_channel}'].shape)
            out_list[step, image_channel] = out_list[step, image_channel] + tensc[f'out_{step}_{image_channel}']
    #         # print(out_list.shape, psf_list.shape, img_res_temp.shape, img_for_conv.shape, npad)
    #         out_list[step, image_channel] = img_res_temp
    # out_list = out_list[:,:,npad:-npad, npad:-npad]
    # out_list = out_list_ori[:,:,npad:-npad, npad:-npad]
    return out_list

def rl_deconvolution_torch(img_list, psf_list, iterations=25, lbd=0.2):
    padding = np.max(psf_list[0].shape)
    # print(padding)
    if img_list.dim() == 3:
        img_list = img_list.unsqueeze(0)
    img_list_pad = torch.nn.functional.pad(img_list, (padding, padding, padding, padding), "reflect")
    min_value_temp = torch.zeros(len(img_list))
    max_value_temp = torch.zeros(len(img_list))
    for k in range(len(img_list)):
        max_value_temp[k] = max_value_temp[k] + torch.max(img_list[k])
        min_value_temp[k] = min_value_temp[k] + torch.min(img_list[k])
        # img_list_pad[k] = img_list_pad[k] - min_value_temp[k]
        img_list_pad[k] = max_value_temp[k] - img_list_pad[k]
        # print(min_value_temp[k], max_value_temp[k])
        # max_value_temp[k] = max_value_temp[k] + torch.max(img_list_pad[k])
    size = np.array(np.array(img_list_pad[0, 0].shape) + np.array(psf_list[0].shape)) - 1
    fsize = [scipy.fftpack.helper.next_fast_len(int(d)) for d in size]
    # fslice = tuple([slice(0, int(sz)) for sz in size])    
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
    #latent_final_estimate = torch.zeros(latent_estimate.shape)
    #latent_final_estimate1 = torch.zeros(latent_estimate.shape)
    # latent_final_estimateregu = torch.zeros(num_iter1)
    #error_estimate = torch.zeros(latent_estimate.shape)
    #error_estimate1 = torch.zeros(latent_estimate.shape)
    #estimate_convolved = torch.zeros(latent_estimate.shape)
    #estimate_convolved1= torch.zeros(latent_estimate.shape)
    #relative_blur = torch.zeros(latent_estimate.shape)
    #relative_blur1 = torch.zeros(latent_estimate.shape)
    #regularization = torch.ones(img_list_pad[0].shape)
    #regularization1 = torch.ones(img_list_pad[0].shape)
    # error_estimate = img_list_pad.clone()
    # fsize_ext = [len(psf_list)]
    size_ext = [3]
    #fsize_ext = np.hstack([fsize_ext, fsize])
    fsize_extend = tuple(sz for sz in fsize)
    size_ext = np.hstack([size_ext, size])
    # fsize_ext2 = np.hstack([size_ext, fsize])
    # fsize_extend2 = tuple(sz for sz in fsize_ext2)
    fslice = tuple([slice(0, int(sz)) for sz in size_ext])
    # print(fsize_extend, psf_list.shape)
    psf_f = torch.fft.rfft2(psf_list, fsize_extend)
    # print(psf_f.shape)
    _psf_flipped = torch.flip(psf_list, dims=[1])
    _psf_flipped = torch.flip(_psf_flipped, dims=[2])
    psf_flipped_f = torch.fft.rfft2(_psf_flipped, s=fsize_extend)
    tens['latent_temp0'] = latent_estimate.clone()
    # for k in range(len(img_list)):
    #     latent_temp0 = torch.fft.rfft2(latent_temp[k],s=fsize_extend, dim=(-2,-1))
    #     estimate_convolved[k] = estimate_convolved[k] + _centered(torch.fft.irfft2(torch.mul(psf_f[k],latent_temp0))[fslice], img_list_pad[k].shape)
    #     # print(estimate_convolved[k])
    #     relative_blur[k] = relative_blur[k] + div0_torch(img_list_pad[k], estimate_convolved[k])
    #     error_estimate[k] = error_estimate[k] + _centered(torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(relative_blur[k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
    #     for k1 in range(3):
    #         regularization[k1] = regularization[k1] + 1.0 - (lbd*div_torch(latent_temp[k,k1]/torch.linalg.norm(latent_temp[k,k1], ord=1)))
    #     latent_final_estimate[k] = latent_final_estimate[k] + latent_temp[k]*error_estimate[k]
    #     # print(latent_final_estimate[k])
    # latent_final_estimate = torch.divide(latent_final_estimate.clone(), regularization/float(len(img_list)))
    # latent_tempnew = latent_final_estimate.clone()
    # for k in range(len(img_list)):
    #     latent_temp1 = torch.fft.rfft2(latent_tempnew[k],s=fsize_extend, dim=(-2,-1))
    #     estimate_convolved1[k] = estimate_convolved1[k] + _centered(torch.fft.irfft2(torch.mul(psf_f[k],latent_temp1))[fslice], img_list_pad[k].shape)
    #     relative_blur1[k] = relative_blur1[k] + div0_torch(img_list_pad[k], estimate_convolved1[k])
    #     error_estimate1[k] = error_estimate1[k] + _centered(torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(relative_blur1[k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
    #     for k1 in range(3):
    #         regularization1[k1] = regularization[k1] + 1.0 - (lbd*div_torch(latent_temp[k,k1]/torch.linalg.norm(latent_temp[k,k1], ord=1)))
    #     latent_final_estimate1[k] = latent_final_estimate1[k] + latent_tempnew[k]*error_estimate1[k]
    # latent_final_estimate1 = torch.divide(latent_final_estimate1.clone(), regularization1/float(len(img_list)))
    for i in range(iterations):
        for k in range(len(img_list)):            
            # latent_temp0 = torch.fft.rfft2(latent_temp[k],s=fsize_extend, dim=(-2,-1))
            # print(latent_temp.dtype)
            # latent_final_estimate = latent_final_estimate.clone()
            tens[f'estimate_convolved{i}'][k] = tens[f'estimate_convolved{i}'][k] + _centered(torch.fft.irfft2(torch.mul(psf_f[k],torch.fft.rfft2(tens[f'latent_temp{i}'][k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            # print(estimate_convolved_c.dtype)
            # estimate_convolved = torch.view_as_real(estimate_convolved_c).real
            #estimate_convolved = torch.fft.irfft2(psf_f[k]*latent_temp)[fslice].real
            # estimate_convolved = _centered(estimate_convolved, img_list_pad[k].shape)
            #print(estimate_convolved.shape)
            tens[f'relative_blur{i}'][k] = tens[f'relative_blur{i}'][k] + div0_torch(img_list_pad[k], tens[f'estimate_convolved{i}'][k])
            # error_estimate_c = torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(relative_blur,s=fsize_extend, dim=(-2,-1))))[fslice]
            # error_estimate[k] = torch.view_as_real(error_estimate_c).real
            tens[f'error_estimate{i}'][k] = tens[f'error_estimate{i}'][k] + _centered(torch.fft.irfft2(torch.mul(psf_flipped_f[k],torch.fft.rfft2(tens[f'relative_blur{i}'][k],s=fsize_extend, dim=(-2,-1))))[fslice], img_list_pad[k].shape)
            for k1 in range(3):
                tens[f'regularization{i}'][k1] = tens[f'regularization{i}'][k1] + 1.0 - (lbd*div_torch(tens[f'latent_temp{i}'][k,k1]/torch.linalg.norm(tens[f'latent_temp{i}'][k,k1], ord=1)))
            tens[f'latent_temp{i+1}'][k] = tens[f'latent_temp{i+1}'][k] + tens[f'latent_temp{i}'][k]*tens[f'error_estimate{i}'][k]
            # latent_final_estimate = latent_estimate[k]*error_estimate[k]
            # if k==0:
            #     latent_final_estimate = latent_estimate[k]*error_estimate[k]
            #     latent_final_estimate = latent_final_estimate.unsqueeze(0)
            # else:
            #     latent_final_estimatenew = latent_estimate[k]*error_estimate[k]
            #     latent_final_estimatenew = latent_final_estimatenew.unsqueeze(0)
            #     latent_final_estimate = torch.cat(latent_final_estimate, latent_final_estimatenew)
        # latent_final_estimate = latent_estimate*error_estimate  
        # for k in range(len(img_list)):
        #     latent_estimate[k] = torch.divide(latent_estimate[k], regularization/float(len(img_list)))
        # latent_estimate = torch.divide(latent_estimate, regularization/float(len(img_list)))
        tens[f'latent_temp{i+1}'] = torch.divide(tens[f'latent_temp{i+1}'], tens[f'regularization{i}']/float(len(img_list)))
    latent_final_estimate1 = tens[f'latent_temp{iterations}'].clone()
    # latent_final_estimate2 = latent_final_estimate1.clone()
    # print(latent_final_estimate)
    for k in range(len(img_list)):
        # latent_final_estimate1 =  latent_final_estimate1 + min_value_temp[k]
        latent_final_estimate1 =  max_value_temp[k] - 2.0*latent_final_estimate1
        # print(max_value_temp[k], torch.max(latent_estimate[k]))
    latent_final_estimate1 = latent_final_estimate1[:, :, padding:-padding, padding:-padding]
    # print('final shape', latent_estimate.shape)
    return latent_final_estimate1

path = "./val1_gt"
path1 = "./val2"
torch.autograd.set_detect_anomaly(True)
# path = "../../../fudan2023_2/202401/train_data0119/Normal1"
# path1 = "../../../fudan2023_2/202401/train_data0119/Normal1_out"
tifdatadirs = os.listdir(path)
#f1 = open("F://fudan2024_1/202403/idrop_lqsy/psfarray32tmp.txt",'a')
step = 0
min_max=(0, 255)
para_sys = Params()

f2 = open("./test_psf_ori1.txt", 'r')
psflines = f2.readlines()
f2.close()
print(len(psflines))
psf_list_focus = np.zeros((63,63))
for i in range(len(psflines)-1):
    test_pst_line = psflines[i][2:]
    test_psf_line = test_pst_line.split('  ')
    for j in range(len(test_psf_line)-1):
        psf_list_focus[i,j] = float(test_psf_line[j+1][:-1])
# print(psf_list)
psf_list_focus = torch.tensor(psf_list_focus)
psf_list_focus = psf_list_focus.unsqueeze(0)

# torch.autograd.set_detect_anomaly(True)
# psf_list_ori = torch.zeros([1,np.int(para_sys.size),np.int(para_sys.size)])
f1 = open("./val2.txt",'a')
for tifdir in tifdatadirs:
    if os.path.isdir(path + '/' + tifdir):
        tifdatanames = os.listdir(path + '/'+ tifdir) 
        for tiffile in tifdatanames:
            filename = path + '/' + tifdir + '/' + tiffile
            tifname = tiffile.split('.')
            tifname0 = tifname[0]
            img = mmcv.imread(filename)
            img_size = 255.0*img.size
            img_var = np.var(img)/img_size
            img_ratio = np.sum(img)/img_size
            if (img_ratio > 0.0003 and img_var > 5e-08):
                para_valid = 0
            else:
                para_valid = 1
            img_tensor = torch.from_numpy(img.transpose(2,0,1))
            img_tensor = img_tensor.type(torch.float32)    
            for img_idx in range(5):
                para_focus = 7.64*np.random.rand()-3.82
                ast_split = np.random.rand()
                angle_split = np.random.rand()
                if ast_split<=0.25:
                    para_sph = 0.0271 - (0.014*para_focus/0.191) + 0.001*np.random.rand()
                    para_ast = 0.0
                    para_angle = 0.0
                    para_coma = 0.0
                    para_tilt = 0.0
                elif ast_split<=0.75:
                    para_sph = 0.0071 - (0.0115*(para_focus-0.0312)/0.189) + 0.001*np.random.rand()
                    para_ast = 0.0664 - (0.0342*(para_focus-0.0312)/0.189) + 0.001*np.random.rand()
                    para_coma = 0.1569 - (0.0015*(para_focus-0.0312)/0.189) + 0.0001*np.random.rand()
                    para_tilt = 0.001727 + 0.00008*np.random.rand() 
                    if angle_split<=0.25:
                        para_angle = 0.0
                    elif angle_split<=0.5:
                        para_angle = 0.5*np.pi
                    elif angle_split<=0.75:
                        para_angle = np.pi
                    else:
                        para_angle = -0.5*np.pi
                else:
                    para_sph = -0.0815 - (0.014*(para_focus-0.0539)/0.188) + 0.001*np.random.rand()
                    para_ast = 0.12385 - (0.0342*(para_focus-0.0539)/0.188) + 0.001*np.random.rand()
                    para_coma = 0.2076 - (0.0009*(para_focus-0.0539)/0.188) + 0.0001*np.random.rand()
                    para_tilt = 0.001109 + 0.00016*np.random.rand()
                    if angle_split<=0.25:
                        para_angle = 0.25*np.pi
                    elif angle_split<=0.5:
                        para_angle = 0.75*np.pi
                    elif angle_split<=0.75:
                        para_angle = -0.25*np.pi
                    else:
                        para_angle = -0.75*np.pi
            # para_ast = 0.0
            # para_ast_angle = 0.0
            # para_temp = np.random.rand()
                psf_array = torch.tensor([para_valid, para_focus, para_ast, para_angle, para_sph, para_tilt, para_coma]).requires_grad_(True)
                psf_kernel, wavefront_lq, pupil_diameter_lq = get_psf(para_sys, psf_array)
                psf_list = psf_kernel.unsqueeze(0)
            #psf_mean = psf_list.mean()
            #psf_mean.backward()
            #print(psf_mean, psf_array.grad)
            # psf_list_dif = psf_list-psf_list_ori
            #print(torch.max(psf_list_dif), torch.min(psf_list_dif))
            # dif_all = psf_list_dif.sum()
            # psf_list_ori = psf_list
                img_list = img_tensor.unsqueeze(0)
            # psf_list = psf_list.requires_grad_(True)
                img_focus_list = rl_deconvolution_torch(img_list, psf_list_focus)
                img_out_list = conv_with_torch(img_focus_list, psf_list)
            # img_out_list = conv_fft_torch(img_list, psf_list)
                img_dc_list = rl_deconvolution_torch(img_out_list, psf_list)
                # img_dc_list = rl_deconvolution_torch(img_list, psf_list)
                # img_dc_list_sum = img_dc_list.mean()
                # img_dc_list_sum.backward()
                # print(img_dc_list_sum, psf_array.grad)
                img_out = img_out_list.squeeze(0).squeeze(0)
                img_outnp = img_out.detach().numpy()
                img_outnp = np.transpose(img_outnp, (1, 2, 0)).astype(np.uint8)
                tiffilenew = tiffile.replace('.tif', '_'+str(img_idx)+'.tif')
                # tiffilenew = tiffile
                save_path = path1 + '/' + tifdir + '/' + tiffilenew
            # print(psf_array)
                mmcv.imwrite(img_outnp, save_path)
                img_out2 = img_dc_list.squeeze(0).squeeze(0)
            
                img_dcnp = img_out2.detach().numpy()
                img_dcnp = np.transpose(img_dcnp, (1, 2, 0)).astype(np.uint8)
                save_path2 = path1 + '/' + tifdir + 'rl/' + tiffilenew
                # img_out3 = img_focus_list.squeeze(0).squeeze(0)
            
                # img_fcnp = img_out3.detach().numpy()
                # img_fcnp = np.transpose(img_fcnp, (1, 2, 0)).astype(np.uint8)
                # save_path3 = path1 + '/' + tifdir + 'focus/' + tiffilenew
            # print(psf_array) 
                # psfarray = np.array2string(psf_array.detach().numpy())
                f1.write(tifdir + '/' + tiffilenew + ' '+ str(para_valid) + ',' + str(para_focus)+','+str(para_ast)+','+str(para_angle)+','+str(para_sph)+','+str(para_tilt)+','+str(para_coma)+'\n')
            # dif_all = np.array2string(dif_all.detach().numpy())
            #f1.write(tiffile + ' ' + psfarray +' ' + dif_all + '\n')
                mmcv.imwrite(img_dcnp, save_path2)
                # mmcv.imwrite(img_fcnp, save_path3)
f1.close()