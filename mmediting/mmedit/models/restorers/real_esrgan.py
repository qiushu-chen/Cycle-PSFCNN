# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
from copy import deepcopy

import mmcv
import torch
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper

from mmedit.core import tensor2img
from ..common import set_requires_grad
from ..registry import MODELS
from .srgan import SRGAN


@MODELS.register_module()
class RealESRGAN(SRGAN):
    """Real-ESRGAN model for single image super-resolution.

    Ref:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
    Synthetic Data, 2021.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        is_use_sharpened_gt_in_pixel (bool, optional): Whether to use the image
            sharpened by unsharp masking as the GT for pixel loss.
            Default: False.
        is_use_sharpened_gt_in_percep (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for perceptual loss.
            Default: False.
        is_use_sharpened_gt_in_gan (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for adversarial loss.
            Default: False.
        is_use_ema (bool, optional): When to apply exponential moving average
            on the network weights. Default: True.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 pixel_loss2=None,
                 pixel_loss3=None,
                 perceptual_loss=None,
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator, discriminator, gan_loss, pixel_loss,
                         perceptual_loss, train_cfg, test_cfg, pretrained)
        from ..builder import build_backbone, build_component, build_loss
        self.pixel_loss2 = build_loss(pixel_loss2) if pixel_loss2 else None
        self.pixel_loss3 = build_loss(pixel_loss3) if pixel_loss3 else None

        self.is_use_sharpened_gt_in_pixel = is_use_sharpened_gt_in_pixel
        self.is_use_sharpened_gt_in_percep = is_use_sharpened_gt_in_percep
        self.is_use_sharpened_gt_in_gan = is_use_sharpened_gt_in_gan

        self.is_use_ema = is_use_ema
        if is_use_ema:
            self.generator_ema = deepcopy(self.generator)
        else:
            self.generator_ema = None

        del self.step_counter
        self.register_buffer('step_counter', torch.zeros(1))

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # during initialization, load weights from the ema model
        # if (self.step_counter == self.start_iter
        #         and self.generator_ema is not None):
        #     if is_module_wrapper(self.generator):
        #         self.generator.module.load_state_dict(
        #             self.generator_ema.module.state_dict())
        #     else:
        #         self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq_ori = data_batch['lq']
        gt = data_batch['gt']
        lq = F.avg_pool2d(lq_ori, kernel_size=3, stride=2, padding=1)
        lq = F.avg_pool2d(lq, kernel_size=3, stride=2, padding=1)
        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        # if self.is_use_sharpened_gt_in_pixel:
        #     gt_pixel = data_batch['gt_unsharp']
        # if self.is_use_sharpened_gt_in_percep:
        #     gt_percep = data_batch['gt_unsharp']
        # if self.is_use_sharpened_gt_in_gan:
        #     gt_gan = data_batch['gt_unsharp']

        # generator
        fake_g_output = self.generator(lq)
        fake_g_output_ema = self.generator_ema(lq)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel)
            if self.pixel_loss2:
                #losses['loss_pix'] = 0.5*(self.pixel_loss(fake_g_output, gt_pixel) + self.pixel_loss2(fake_g_output, gt_pixel))
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel) + self.pixel_loss2(fake_g_output, fake_g_output_ema, gt_pixel)+ self.pixel_loss3(fake_g_output, gt_pixel)
            if self.perceptual_loss:
                loss_percep, loss_style = self.perceptual_loss(
                    fake_g_output, gt_percep)
                if loss_percep is not None:
                    losses['loss_perceptual'] = loss_percep
                if loss_style is not None:
                    losses['loss_style'] = loss_style

            # gan loss for generator
            if self.gan_loss:
                real_d_pred = self.discriminator(gt).detach()  #added to here, real images should be taken into consideration
                fake_g_pred = self.discriminator(fake_g_output)
                # losses['loss_gan'] = self.gan_loss(
                #     fake_g_pred, target_is_real=True, is_disc=False)
                loss_gan_fake = self.gan_loss(
                    fake_g_pred - torch.mean(real_d_pred),
                    target_is_real=True,
                    is_disc=False)
                loss_gan_real = self.gan_loss(
                    real_d_pred - torch.mean(fake_g_pred),
                    target_is_real=False,
                    is_disc=False)
                losses['loss_gan'] = (loss_gan_fake + loss_gan_real) / 2

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            fake_d_pred = self.discriminator(fake_g_output).detach()  #added
            real_d_pred = self.discriminator(gt_gan)
            # loss_d_real = self.gan_loss(
            #     real_d_pred, target_is_real=True, is_disc=True)
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
            fake_d_pred = self.discriminator(fake_g_output.detach())
            # loss_d_fake = self.gan_loss(
            #     fake_d_pred, target_is_real=False, is_disc=True)
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
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        # _model = self.generator_ema if self.is_use_ema else self.generator
        _model = self.generator
        lq = F.avg_pool2d(lq, kernel_size=3, stride=2, padding=1)
        lq = F.avg_pool2d(lq, kernel_size=3, stride=2, padding=1)
        output = _model(lq)

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
