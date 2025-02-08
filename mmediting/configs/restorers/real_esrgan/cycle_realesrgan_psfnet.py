# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:11:42 2024

@author: cqsfdu
"""

exp_name = 'realesrgan_psf_rrdab'
scale = 1

model = dict(
    type='PSFCycleGAN',
    generator=dict(
#         type='RRDBNet',
        type='RRDABNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=13,
        growth_channels=32,
        upscale_factor=scale),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    psfcnn=dict(
        # type = 'mmcls.ResNetWithFC',
        type = 'mmcls.ResNeStWithFC',
        depth=50,
        num_classes=7,
        in_channels=5,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained', 
            # checkpoint='../editing_temp/checkpoint/resnest_cqsfft1202_epoch_300.pth'),
        #     checkpoint='../mmclassification/work_dirs/resnest50_withfft1202/epoch_300.pth'),
        ),
    # psfcnnneck=dict(type='mmcls.GlobalAveragePooling'),
    psfcnnhead=dict(
        type='mmcls.MultiLabelL2PSFHead_WithoutFC',
        num_classes=7,
        in_channels=2048,
        loss=dict(type='mmcls.PSFL2Loss',
                     reduction='none',
                     loss_weight=1.0),
        init_cfg=dict(type='Normal', layer='Linear', std=0.01),
        ),
    pixel_loss=dict(type='MSELoss', loss_weight=10.0, reduction='mean'),
    # pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    pixel_loss2=dict(type='SSIMLoss', loss_weight=1.0),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        loss_weight=2e-1,
        real_label_val=1.0,
        fake_label_val=-1.0),
    # perceptual_loss=None,
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    cycle_loss=dict(type='MSELoss', loss_weight=10.0, reduction='mean'),
    cycle_loss2=dict(type='SSIMLoss', loss_weight=1.0),
    # is_use_sharpened_gt_in_pixel=False,
    # is_use_sharpened_gt_in_percep=False,
    # is_use_sharpened_gt_in_gan=False,
    # is_use_ema=False,
)

# model training and testing settings
train_cfg = dict(start_iter=2000)
test_cfg = dict(metrics=['PSNR', 'SSIM', 'MAE'], crop_border=0)

train_dataset_type = 'SRMultiAnnotationDataset'
val_dataset_type = 'SRFolderDataset'
# val_dataset_type = 'SRMultiAnnotationDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb',
        flag='unchanged'),
    dict(type='CopyValues', src_keys=['lq'], dst_keys=['lq2']),
    # dict(type='RandomResizedCrop', keys=['lq2'], crop_size=256),
    dict(
        type='Normalize',
        keys=['lq2'],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(
    #     type='Normalize',
    #     keys=['lq', 'gt'],
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'lq2', 'gt']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['lq', 'lq2', 'gt', 'gt_label'], meta_keys=['lq_path', 'gt_path', 'gt_label'])
    ]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb',
        flag='unchanged'),
    dict(type='CopyValues', src_keys=['lq'], dst_keys=['lq2']),
    dict(type='Resize', keys=['lq2'], scale=(256,256)),
    dict(type='Crop', keys=['lq2'], crop_size=(256,256)),
    dict(
        type='Normalize',
        keys=['lq2'],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'lq2', 'gt']),
    dict(type='Collect', keys=['lq', 'lq2', 'gt'], meta_keys=['lq_path', 'gt_path']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='CopyValues', src_keys=['lq'], dst_keys=['lq2']),
    dict(type='Resize', keys=['lq2'], scale=(256,256)),
    dict(type='Crop', keys=['lq2'], crop_size=(256,256)),
    dict(
        type='Normalize',
        keys=['lq2'],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='ImageToTensor', keys=['lq', 'lq2']),
    dict(type='Collect', keys=['lq', 'lq2'], meta_keys=['lq_path']),
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(
        samples_per_gpu=8, drop_last=True,
        persistent_workers=False),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='./data/lxaset2blur_test/lxaset2blur1',
            gt_folder='./data/lxaset2blur_test/lxaset2blur1_gt',
            ann_file='./data/lxaset2blur_test/meta/lxaset2blur_train1.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/lxaset2blur_test/val1',
        gt_folder='./data/lxaset2blur_test/val1_gt',
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='./data/newbeads_cont4/group3',
        # lq_folder='./data/lxaset2blur_test/cont3_seg_res2',
        # gt_folder='./data/lxaset2blur_test/cont3_seg_res2',
        gt_folder='./data/newbeads_cont4/group3',
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=4e-5, betas=(0.9, 0.999)),
    discriminator=dict(type='Adam', lr=4e-5, betas=(0.9, 0.999)),
    psfcnn=dict(type='SGD', lr=4e-6, momentum=0.9, weight_decay=0.0001))
    # psfcnnhead=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[25000, 25000, 25000, 25000],
    restart_weights=[1, 1, 1, 1],
    min_lr=1e-7)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)

evaluation = dict(interval=200, save_image=False, gpu_collect=True)
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./tutorial_exps/cycle_esrgan_1025'
load_from = f'./tutorial_exps/cycle_esrgan_1025/cycle_1025_fix.pth'  # noqa
# load_from = None
resume_from = None
workflow = [('train', 1)]