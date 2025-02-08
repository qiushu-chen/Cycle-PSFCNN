exp_name = 'realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost'
# custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

scale = 1
#gt_crop_size = 400

# model settings
model = dict(
    type='RealESRGAN',
    generator=dict(
#         type='RRDBNet',
        type='RRDABNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=13,
        # num_blocks=23,
        growth_channels=32,
        upscale_factor=scale),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    # task_classifier_backbone=dict(
    #     type='mmcls.ResNeSt',
    #     depth=50,
    #     in_channels=5,
    #     num_stages=4,
    #     out_indices=(3, ),
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained', 
            # checkpoint='../editing_temp/checkpoint/resnest_cqsfft1202_epoch_300.pth'),
    #         checkpoint='../mmclassification/work_dirs/resnest50_withfft1202/epoch_300.pth'),
    #     ),
    # task_classifier_neck=dict(type='mmcls.GlobalAveragePooling'),
    # task_classifier_head=dict(
    #     type='mmcls.LinearClsHead',
    #     num_classes=5,
    #     in_channels=2048,
        # loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    #     loss=dict(
    #         type='mmcls.LabelSmoothLoss',
    #         label_smooth_val=0.1,
    #         mode = 'original',
    #         num_classes=6,
    #         reduction='mean',
    #         loss_weight=1.0,
    #         ),
    #     init_cfg=dict(
    #         checkpoint='../editing_temp/checkpoint/resnest_cqsfft1202_epoch_300.pth'),
            # checkpoint='../mmclassification/work_dirs/resnest50_withfft1202/epoch_300.pth'),
    # ),
    pixel_loss=dict(type='MSELoss', loss_weight=50.0, reduction='mean'),
    pixel_loss2=dict(type='ArtifactsLoss', loss_weight=10.0),
    pixel_loss3=dict(type='SSIMLoss', loss_weight=1.0),
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
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        loss_weight=2e-1,
        real_label_val=1.0,
        fake_label_val=-1.0),
    is_use_sharpened_gt_in_pixel=False,
    is_use_sharpened_gt_in_percep=False,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,
)
# model training and testing settings
train_cfg = dict(start_iter=1000)
test_cfg = dict(metrics=['PSNR', 'SSIM', 'MAE'], crop_border=0)

# dataset settings
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    # dict(
    #     type='Crop',
    #     keys=['gt','lq'],
    #     crop_size=(gt_crop_size, gt_crop_size),
    #     random_crop=True),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    #dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    # dict(
    #     type='RandomResize',
    #     params=dict(
    #         resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
    #         resize_scale=[0.15, 1.5],
    #         resize_opt=['bilinear', 'area', 'bicubic'],
    #         resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
    #     keys=['lq'],
    # ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    # dict(
    #     type='RandomJPEGCompression',
    #     params=dict(quality=[30, 95]),
    #     keys=['lq']),
    dict(
        type='RandomBlur',
        params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2]),
        keys=['lq'],
    ),
    # dict(
    #     type='RandomResize',
    #     params=dict(
    #         resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
    #         resize_scale=[0.3, 1.2],
    #         resize_opt=['bilinear', 'area', 'bicubic'],
    #         resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0]),
    #     keys=['lq'],
    # ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4),
        keys=['lq'],
    ),
    dict(
        type='DegradationsWithShuffle',
        degradations=[
            dict(
                type='RandomJPEGCompression',
                params=dict(quality=[5, 50]),
            ),
     
            [
     #            dict(
     #                type='RandomResize',
     #                params=dict(
     #                    target_size=(gt_crop_size // scale,
     #                                 gt_crop_size // scale),
     #                    resize_opt=['bilinear', 'area', 'bicubic'],
     #                    resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
     #            ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416]),
                ),
            ]
        ],
        keys=['lq'],
    ),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='Quantize', keys=['lq']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    #dict(type='ImageToTensor', keys=['lq', 'gt', 'gt_unsharp']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    #dict(
    #    type='Collect', keys=['lq', 'gt', 'gt_unsharp'], meta_keys=['gt_path'])
    dict(
        type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path']),
]
test_pipeline = val_pipeline

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
            lq_folder='./data/cqs_psf_fixed/train_rl',
            gt_folder='./data/cqs_psf_fixed/train_gt',
            ann_file='./data/cqs_psf_fixed/meta/train_final.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='./data/cqs_psf_fixed/val_backup',
        gt_folder='./data/cqs_psf_fixed/val_gt',
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='./data/cqs_psf_fixed/val_backup',
        gt_folder='./data/cqs_psf_fixed/val_gt',
        # gt_folder=None,
        pipeline=val_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=4e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=4e-4, betas=(0.9, 0.99)))
    # generator=dict(type='Adam', lr=2.5e-5, betas=(0.9, 0.99)),
    # discriminator=dict(type='Adam', lr=2.5e-5, betas=(0.9, 0.99)))

# learning policy
# total_iters = 400000
# lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=1)
total_iters = 60000
# lr_config = dict(policy='Step', by_epoch=False, step=[8000, 24000], gamma=0.25)
lr_config = dict(policy='Step', by_epoch=False, step=[8000, 24000], gamma=0.25)

# checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
# log_config = dict(
#     interval=100,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='TensorboardLoggerHook'),
#     ])
# visual_config = None
checkpoint_config = dict(interval=4000, save_optimizer=True, by_epoch=False)

evaluation = dict(interval=2000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
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
work_dir = f'./tutorial_exps/real_esrgan_1019'
# load_from = None
load_from = './tutorial_exps/real_esrgan_0913/iter_20000.pth'  # noqa
# resume_from = './tutorial_exps/real_esrgan_1002_1/iter_24000.pth'
resume_from = None
workflow = [('train', 1)]
