_base_ = ['../_base_/models/resnest50.py', '../_base_/default_runtime.py']

# dataset_type = 'MyDataset'
# classes = ['normal', 'cancer']
dataset_type = 'MultiLabelPSFDataset'
classes = ['valid', 'focus', 'ast', 'angle', 'sph', 'tilt', 'coma']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='RandomResizedCrop', size=256),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='data/lxaset2blur_test/lxaset2blur1',
        ann_file='data/lxaset2blur_test/meta/lxaset2blur_train1.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/lxaset2blur_test/val1',
        ann_file='data/lxaset2blur_test/meta/lxaset2blur_val1.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/newbeads_cont4/cont4',
        ann_file='data/newbeads_cont4/meta/cont4.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)

# optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.8,
#     momentum=0.9,
#     weight_decay=1e-4,
#     paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)

# learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=5,
#     warmup_ratio=1e-6,
#     warmup_by_epoch=True)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[50, 100], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=150)

# evaluation = dict(interval=1, metric='accuracy')
evaluation = dict(interval=1, metric=['mAP'])