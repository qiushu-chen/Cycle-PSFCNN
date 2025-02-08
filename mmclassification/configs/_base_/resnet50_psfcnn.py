# model settings
model = dict(
    # type='ImageClassifier',
    type='ImageFFTClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=5,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelL2PSFHead',
        num_classes=7,
        in_channels=2048,
        loss=dict(type='PSFL2Loss', loss_weight=1.0)
    ))
