# model settings
model = dict(
    # type='ImageClassifier',
    type='ImageFFTClassifier',
    backbone=dict(
        type='ResNeSt',
        depth=50,
        in_channels=5,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        # type='LinearClsHead',
        type='MultiLabelL2PSFHead',
        num_classes=7,
        in_channels=2048,
        loss=dict(type='PSFL2Loss', loss_weight=1.0)))
        # loss=dict(
        #     type='LabelSmoothLoss',
        #     label_smooth_val=0.1,
        #     num_classes=5,
        #     reduction='mean',
        #     loss_weight=1.0),
        # topk=(1, 5),
        # cal_acc=False))