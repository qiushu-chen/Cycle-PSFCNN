# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 100], gamma=0.2)
runner = dict(type='EpochBasedRunner', max_epochs=150)
