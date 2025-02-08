_base_ = [
    '../_base_/models/vgg19.py',
    '../_base_/datasets/mydataset.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
optimizer = dict(lr=0.01)