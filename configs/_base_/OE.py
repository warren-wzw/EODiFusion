# dataset settings
H,W= 480,640
data_root = f'/home/BlueDisk/Dataset/FusionDataset/RGBT/Total/'
test_root=f"/home/BlueDisk/Dataset/FusionDataset/RGBT/OverExposure/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (W//2, H)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadIrFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(W, H), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),#deepcopy
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','ir','gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadIrFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(W, H),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CityscapesDataset',
        data_root=data_root,
        img_dir='train/overexposure',
        ir_dir='train/ir',
        ann_dir='train/mask',
        pipeline=train_pipeline),
    val=dict(
        type='CityscapesDataset',
        data_root=test_root,
        img_dir='./test/vi',
        ir_dir='./test/ir',
        ann_dir='test/Segmentation_labels',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root=test_root,
        img_dir='./test/vi',
        ir_dir='./test/ir',
        ann_dir='./test/Segmentation_labels',
        pipeline=test_pipeline
    ))
