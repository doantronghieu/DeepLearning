# Project configuration
project_name = 'MODEL DATASET'
work_dir = './exp/MODEL_DATASET'
seed = 42

# Model configuration
model = dict(
    type='MyModel',
    init_args=dict(),
    pretrained=None,
    backbone=dict(type='ResNet', init_args=dict(depth=50)),
    neck=dict(type='FPN', init_args=dict(in_channels=[256, 512, 1024, 2048], out_channels=256)),
    head=dict(type='RetinaHead', init_args=dict(num_classes=80)),
    data_preprocessor=dict(),
    feature_flags=dict(),
    init_cfg=None
)

# Dataset configuration
# dataset = dict(
#     train=dict(
#         type='CocoDataset',
#         data_root='data/coco/',
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='train2017/'),
#         pipeline=[...],
#         test_mode=False
#     ),
#     val=dict(
#         type='CocoDataset',
#         data_root='data/coco/',
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
#         pipeline=[...],
#         test_mode=True
#     )
# )

# DataLoader configuration
# https://pytorch.org/docs/stable/data.html
train_dataloader = dict(
    dataset = dict(
        type='MyDataset', is_train=True, size=10000,
    ),
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    pin_memory=True,
    persistent_workers=True
)
val_dataloader = dict(
    dataset = dict(
        type='MyDataset', is_train=True, size=1000,
    ),
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    pin_memory=True,
    persistent_workers=True
)

train_cfg = dict(
    by_epoch=True, # display in epoch number instead of iterations
    max_epochs=10,
    val_begin=2, # start validation from the 2nd epoch
    val_interval=1, # do validation every 1 epoch
)
val_cfg = dict(),

# Evaluation configuration
val_evaluator = dict(type='Accuracy')

evaluation = dict(
    interval=1,
    metrics=[dict(type='Accuracy')]
)

# Optimizer configuration
# https://pytorch.org/docs/stable/optim.html
optim_wrapper = dict(
    optimizer = dict(
        type='Adam',
        lr=0.001,
        weight_decay=0.0001,
        momentum=0.9
    )
)

# Learning rate scheduler configuration
param_schedulers = dict(
    type='LinearLR', # MultiStepLR, LinearLR
    by_epoch=False,
    begin=0,
    end=500,
    param_name='lr',
    init_args=dict(start_factor=0.001),
    gamma=0.1,
)

# Runner configuration
runner = dict(
    type='EpochBasedRunner',
    max_epochs=12,
    val_interval=1,
    log_interval=50
)

# Hooks configuration
hooks = [
    dict(type='IterTimerHook'),
    dict(type='LoggerHook', priority=None, init_args=dict(interval=50)),
    dict(type='ParamSchedulerHook'),
    dict(type='CheckpointHook', priority=None, init_args=dict(interval=1)),
    dict(type='DistSamplerSeedHook'),
    dict(type='DetVisualizationHook')
]

# Feature flags configuration
feature_flags = dict(
    use_amp=False,
    use_gradient_accumulation=False,
    use_multi_optimizer=False,
    use_parameter_scheduling=False,
    use_custom_hooks=False
)

# Checkpoint configuration
checkpoint = dict(
    interval=1
)

# Visualizer configuration
visualizer = dict(
    type='LocalVisBackend',
    save_dir='visual_results',
    vis_backends=[dict(type='LocalVisBackend')]
)

# Resume and load from configuration
resume_from = None
load_from = None

# CUDA configuration
cudnn_benchmark = False
mp_start_method = 'fork'

# Distributed configuration
dist_params = dict(
    backend='nccl'
)

# Log configurations
log = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
log_level = 'INFO'
default_scope = 'mmengine'
log_processor = dict(
    window_size=10,
    by_epoch=True,
    custom_cfg=None,
    num_digits=4
)

# Default hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Launcher configuration
launcher = 'none'
distributed = False

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False, # whether enable cudnn_benchmark
    mp_cfg=dict( # multiprocessing configs
        mp_start_method='fork',
        opencv_num_threads=0
    ),
    dist_cfg=dict(
        backend='nccl' # distributed communication backend
    )
)
