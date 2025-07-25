custom_imports = dict(
    imports=['inter2former', 'mmdet.models'],
    allow_failed_imports=False
)

randomness = dict(seed=42)
train_cfg = dict(type='IterBasedTrainLoop',
                 max_iters=40000,
                 val_interval=10000)
val_cfg = None
test_cfg = None
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = None
log_processor = dict(window_size=1000, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# log time
custom_hooks = [dict(type='SimpleTimeLoggerHook')]
