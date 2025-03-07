custom_hooks = [
    dict(type='SimpleTimeLoggerHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'inter2former',
        'mmdet.models',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=0,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=256),
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=200, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depth=12,
        downsample_sizes=[
            256,
        ],
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dim=768,
        final_embed_dim=256,
        img_size=224,
        in_dim=3,
        mlp_ratio=4.0,
        num_heads=12,
        out_indices=(
            2,
            5,
            8,
            11,
        ),
        patch_size=16,
        type='HRSAMViTNoSSM',
        use_checkpoint=False,
        window_size=16),
    decode_head=dict(
        embed_dim=256,
        expand_ratio=1.4,
        loss_decode=[
            dict(loss_weight=1.0, type='NormalizedFocalLoss'),
            dict(type='BinaryIoU'),
        ],
        num_upsamplers=4,
        threshold=-2.0,
        type='DynamicLocalUpsampling'),
    edge_detector=dict(depth=4, strides=(
        2,
        2,
        2,
        2,
    ), type='CannyNet'),
    freeze_backbone=False,
    freeze_decode_head=False,
    freeze_edge_detector=False,
    freeze_neck=False,
    init_cfg=dict(
        checkpoint='pretrain/interformer/hrsam_mae_vit_base_enc.pth',
        type='Pretrained'),
    neck=dict(
        code_size=8,
        depth=2,
        downscale=8,
        embed_dim=256,
        mlp_ratio=2.0,
        num_heads=8,
        ref_dims=(
            32,
            64,
            128,
            256,
        ),
        ref_strides=(
            2,
            2,
            2,
            2,
        ),
        type='Inter2FormerDecoderNeck',
        uncertainty_kernel_size=7),
    test_cfg=dict(
        fast_mode=True, inner_radius=5, num_clicks=20, outer_radius=0),
    train_cfg=dict(
        fast_mode=True,
        gamma=0.6,
        inner_radius=5,
        max_num_clicks=20,
        outer_radius=0,
        sfc_inner_k=1.7),
    type='Inter2FormerClickSegmentorBSQA')
randomness = dict(seed=42)
resume = False
test_cfg = None
train_cfg = None
val_cfg = None
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = None
work_dir = './work_dirs/./inter2former_eval'
