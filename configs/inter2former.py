data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=0,
    test_cfg=dict(size_divisor=256)
)

model = dict(
    type='Inter2FormerClickSegmentorBSQA',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='pretrain/interformer/hrsam_mae_vit_base_enc.pth'
    ),
    freeze_backbone=False,
    freeze_neck=False,
    freeze_decode_head=False,
    freeze_edge_detector=False,
    edge_detector=dict(
        type='CannyNet',
        depth=4,
        strides=(2, 2, 2, 2)
    ),
    backbone=dict(
        type='HRSAMViTNoSSM',
        downsample_sizes=[16 * 16],
        window_size=16,
        in_dim=3,
        img_size=224,
        patch_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        use_checkpoint=False,
        out_indices=(2, 5, 8, 11),

        final_embed_dim=256
    ),
    neck=dict(
        type='Inter2FormerDecoderNeck',
        downscale=8,
        depth=2,
        embed_dim=256,

        ref_dims=(32, 64, 128, 256),
        ref_strides=(2, 2, 2, 2),

        num_heads=8,
        code_size=8,
        mlp_ratio=2.0,
        uncertainty_kernel_size=7),
    decode_head=dict(
        type='DynamicLocalUpsamplingTrain',
        embed_dim=256,
        num_upsamplers=4,
        expand_ratio=1.4,
        threshold=-2.0,
        loss_decode=[
            dict(type='NormalizedFocalLoss', loss_weight=1.0),
            dict(type='BinaryIoU')]),
    train_cfg=dict(
        max_num_clicks=20,
        gamma=0.6,
        inner_radius=5,
        outer_radius=0,
        sfc_inner_k=1.7,
        fast_mode=True),
    test_cfg=dict(
        num_clicks=20,
        inner_radius=5,
        outer_radius=0,
        fast_mode=True)
)
