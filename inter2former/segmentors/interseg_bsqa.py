import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.utils import add_prefix

from ..utils import REF_UNKNOWN
from .interseg import Inter2FormerClickSegmentor
from ..utils.init_clk import sample_init_clicks


@MODELS.register_module()
class Inter2FormerClickSegmentorBSQA(Inter2FormerClickSegmentor):

    def __init__(self,
                 edge_detector,
                 backbone,
                 neck,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None,
                 remove_backbone=False,
                 freeze_backbone=False,
                 freeze_neck=False,
                 freeze_decode_head=False,
                 freeze_edge_detector=False):
        super(Inter2FormerClickSegmentorBSQA, self).__init__(
            backbone=backbone,
            neck=neck,
            decode_head=decode_head,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            remove_backbone=remove_backbone)
        if edge_detector is not None:
            self.edge_detector = MODELS.build(edge_detector)
        else:
            self.edge_detector = None

        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.freeze_decode_head = freeze_decode_head
        self.freeze_edge_detector = freeze_edge_detector

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if freeze_neck:
            for param in self.neck.parameters():
                param.requires_grad = False
        if freeze_decode_head:
            for param in self.decode_head.parameters():
                param.requires_grad = False
        if freeze_edge_detector and self.edge_detector is not None:
            for param in self.edge_detector.parameters():
                param.requires_grad = False

    def interact_train(self, inputs, data_samples):  # noqa
        self.step.data.add_(1)
        inputs_dict, data_samples_dict, _ = \
            self.redistribute_tensor(inputs, data_samples)
        if len(inputs_dict) > 1:
            raise ValueError(f'Not support multiple datasets, '
                             f'but got {list(inputs_dict.keys())}')
        dataset = next(iter(inputs_dict.keys()))
        inputs, data_samples = inputs_dict[dataset], data_samples_dict[dataset]

        cfg = self.parse_train_cfg(dataset)
        gt_sem_segs = self.check_gt_validity(data_samples, train=True)

        if hasattr(self.train_cfg, 'fast_mode'):
            fast_mode = self.train_cfg.fast_mode
        else:
            warnings.warn(
                'Please set `fast_mode` in `train_cfg`, default to True.'
            )
            fast_mode = True

        update_ref_label_by_point_lists = partial(
            self.update_ref_label_by_point_lists,
            inner_radius=self.train_cfg.inner_radius,
            outer_radius=self.train_cfg.outer_radius,
            fast_mode=fast_mode
        )

        self.eval()
        pre_labels = torch.zeros_like(gt_sem_segs)
        seg_labels = gt_sem_segs.detach().clone()

        rng = np.random.default_rng(self.step.item())
        if rng.random() < self.train_cfg.sample_init_clicks_prob:
            points_list = sample_init_clicks(seg_labels, self.train_cfg)
            if any([pts is None for pts in points_list]):
                points_list = \
                    self.update_clicks(pre_labels, seg_labels, None, 1.0)
        else:
            points_list = self.update_clicks(pre_labels, seg_labels, None, 1.0)
        ref_labels = update_ref_label_by_point_lists(
            torch.ones_like(gt_sem_segs) * REF_UNKNOWN, points_list)

        with torch.no_grad():
            image_embeds = self.backbone(inputs)
            if self.edge_detector is not None:
                edge_maps = self.edge_detector(inputs)
            else:
                edge_maps = None
            for _ in range(self.sample_num_clicks(
                    cfg.max_num_clicks, cfg.gamma, rng)):
                x = self.neck(image_embeds, ref_label=ref_labels)
                logits = self.decode_head(x, edge_maps)
                if logits.shape[-2:] != inputs.shape[-2:]:
                    logits = self.interpolate(logits, inputs.shape[-2:])
                if logits.shape[1] == 2:
                    pre_labels = logits.argmax(dim=1, keepdim=True)
                elif logits.shape[1] == 1:
                    pre_labels = (logits > 0.0).to(pre_labels)
                else:
                    raise ValueError(
                        f'Invalid logits shape {tuple(logits.shape)}')
                ref_labels = \
                    self.update_ref_label_by_prediction(ref_labels, pre_labels)
                n_prev_pts_list = [len(pts) for pts in points_list]
                points_list = self.update_clicks(
                    pre_labels, seg_labels, points_list, cfg.sfc_inner_k)
                ref_labels = update_ref_label_by_point_lists(
                    ref_labels,
                    [pts[n:] for pts, n in zip(points_list, n_prev_pts_list)])

        if hasattr(self.train_cfg, 'require_gt_sem_seg'):
            require_gt_sem_seg = self.train_cfg.require_gt_sem_seg
        else:
            warnings.warn(
                'Please set `require_gt_sem_seg` in `train_cfg`, '
                'default to False.'
            )
            require_gt_sem_seg = False

        self.train()
        all_losses = dict()
        if not self.freeze_backbone:
            image_embeds = self.backbone(inputs)
        if not self.freeze_edge_detector:
            edge_maps = self.edge_detector(inputs)
        if require_gt_sem_seg:
            x, losses = \
                self.neck(image_embeds, ref_label=ref_labels, mode='train',
                          gt_sem_seg=gt_sem_segs)
        else:
            x, losses = \
                self.neck(image_embeds, ref_label=ref_labels, mode='train')
        all_losses.update(add_prefix(losses, dataset))
        for prefix, x in (
            list(x.items())
            if isinstance(x, dict)
            else [(None, x)]
        ):
            losses = dict()
            logits, aux = self.decode_head(x, edge_maps)

            logits = self.interpolate(logits, inputs.shape[-2:])
            if logits.size(1) == 2:
                logits = logits[:, 1:] - logits[:, :1]
            elif logits.size(1) != 1:
                raise ValueError(f'Invalid logits shape {tuple(logits.shape)}')
            losses.update(self.loss_by_decode(logits, gt_sem_segs))

            if gt_sem_segs.shape[-2:] != aux.shape[-2:]:
                aux_gt_sem_segs = F.adaptive_max_pool2d(
                    (gt_sem_segs > 0).float(), aux.shape[-2:]
                ).to(gt_sem_segs)
            else:
                aux_gt_sem_segs = gt_sem_segs
            if aux.size(1) == 2:
                aux = aux[:, 1:] - aux[:, :1]
            elif aux.size(1) != 1:
                raise ValueError(f'Invalid aux logits shape {tuple(aux.shape)}')
            losses.update(add_prefix(
                self.loss_by_decode(aux, aux_gt_sem_segs), 'aux'))

            if self.with_metric:
                losses.update(self.metric_by_decode(logits, gt_sem_segs))
                losses.update(
                    add_prefix(self.metric_by_decode(aux, aux_gt_sem_segs), 'aux'))
            if prefix is not None:
                losses = add_prefix(losses, prefix)
            all_losses.update(add_prefix(losses, dataset))
        return all_losses

    @torch.no_grad()
    def interact_test(self, inputs, data_samples):
        cfg = self.test_cfg
        gt_sem_segs = self.check_gt_validity(data_samples, train=False)
        gt_sem_segs = gt_sem_segs.to(device=inputs.device)

        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.0
        if hasattr(cfg, 'fast_mode'):
            fast_mode = cfg.fast_mode
        else:
            fast_mode = False
        pts_cfg = dict(
            inner_radius=cfg.inner_radius, outer_radius=cfg.outer_radius,
            fast_mode=fast_mode
        )

        self.eval()
        points, results, logits_list = list(), list(), list()
        pre_labels = torch.zeros_like(gt_sem_segs)
        seg_labels = gt_sem_segs
        ref_labels = torch.ones_like(inputs)[:, :1] * REF_UNKNOWN

        h, w = seg_labels.shape[-2:]
        image_embeds = self.backbone(inputs)
        if self.edge_detector is not None:
            edge_maps = self.edge_detector(inputs)
        else:
            edge_maps = None
        for _ in range(cfg.num_clicks):
            n_prev_pts = len(points)
            points, *_ = self.update_clicks(
                pre_labels, seg_labels, [points], sfc_inner_k)
            if len(points) > n_prev_pts:
                ref_labels = self.update_ref_label_by_point_lists(
                    ref_labels, [points[n_prev_pts:]], **pts_cfg)
            logits = self.decode_head(
                self.neck(image_embeds, ref_label=ref_labels), edge_maps)
            if logits.shape[-2:] != inputs.shape[-2:]:
                logits = self.interpolate(logits, inputs.shape[-2:])
            if logits.shape[1] == 2:
                pre_labels = logits.argmax(dim=1, keepdim=True)
            elif logits.shape[1] == 1:
                pre_labels = (logits > 0.0).to(pre_labels)
            else:
                raise ValueError(
                    f'Invalid logits shape {tuple(logits.shape)}')
            ref_labels = \
                self.update_ref_label_by_prediction(ref_labels, pre_labels)
            pre_labels = pre_labels[..., :h, :w]
            results.append(pre_labels.squeeze().detach().cpu().numpy())
            logits_list.append(logits[..., :h, :w].squeeze().detach().cpu().numpy())
        gt_sem_segs = gt_sem_segs.squeeze().detach().cpu().numpy()
        return points, results, gt_sem_segs, logits_list


@MODELS.register_module()
class Inter2FormerClickSegmentorBSQATargetSize(Inter2FormerClickSegmentorBSQA):

    @torch.no_grad()
    def interact_test(self, inputs, data_samples):
        cfg = self.test_cfg
        gt_sem_segs = self.check_gt_validity(data_samples, train=False)
        gt_sem_segs = gt_sem_segs.to(device=inputs.device)

        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.0
        if hasattr(cfg, 'fast_mode'):
            fast_mode = cfg.fast_mode
        else:
            fast_mode = False
        pts_cfg = dict(
            inner_radius=cfg.inner_radius, outer_radius=cfg.outer_radius,
            fast_mode=fast_mode
        )

        target_size = cfg.target_size

        if hasattr(cfg, 'extra_decode_args'):
            decode_args = cfg.extra_decode_args
        else:
            decode_args = dict()

        self.eval()
        points, results, logits_list = list(), list(), list()
        pre_labels = torch.zeros_like(gt_sem_segs)
        seg_labels = gt_sem_segs

        h, w = inputs.shape[-2:]
        if h != w:
            size = max(h, w)
            padded_inputs = F.pad(inputs, (0, size - w, 0, size - h))
        else:
            padded_inputs = inputs
        resize_shape = (target_size, target_size)
        scale_factor = target_size / max(h, w)
        resized_inputs = F.interpolate(padded_inputs,
                                       size=resize_shape,
                                       mode='bilinear',
                                       align_corners=False)
        image_embeds = self.backbone(resized_inputs)
        edge_maps = self.edge_detector(resized_inputs)
        ref_labels = torch.ones_like(resized_inputs)[:, :1] * REF_UNKNOWN
        for _ in range(cfg.num_clicks):
            n_prev_pts = len(points)
            points, *_ = self.update_clicks(
                pre_labels, seg_labels, [points], sfc_inner_k)
            if len(points) > n_prev_pts:
                ref_labels = self.update_ref_label_by_point_lists(
                    ref_labels,
                    [[(int(scale_factor * y), int(scale_factor * x), mode)
                      for y, x, mode in points[n_prev_pts:]]],
                    **pts_cfg)
            ori_logits = self.decode_head(
                self.neck(image_embeds, ref_label=ref_labels),
                edge_maps, **decode_args)
            if ori_logits.shape[-2:] != resize_shape:
                logits = self.interpolate(ori_logits, resize_shape)
            else:
                logits = ori_logits
            pre_labels = self.convert_logits_to_labels(logits, pre_labels.dtype)
            ref_labels = \
                self.update_ref_label_by_prediction(ref_labels, pre_labels)
            if ori_logits.shape[-2:] != padded_inputs.shape[-2:]:
                logits = self.interpolate(ori_logits, padded_inputs.shape[-2:])
            else:
                logits = ori_logits
            pre_labels = self.convert_logits_to_labels(logits, pre_labels.dtype)
            sh, sw = seg_labels.shape[-2:]
            pre_labels = pre_labels[..., :sh, :sw]
            results.append(pre_labels.squeeze().detach().cpu().numpy())
            logits_list.append(logits[..., :sh, :sw].squeeze().detach().cpu().numpy())
        gt_sem_segs = gt_sem_segs.squeeze().detach().cpu().numpy()
        return points, results, gt_sem_segs, logits_list

    @staticmethod
    def convert_logits_to_labels(logits, dtype):
        if logits.shape[1] == 2:
            return logits.argmax(dim=1, keepdim=True)
        elif logits.shape[1] == 1:
            return (logits > 0.0).to(dtype=dtype)
        else:
            raise ValueError(f'Invalid logits shape {tuple(logits.shape)}')
