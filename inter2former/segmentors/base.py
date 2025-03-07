import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import add_prefix, SampleList
from mmengine.config import ConfigDict

from engine.utils.click_fast import (CLK_POSITIVE, fast_generate_clicks,
                                     fast_generate_single_click)
from engine.utils.resize import resize_along_longest_side
from engine.utils.resize import resize_image_along_longest_size
from engine.segmentors import BaseInterSegmentor


class BaseInter2FormerClickSegmentor(BaseInterSegmentor):  # noqa

    def __init__(self,
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
                 **kwargs):
        if len(kwargs) > 0:
            warnings.warn(f'Ignoring kwargs {kwargs}')

        super(BaseInter2FormerClickSegmentor, self).__init__(
            backbone=backbone,
            neck=neck,
            decode_head=decode_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            remove_backbone=remove_backbone)

        if auxiliary_head is not None:
            self._init_auxiliary_head(auxiliary_head)
        else:
            self.auxiliary_head = None

        self.register_buffer('step', torch.tensor(0))

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if self.auxiliary_head is None:
            return losses
        elif isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    @staticmethod
    def check_gt_validity(data_samples, train):
        gt_sem_seg = torch.stack([
            sample.gt_sem_seg.data for sample in data_samples], dim=0)
        if train:
            # Check if gt_semantic_seg is multi-label
            if gt_sem_seg.ndim == 4 and gt_sem_seg.size(1) != 1:
                raise ValueError(f'Cannot handle multi `gt_sem_seg` '
                                 f'with shape {tuple(gt_sem_seg.shape)}')
            elif gt_sem_seg.ndim not in (3, 4):
                raise ValueError(f'`gt_sem_seg` is expected to have '
                                 f'shape (batch_size, height, width) or '
                                 f'(batch_size, 1, height, width), but '
                                 f'got shape {tuple(gt_sem_seg.shape)}')
        else:
            # Check if only one sample is present in the batch.
            if gt_sem_seg.size(0) > 1:
                raise ValueError(f'Only a single sample per batch is allowed, '
                                 f'but got {gt_sem_seg.size(0)} samples '
                                 f'in this batch')
            # Check if gt_semantic_seg has the correct shape.
            if gt_sem_seg.ndim not in (3, 4) or \
                    gt_sem_seg[..., 0, 0].nelement() > 1:
                raise ValueError(f'`gt_sem_seg` is expected to have '
                                 f'shape (1, height, width) or '
                                 f'(1, 1, height, width), but '
                                 f'got shape {tuple(gt_sem_seg.shape)}')
        if gt_sem_seg.ndim == 3:
            gt_sem_seg = gt_sem_seg.unsqueeze(1)
        return gt_sem_seg

    @staticmethod
    def redistribute_tensor(inputs, data_samples):
        """Redistribute tensor inputs and data_samples according to dataset"""
        inputs_dict, data_samples_dict, sample_idxs2idxs = {}, {}, {}
        for idx, (x, data_sample) in enumerate(zip(inputs, data_samples)):
            sample_idx = data_sample.metainfo['sample_idx']
            sample_idxs2idxs[sample_idx] = idx
            dataset = data_sample.metainfo['dataset']
            inputs_dict.setdefault(dataset, []).append(x)
            data_samples_dict.setdefault(dataset, []).append(data_sample)
        inputs_dict = {dataset: torch.stack(inputs, dim=0)
                       for dataset, inputs in inputs_dict.items()}
        return inputs_dict, data_samples_dict, sample_idxs2idxs

    def parse_train_cfg(self, dataset):
        cfg = self.train_cfg
        if hasattr(cfg, 'interact_params'):
            interact_params = cfg.interact_params
        else:
            warnings.warn(f'Not found interact_params in train_cfg')
            interact_params = dict()
        if dataset in interact_params:
            params = interact_params[dataset]
            max_num_clicks = params.get('max_num_clicks', cfg.max_num_clicks)
            gamma = params.get('gamma', cfg.gamma)
        else:
            warnings.warn(f'Not found interact_params of {dataset}')
            max_num_clicks = cfg.max_num_clicks
            gamma = cfg.gamma
        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.7
        return ConfigDict(max_num_clicks=max_num_clicks,
                          gamma=gamma, sfc_inner_k=sfc_inner_k)

    @staticmethod
    def sample_num_clicks(max_num_clicks, gamma, rng=None):
        probs = gamma ** np.arange(max_num_clicks + 1)
        probs /= probs.sum()
        if rng is None:
            rng = np.random
        return rng.choice(range(len(probs)), p=probs)

    @staticmethod
    def update_clicks(pre_label,
                      seg_label,
                      points_list,
                      sfc_inner_k=1.0,
                      downsample_factor=1.0,
                      ignore_masks=None):

        ori_pre_label = pre_label.detach().clone()
        ori_seg_label = seg_label.detach().clone()

        if points_list is None:
            points_list = [None for _ in pre_label]
        ori_points_list = deepcopy(points_list)
        scale = 1.0
        points_list = deepcopy(ori_points_list)
        if downsample_factor > 1.0:
            warnings.warn(f'`downsample_factor` > 1.0, ignored')
        elif downsample_factor < 1.0:
            scale = 1.0 / downsample_factor
            ori_h, ori_w = pre_label.shape[-2:]
            tar_h = int(ori_h * downsample_factor)
            tar_w = int(ori_w * downsample_factor)
            pre_label = F.interpolate(
                pre_label.float(), size=(tar_h, tar_w),
                mode='bilinear', align_corners=False)
            pre_label = (pre_label > 0.5).long()
            seg_label = F.interpolate(
                seg_label.float(), size=(tar_h, tar_w),
                mode='bilinear', align_corners=False)
            seg_label = (seg_label > 0.5).long()

            def clamp(y_, x_):
                return max(0, min(y_, pre_label.shape[-2] - 1)), \
                       max(0, min(x_, pre_label.shape[-1] - 1))

            for points in points_list:
                if points is not None:
                    for idx, (y, x, mode) in list(enumerate(points)):
                        points[idx] = \
                            clamp(
                                int(downsample_factor * y),
                                int(downsample_factor * x)
                            ) + (mode, )

        clicks = fast_generate_clicks(
            pre_labels=pre_label, seg_labels=seg_label,
            points_list=points_list, sfc_inner_k=sfc_inner_k,
            ignore_masks=ignore_masks)
        points_list = deepcopy(ori_points_list)
        for idx, (y, x, mode) in enumerate(clicks):
            if mode is not None:
                if points_list[idx] is None:
                    points_list[idx] = []
                points_list[idx].append((int(scale * y), int(scale * x), mode))
            elif points_list[idx] is None:
                warnings.warn(
                    f'No clicks generated for sample {idx}, '
                    f'please increase the downsample_factor '
                    f'{downsample_factor}')
                if ignore_masks is None:
                    ignore_mask = None
                else:
                    ignore_mask = ignore_masks[idx].squeeze()
                y, x, mode = fast_generate_single_click(
                    pre_label=ori_pre_label[idx].squeeze(),
                    seg_label=ori_seg_label[idx].squeeze(),
                    sfc_inner_k=sfc_inner_k,
                    ignore_mask=ignore_mask)
                if mode is None:
                    raise ValueError(f'Still not found valid clicks in '
                                     f'sample {idx} for the original size, '
                                     f'please check the input data')
                points_list[idx] = [(y, x, mode)]
        return points_list

    @staticmethod
    def resize_and_pad_to_target_size(x, target_size):
        x = resize_image_along_longest_size(x, target_size)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, target_size - w, 0, target_size - h))
        return x

    def crop_and_resize_to_original_size(
            self, x, ori_hw, target_size, mode='bilinear'):
        h, w = resize_along_longest_side(ori_hw, target_size)
        x = self.interpolate(x[..., :h, :w], ori_hw, mode=mode)
        return x

    def encode_decode(self, inputs, batch_data_samples):
        raise NotImplementedError

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def predict(self, inputs, data_samples=None):
        raise NotImplementedError
