import warnings
from functools import partial

import numpy as np
import torch
from mmseg.registry import MODELS
from mmseg.utils import add_prefix

from ..utils import (point_list_to_ref_labels, update_ref_labels,
                     update_ref_labels_with_masks)
from ..utils import (REF_POSSIBLY_FOREGROUND, REF_POSSIBLY_BACKGROUND,
                     REF_UNKNOWN)
from ..utils import fast_point_list_to_ref_labels
from .base import BaseInter2FormerClickSegmentor


@MODELS.register_module()
class Inter2FormerClickSegmentor(BaseInter2FormerClickSegmentor):

    @staticmethod
    def update_ref_label_by_point_lists(ref_labels,
                                        point_lists,
                                        inner_radius=5,
                                        outer_radius=0,
                                        fast_mode=False):
        if ref_labels[..., 0, 0].nelement() != len(point_lists):
            raise ValueError(
                f'Number of point lists {len(point_lists)} '
                f'less or more than the number of ref_labels '
                f'{tuple(ref_labels[..., 0, 0].shape)}')

        for idx, points in enumerate(list(point_lists)):
            points = [
                (y, x, mode)
                for y, x, mode in points
                if y is not None and
                   x is not None and
                   mode is not None]
            point_lists[idx] = points

        if fast_mode:
            new_ref_labels = fast_point_list_to_ref_labels(
                ref_labels, point_lists, inner_radius, outer_radius)
        else:
            new_ref_labels = point_list_to_ref_labels(
                ref_labels, point_lists, inner_radius, outer_radius)
        ref_label = update_ref_labels(ref_labels, new_ref_labels)
        return ref_label

    @staticmethod
    def update_ref_label_by_prediction(ref_labels, pre_labels):
        ref_labels = update_ref_labels_with_masks(
            ref_labels, pre_labels == 1, REF_POSSIBLY_FOREGROUND)
        ref_labels = update_ref_labels_with_masks(
            ref_labels, pre_labels == 0, REF_POSSIBLY_BACKGROUND)
        return ref_labels

    def interact_train(self, inputs, data_samples):  # noqa
        self.step.data.add_(1)
        inputs_dict, data_samples_dict, _ = \
            self.redistribute_tensor(inputs, data_samples)
        if len(inputs_dict) > 1:
            raise ValueError(f'Not support multiple datasets, '
                             f'but got {list(inputs_dict.keys())}')
        dataset = next(iter(inputs_dict.keys()))
        inputs, data_samples = inputs_dict[dataset], data_samples_dict[dataset]

        device = inputs.device
        cfg = self.parse_train_cfg(dataset)
        gt_sem_segs = self.check_gt_validity(data_samples, train=True)
        update_ref_label_by_point_lists = partial(
            self.update_ref_label_by_point_lists,
            inner_radius=self.train_cfg.inner_radius,
            outer_radius=self.train_cfg.outer_radius
        )

        self.eval()
        pre_labels = torch.zeros_like(gt_sem_segs)
        seg_labels = gt_sem_segs.detach().clone()
        points_list = self.update_clicks(pre_labels, seg_labels, None, 1.0)
        ref_labels = update_ref_label_by_point_lists(
            torch.ones_like(gt_sem_segs) * REF_UNKNOWN, points_list)

        with torch.no_grad():
            image_embeds = self.backbone(inputs)
            for _ in range(self.sample_num_clicks(
                    cfg.max_num_clicks,
                    cfg.gamma,
                    np.random.default_rng(int(self.step.item()))
            )):
                logits = self.decode_head(
                    self.neck(image_embeds, ref_label=ref_labels))
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

        self.train()
        losses = dict()
        image_embeds = self.backbone(inputs)
        x = self.neck(image_embeds, ref_label=ref_labels)
        logits = self.decode_head(x)
        logits = self.interpolate(logits, inputs.shape[-2:])
        if logits.size(1) == 2:
            logits = logits[:, 1:] - logits[:, :1]
        elif logits.size(1) != 1:
            raise ValueError(f'Invalid logits shape {tuple(logits.shape)}')
        loss = self.loss_by_decode(logits, gt_sem_segs)
        losses.update(loss)
        losses.update(self._auxiliary_head_forward_train(x, data_samples))
        if self.with_metric:
            losses.update(self.metric_by_decode(logits, gt_sem_segs))
        return add_prefix(losses, dataset)

    @torch.no_grad()
    def interact_test(self, inputs, data_samples):
        cfg = self.test_cfg
        gt_sem_segs = self.check_gt_validity(data_samples, train=False)
        gt_sem_segs = gt_sem_segs.to(device=inputs.device)

        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.0
        pts_cfg = dict(
            inner_radius=cfg.inner_radius, outer_radius=cfg.outer_radius)

        self.eval()
        points, results = list(), list()
        pre_labels = torch.zeros_like(gt_sem_segs)
        seg_labels = gt_sem_segs
        ref_labels = torch.ones_like(inputs)[:, :1] * REF_UNKNOWN

        h, w = seg_labels.shape[-2:]
        image_embeds = self.backbone(inputs)
        for _ in range(cfg.num_clicks):
            n_prev_pts = len(points)
            points, *_ = self.update_clicks(
                pre_labels, seg_labels, [points], sfc_inner_k)
            if len(points) > n_prev_pts:
                ref_labels = self.update_ref_label_by_point_lists(
                    ref_labels, [points[n_prev_pts:]], **pts_cfg)
            logits = self.decode_head(
                self.neck(image_embeds, ref_label=ref_labels))
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
        gt_sem_segs = gt_sem_segs.squeeze().detach().cpu().numpy()
        return points, results, gt_sem_segs
