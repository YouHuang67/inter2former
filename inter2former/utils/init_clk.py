import random
import warnings

import torch

from engine.utils.click import CLK_POSITIVE, CLK_NEGATIVE
from engine.utils.boundary_fast import fast_erode, fast_dilate
from engine.utils.zoom_in import get_bbox_from_mask, expand_bbox, convert_bbox_to_mask


def generate_points(mask, N, min_dist, M):
    """
    :param mask: (H, W)
    :param N:
    :param min_dist:
    :param M:
    :return: (N, 2)
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must have dimensions (H, W)")
    device = mask.device

    left, up, right, bottom = get_bbox_from_mask(mask).view(4)
    indices = torch.nonzero(mask[up:bottom, left:right], as_tuple=False)
    indices += torch.stack([up, left]).view(1, 2)
    if indices.size(0) < N:
        warnings.warn(f"Not enough points in mask to select {N} points "
                      f"for {M} samples. Reducing N to maximum available points.")
        N = indices.size(0)
        if N == 0:
            return torch.empty((0, 2), device=device)
    if N == 1:
        return indices[torch.randint(len(indices), (1, ))].view(1, 2).float()

    chosen_indices = indices[
        torch.rand(M, len(indices),
                   device=device).argsort(dim=-1)[:, :N]
    ]
    chosen_indices = chosen_indices.view(M, N, 2).float()

    dist_matrices = torch.norm(
        chosen_indices[:, :, None, :] - chosen_indices[:, None, :, :], dim=-1
    )
    dist_matrices += torch.eye(N, device=device).unsqueeze(0) * float('inf')
    min_distances_per_group = dist_matrices.min(dim=-1).values.min(dim=-1).values

    valid_groups = min_distances_per_group >= min_dist
    if valid_groups.any():
        chosen_indices = chosen_indices[valid_groups]
        return chosen_indices[0].view(N, 2)
    best_index = min_distances_per_group.argmax()
    return chosen_indices[best_index].view(N, 2)


def sample_init_clicks(seg_labels, cfg, min_area=1000):
    """
    :param seg_labels:
    :param cfg: for example, dict(
        pos_cfg=dict(dm=10, ds=40, N=5, M=10)
        neg_s1_cfg=dict(dm=20, ds=40, N=3, expr_h=1.4, expr_w=1.4, M=10)
        neg_s3_cfg=dict(dm=10, ds=40, N=3, M=10)
    )
    :param min_area:
    :return:
    """
    pos_cfg = cfg.pos_cfg
    neg_s1_cfg = cfg.neg_s1_cfg
    neg_s3_cfg = cfg.neg_s3_cfg

    points_list = []
    H, W = seg_labels.shape[-2:]
    seg_labels = seg_labels.view(-1, H, W)
    for mask, bbox in zip(seg_labels, get_bbox_from_mask(seg_labels)):
        if not mask.any():
            raise ValueError("Mask must have at least one positive pixel.")
        if mask.sum() < min_area:
            points_list.append(None)
            continue
        points = []
        extra_kwargs = dict(iterations=1, bbox=bbox)

        dm, ds, N, M = pos_cfg.dm, pos_cfg.ds, pos_cfg.N, pos_cfg.M
        N = random.randint(1, N)
        while True:
            dm += int(dm % 2 == 0)
            pos_inner = fast_erode(mask, kernel_size=dm, **extra_kwargs)
            pos_inner[~mask.bool()] = 0
            if pos_inner.sum() >= min_area:
                break
            dm //= 2
            if dm == 0:
                pos_inner = mask.clone()
                break
        points.extend([
            (int(y), int(x), CLK_POSITIVE)
            for y, x in generate_points(pos_inner, N, ds, M).tolist()
        ])

        dm, ds, N, expr_h, expr_w, M = \
            neg_s1_cfg.dm, neg_s1_cfg.ds, neg_s1_cfg.N, \
            neg_s1_cfg.expr_h, neg_s1_cfg.expr_w, neg_s1_cfg.M
        dm += int(dm % 2 == 0)
        neg_outer = fast_dilate(mask, kernel_size=dm, **extra_kwargs)
        N = random.randint(0, N)
        if N > 0:
            bbox_mask = convert_bbox_to_mask(
                expand_bbox(bbox, expr_h, expr_w, H, W), (H, W),
                device=mask.device
            )
            bbox_mask[neg_outer.bool()] = 0
            bbox_mask[mask.bool()] = 0
            if bbox_mask.sum() >= min_area:
                points.extend([
                    (int(y), int(x), CLK_NEGATIVE)
                    for y, x in generate_points(bbox_mask, N, ds, M).tolist()
                ])

        dm, ds, N, M = neg_s3_cfg.dm, neg_s3_cfg.ds, neg_s3_cfg.N, neg_s3_cfg.M
        dm += int(dm % 2 == 0)
        neg_inner = fast_dilate(mask, kernel_size=dm, **extra_kwargs)
        N = random.randint(0, N)
        if N > 0:
            neg_outer[neg_inner.bool()] = 0
            neg_outer[mask.bool()] = 0
            if neg_outer.sum() >= min_area:
                points.extend([
                    (int(y), int(x), CLK_NEGATIVE)
                    for y, x in generate_points(neg_outer, N, ds, M).tolist()
                ])
        points_list.append(points)
    return points_list
