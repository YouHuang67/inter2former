import random
from typing import Union, Optional, List, Tuple
import numpy as np
import mmcv
import torch

from engine.utils import (CLK_POSITIVE, CLK_NEGATIVE,
                          fast_mask_to_distance, mask_to_distance)


REF_DEFINITELY_BACKGROUND = 0
REF_POSSIBLY_BACKGROUND = 1
REF_UNKNOWN = 2
REF_POSSIBLY_FOREGROUND = 3
REF_DEFINITELY_FOREGROUND = 4
REF_MODES = [
    REF_DEFINITELY_BACKGROUND,
    REF_POSSIBLY_BACKGROUND,
    REF_UNKNOWN,
    REF_POSSIBLY_FOREGROUND,
    REF_DEFINITELY_FOREGROUND]
REF_INVERSE_MODES = {
    REF_POSSIBLY_BACKGROUND: REF_POSSIBLY_FOREGROUND,
    REF_POSSIBLY_FOREGROUND: REF_POSSIBLY_BACKGROUND
}


def points_to_ref_label(
        label: Union[torch.Tensor, np.ndarray],
        points: List[Tuple[int, int, str]],
        inner_radius: float,
        outer_radius: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert the clicked points to a reference label.
    :param label: A binary mask of shape (height, width).
    :param points: A list of (y, x, mode),
                   where y and x are the coordinates of a point,
                   and mode is either 'positive' or 'negative'.
    :param inner_radius: The inner radius for the reference label.
    :param outer_radius: The outer radius for the reference label.
    :return: mask of shape (height, width) representing the reference label.
    """

    if not isinstance(label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(label)}')
    if len(label.shape) != 2:
        raise ValueError(f'Cannot handle label of shape: {tuple(label.shape)} '
                         f', expected (height, width)')
    for y, x, mode in points:
        if isinstance(float(y), float) and \
           isinstance(float(x), float) and \
           mode in [CLK_POSITIVE, CLK_NEGATIVE]:
            continue
        raise ValueError(
            f'Found invalid point {(y, x, mode)} among points: {points}')

    ori_label = label
    to_tensor = isinstance(label, torch.Tensor)
    label = torch.ones_like(label.cpu()) if to_tensor else np.ones_like(label)
    label = label * REF_UNKNOWN

    if len(points) == 0:
        label = label.to(ori_label) if to_tensor else label.astype(ori_label.dtype)
        return label

    inner_masks = dict()
    outer_masks = dict()
    shape = tuple(label.shape[-2:])
    for y, x, mode in points:
        if mode == CLK_POSITIVE:
            inner, outer = REF_DEFINITELY_FOREGROUND, REF_POSSIBLY_FOREGROUND
        else:
            inner, outer = REF_DEFINITELY_BACKGROUND, REF_POSSIBLY_BACKGROUND
        inner_mask = inner_masks.setdefault(inner, np.ones(shape))
        outer_mask = outer_masks.setdefault(outer, np.ones(shape))
        inner_mask[y, x], outer_mask[y, x] = 0, 0

    for outer in list(outer_masks):
        dist = mask_to_distance(outer_masks[outer], False)
        mask = (dist <= (inner_radius + outer_radius))
        mask = torch.from_numpy(mask) if to_tensor else mask
        label[mask] = outer
        outer_masks[outer] = mask
    if REF_POSSIBLY_FOREGROUND in outer_masks and \
       REF_POSSIBLY_BACKGROUND in outer_masks:
        mask = outer_masks[REF_POSSIBLY_FOREGROUND] & \
               outer_masks[REF_POSSIBLY_BACKGROUND]
        label[mask] = REF_UNKNOWN

    for inner in list(inner_masks):
        dist = mask_to_distance(inner_masks[inner], False)
        mask = (dist <= inner_radius)
        mask = torch.from_numpy(mask) if to_tensor else mask
        label[mask] = inner
    label = label.to(ori_label) if to_tensor else label.astype(ori_label.dtype)
    return label


def point_list_to_ref_labels(
        labels: Union[torch.Tensor, np.ndarray],
        point_list: List[List[Tuple[int, int, str]]],
        inner_radius: float,
        outer_radius: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a list of point lists to reference labels.

    :param labels: A tensor of shape (*, height, width).
    :param point_list: A list of point lists, where each point list contains
                       tuples of (x, y, class) representing the coordinates and
                       class label of each point.
    :param inner_radius: The inner radius for creating reference labels.
    :param outer_radius: The outer radius for creating reference labels.
    :return: A tensor of shape (*, height, width).
    """

    if not isinstance(labels, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(labels)}')

    if len(point_list) != len(labels[..., 0, 0].reshape((-1,))):
        raise ValueError(f'Number of point lists {len(point_list)} '
                         f'less or more than the number of labels '
                         f'{tuple(labels[..., 0, 0].shape)}')

    ori_labels = labels
    to_tensor = isinstance(labels, torch.Tensor)
    labels = labels.cpu() if to_tensor else labels
    labels = labels.reshape((-1, ) + labels.shape[-2:])

    ref_labels = list()
    for idx, label in enumerate(labels):
        points = point_list[idx]
        ref_label = points_to_ref_label(
            label, points, inner_radius, outer_radius)
        ref_labels.append(ref_label)

    if to_tensor:
        ref_labels = torch.stack(ref_labels, dim=0)
        ref_labels = ref_labels.view_as(ori_labels).to(ori_labels)
    else:
        ref_labels = np.stack(ref_labels, axis=0)
        ref_labels = ref_labels.reshape(ori_labels.shape)

    return ref_labels


def update_ref_labels_with_masks(
    x: Union[torch.Tensor, np.ndarray],
    masks: Union[torch.Tensor, np.ndarray],
    mode: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Update the reference label x with a binary mask

    :param x: Input array, shape (*, height, width)
    :param masks: Binary mask array, shape (*, height, width)
    :param mode: The mode of the reference label, must be one of `REF_MODES`
    :return: The updated reference label array, shape (*, height, width)
    """
    if not isinstance(x, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of x: {type(x)}')
    if not isinstance(masks, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(masks)}')
    if mode not in REF_MODES:
        raise ValueError(f'Unknown value of mode: {mode}')

    if isinstance(x, np.ndarray):
        res = x.copy()
        if mode == REF_UNKNOWN:
            return res
        masks = masks.astype(bool)
        if mode in [REF_DEFINITELY_BACKGROUND, REF_DEFINITELY_FOREGROUND]:
            res[masks] = mode
        else:
            inverse_mode = REF_INVERSE_MODES[mode]
            res[np.logical_and(masks, x == REF_UNKNOWN)] = mode
            res[np.logical_and(masks, x == inverse_mode)] = REF_UNKNOWN
    else:
        res = x.detach().clone()
        if mode == REF_UNKNOWN:
            return res
        masks = masks.bool()
        if mode in [REF_DEFINITELY_BACKGROUND, REF_DEFINITELY_FOREGROUND]:
            res[masks] = mode
        else:
            inverse_mode = REF_INVERSE_MODES[mode]
            res[torch.logical_and(masks, x == REF_UNKNOWN)] = mode  # noqa
            res[torch.logical_and(masks, x == inverse_mode)] = REF_UNKNOWN  # noqa
    return res


def update_ref_labels(
    x: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Update the reference label x with a reference label

    :param x: Input array, shape (*, height, width)
    :param labels: Reference label array, shape (*, height, width)
    :return: The updated reference label array, shape (*, height, width)
    """
    for mode in REF_MODES:
        x = update_ref_labels_with_masks(x, labels == mode, mode)  # noqa
    return x
