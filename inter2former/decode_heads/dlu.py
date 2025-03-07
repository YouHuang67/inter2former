import warnings

import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS
from mmseg.models.builder import build_loss

from engine.utils.zoom_in import get_bbox_from_mask
try:
    from inter2former.cpp_extension.fast_mask_convert import _fast_mask_convert  # noqa
    get_bbox_from_mask_cpu = _fast_mask_convert.get_bbox_from_mask
except ImportError:
    warnings.warn('Cannot import cpp-based fast_mask_convert, use PyTorch instead')
    _fast_mask_convert = None


class Upsample2x(nn.Module):

    def __init__(self, in_channels, downscale=4, silu=True):
        super(Upsample2x, self).__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv2d(
            in_channels, int(round(in_channels / downscale)),
            kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU() if silu else nn.Identity()

    def forward(self, x, edge_map):
        x = self.norm(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = x + edge_map
        x = self.conv(x)
        x = self.silu(x)
        return x


class SimMLP(nn.Sequential):

    def __init__(self, in_channels, channels):
        super(SimMLP, self).__init__()
        in_channels = [in_channels] + list(channels[:-1])
        out_channels = channels
        for in_dim, out_dim in zip(in_channels, out_channels):
            self.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            if out_dim > 1:
                self.append(nn.SiLU())


@HEADS.register_module()
class DynamicLocalUpsamplingTrain(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_upsamplers=4,
                 expand_ratio=1.4,
                 loss_decode=dict(  # noqa
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 align_corners=False,
                 no_aux_input_grad=True,
                 threshold=0.0):
        super(DynamicLocalUpsamplingTrain, self).__init__()
        self.embed_dim = embed_dim
        self.num_upsamplers = num_upsamplers
        self.upsamplers = nn.ModuleList([
            Upsample2x(embed_dim // 4 ** i, silu=(i < num_upsamplers - 1))
            for i in range(num_upsamplers)
        ])

        self.mlp = SimMLP(
            embed_dim,
            [embed_dim // 4 ** i for i in range(1, num_upsamplers + 1)]
        )
        self.expand_ratio = expand_ratio
        self.align_corners = align_corners
        self.num_classes = 2
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict, '
                            f'but got {type(loss_decode)}')

        self.strides = [2 ** i for i in range(1, num_upsamplers + 1)]
        self.no_aux_input_grad = no_aux_input_grad
        self.threshold = threshold

    def forward(self, x, edge_maps):
        ori_x = x
        if len(edge_maps) > self.num_upsamplers:
            edge_maps = edge_maps[-self.num_upsamplers:]
        for i, up in enumerate(self.upsamplers):
            x = up(x, edge_maps[i])
        if self.training:
            if self.no_aux_input_grad:
                ori_x = ori_x.detach()
            return x, self.mlp(ori_x)
        else:
            return x


@HEADS.register_module()
class DynamicLocalUpsampling(DynamicLocalUpsamplingTrain):

    def forward(self, x, edge_maps, gt_mask=None):
        if self.no_aux_input_grad:
            pre_mask = self.mlp(x.detach())
        else:
            pre_mask = self.mlp(x)
        if gt_mask is not None:
            ref_mask = (gt_mask > self.threshold).float()
        else:
            ref_mask = (pre_mask > self.threshold).float()
        if _fast_mask_convert is not None:
            bboxes = get_bbox_from_mask_cpu(ref_mask, 1).view(-1, 4)
        else:
            bboxes = get_bbox_from_mask(ref_mask, 1).view(-1, 4)
        mask = x.new_empty(
            len(x), 1,
            x.shape[-2] * self.strides[-1],
            x.shape[-1] * self.strides[-1])
        mask.fill_(-100.0)
        for i, z in enumerate(x.chunk(len(x), dim=0)):
            left, top, right, bottom = bboxes[i]
            z = z[..., top:bottom, left:right]
            for j, edge_map in enumerate(edge_maps):
                left, top, right, bottom = bboxes[i] * self.strides[j]
                z = self.upsamplers[j](
                    z, edge_map[i:i + 1, :, top:bottom, left:right])
            left, top, right, bottom = bboxes[i] * self.strides[-1]
            mask[i:i + 1, :, top:bottom, left:right] = z
        return mask


if __name__ == '__main__':
    import torch
    _size = 1024
    head = DynamicLocalUpsampling(
        embed_dim=256,
        num_upsamplers=4)
    _x = torch.randn(2, 256, _size // 16, _size // 16)
    _gt = torch.zeros(2, 1, _size // 16, _size // 16).long()
    _gt[..., 20:30, 20:30] = 1
    _edge_maps = [torch.randn(2, 256, _size // 8, _size // 8),
                  torch.randn(2, 64, _size // 4, _size // 4),
                  torch.randn(2, 16, _size // 2, _size // 2),
                  torch.randn(2, 4, _size // 1, _size // 1)]
    print(head(_x, _edge_maps, gt_mask=_gt).shape)
