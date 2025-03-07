import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import MODELS
from inter2former.edge_detectors.canny import Canny


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        return self.shortcut(x) + h


class BlockSequence(nn.Sequential):

    def __init__(self, dim, depth):
        super(BlockSequence, self).__init__(
            *[ResnetBlock(dim) for _ in range(depth)]
        )


@MODELS.register_module()
class CannyNet(nn.Module):

    def __init__(self,
                 depth,
                 strides,  # (2, 2, 2, 2)
                 downscale=4,
                 threshold=2.0,
                 mean=(123.675, 116.28, 103.53),
                 std=(58.395, 57.12, 57.375),
                 zero_init_lateral_convs=True,
                 downsample=1):
        super(CannyNet, self).__init__()
        self.depth = depth
        self.strides = strides
        self.downscale = downscale
        self.threshold = threshold
        self.canny = Canny(threshold=threshold)
        self.register_buffer(
            'mean', torch.tensor(mean).view(3, 1, 1), persistent=False)
        self.register_buffer(
            'std', torch.tensor(std).view(3, 1, 1), persistent=False)
        self.downsample = downsample

        self.conv1 = nn.Conv2d(5, downscale, 3, stride=1, padding=1)
        self.block_list = nn.ModuleList([
            BlockSequence(downscale ** i, depth)
            for i in range(1, len(strides) + 1)
        ])
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(downscale ** i, downscale ** i, 1)
            for i in range(1, len(strides) + 1)
        ])
        self.downsamplers = nn.ModuleList([
            nn.Conv2d(downscale ** i, downscale ** (i + 1), 3,
                      stride=stride, padding=1)
            for i, stride in enumerate(strides[:-1], 1)
        ])
        if zero_init_lateral_convs:
            for conv in self.lateral_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        """
        :param x: shape (B, 3, H, W)
        :return:
        """
        x = ((x * self.std + self.mean) / 255.0).clamp(0.0, 1.0)
        _, *edge_maps = self.canny(x)
        edge_maps = torch.cat(edge_maps, dim=1)  # (B, 5, H, W)
        edge_maps = F.normalize(edge_maps, p=2, dim=(-2, -1))  # noqa

        x = self.conv1(edge_maps)
        outs = []
        for i, block in enumerate(self.block_list):
            x = block(x)
            outs.append(self.lateral_convs[i](x))
            if i < len(self.strides) - 1:
                x = self.downsamplers[i](x)
        if self.downsample == 1:
            return outs[::-1]
        else:
            return [
                F.interpolate(_, scale_factor=1 / self.downsample, mode='area')
                for _ in outs[::-1]
            ]


if __name__ == '__main__':
    model = CannyNet(
        depth=2, strides=(2, 2, 2, 2), downscale=4, threshold=2.0).cuda()
    _x = torch.randn(2, 3, 512, 512).cuda()
    for _ in model(_x):
        print(_.shape, _.flatten()[:10])
