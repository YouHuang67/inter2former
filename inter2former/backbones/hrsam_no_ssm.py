import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mmengine.utils.misc import to_2tuple
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from timm.models.layers import drop_path as timm_drop_path
from engine.utils import rearrange, repeat

try:
    from xformers.ops import fmha  # noqa
except ImportError:
    warnings.warn('Cannot import xformers')
    fmha = None


class Rearrange(nn.Module):

    def __init__(self, pattern, **kwargs):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kwargs)


class Permute(nn.Module):

    def __init__(self, *pattern):
        super(Permute, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return x.permute(*self.pattern)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def compute_padding_size(size, window_size):
    pad = (window_size - size % window_size) % window_size
    return size + pad, pad


def expand_and_concat(tensors, dim=-1):
    assert all(tensors[0].ndim == tensor.ndim for tensor in tensors), \
        "All tensors must have the same number of dimensions"
    dim = tuple(range(tensors[0].ndim))[dim]
    tensors = [
        tensor.expand(*[
            max_size if i != dim else -1
            for i, max_size in enumerate(
                map(max, zip(*[tensor.shape for tensor in tensors]))  # noqa
            )
        ]) for tensor in tensors
    ]
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim, ratio=0.5, theta=10000):
        super(VisionRotaryEmbedding, self).__init__()
        self.register_buffer('freqs', theta ** (
            -torch.arange(0, dim, 2)[:dim // 2].float() / dim
        ))
        self.ratio = ratio
        self.cache = dict()

    def forward(self,
                x,
                mode,  # 'query' or 'key'
                batch_size,
                hw_shapes,
                pad_hw_shapes,
                window_size,
                shift,
                padding_list,
                q_indices,
                **kwargs):  # noqa
        if mode not in ['query', 'key']:
            raise ValueError(f'Unsupported mode: {mode}')
        key = f'{mode}-{batch_size}-{hw_shapes}-{window_size}-{shift}'
        if key in self.cache:
            freqs_cos, freqs_sin = self.cache[key]
            return x * freqs_cos + self.rotate_half(x) * freqs_sin
        elif len(self.cache) == 8:
            self.cache.clear()

        freqs_list = []
        for i, (H, W) in enumerate(pad_hw_shapes):
            freqs_h = torch.einsum(
                '..., f -> ... f',
                torch.arange(H, device=x.device) * self.ratio,
                self.freqs)
            freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # H, C // 2

            freqs_w = torch.einsum(
                '..., f -> ... f',
                torch.arange(W, device=x.device) * self.ratio,
                self.freqs)
            freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # W, C // 2

            freqs = expand_and_concat(
                [rearrange(freqs_h, 'd ... -> d () ...'),
                 rearrange(freqs_w, 'd ... -> () d ...')],
                dim=-1
            )  # H, W, C
            Ws1, Ws2 = window_size
            if mode == 'query':
                Ph, Pw = padding_list[i]
                h, w = hw_shapes[i]
                freqs = freqs[Ph:Ph + h, Pw:Pw + w, :]
                freqs = rearrange(freqs, 'h w ... -> (h w) ...')
            elif mode == 'key':
                freqs = rearrange(
                    freqs, '(h ws1) (w ws2) ... -> (h w ws1 ws2) ...',
                    ws1=Ws1, ws2=Ws2)
            else:
                raise NotImplementedError
            freqs_list.append(freqs)

        freqs = torch.cat(freqs_list, dim=0)
        freqs = repeat(freqs, 'l c -> (b l) () c', b=batch_size)
        if mode == 'query':
            freqs = freqs[q_indices]
        elif mode != 'key':
            raise NotImplementedError
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        self.cache[key] = (freqs_cos, freqs_sin)
        return x * freqs_cos + self.rotate_half(x) * freqs_sin

    @staticmethod
    def rotate_half(x):
        new_x = x.new_empty(x.shape)
        new_x[..., 0::2] = x[..., 1::2]
        new_x[..., 1::2] = -x[..., 0::2]
        return new_x


class DropPath(nn.Module):

    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return timm_drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_xformer=True):
        super(WindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        hidden_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.padding = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.use_xformer = use_xformer

    def forward(self,
                x,
                q_indices,
                kv_indices,
                inv_indices,
                rope,
                attn_bias,
                window_size,
                **kwargs):  # noqa
        x = torch.cat([x, self.padding], dim=0)
        q, k, v = rearrange(
            self.qkv(x),
            'bl (n3 n d) -> bl n3 n d',
            n3=3, n=self.num_heads).unbind(dim=-3)
        if self.use_xformer and x.is_cuda:
            q = rope(q[q_indices], mode='query').unsqueeze(0)
            k = rope(k[kv_indices], mode='key').unsqueeze(0)
            v = v[kv_indices].unsqueeze(0)
            x = fmha.memory_efficient_attention(  # noqa
                q, k, v, attn_bias=attn_bias, scale=self.scale)
            x = rearrange(x, '() bl n d -> bl (n d)')[inv_indices]
        else:
            Ws = window_size[0] * window_size[1]
            q = rope(q[kv_indices], mode='key')
            q = rearrange(q, '(nw ws) n d -> nw n ws d', ws=Ws)
            k = rope(k[kv_indices], mode='key')
            k = rearrange(k, '(nw ws) n d -> nw n d ws', ws=Ws)
            v = v[kv_indices]
            v = rearrange(v, '(nw ws) n d -> nw n ws d', ws=Ws)
            attn = ((self.scale * q) @ k).softmax(dim=-1)
            ori_x, x = x.new_empty(x.shape), attn @ v
            ori_x[kv_indices] = rearrange(x, 'nw n ws d -> (nw ws) (n d)')
            x = ori_x[:-1]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 use_xformer=True):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_xformer=use_xformer)

        self.drop_path = \
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embed_dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, **kwargs):
        """
        params x: (B * L, C)
        return:   (B * L, C)
        """
        x = x + self.drop_path(self.attn(self.norm1(x), **kwargs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_dim=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        H, W = to_2tuple(img_size)
        pH, pW = to_2tuple(patch_size)
        num_patches = (W // pW) * (H // pH)
        self.patch_shape = (H // pH, W // pW)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):  # noqa
        P = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b h w (c p1 p2)', p1=P, p2=P)
        weight = rearrange(self.proj.weight, 'd c p1 p2 -> d (c p1 p2)')
        bias = self.proj.bias
        x = F.linear(x, weight, bias)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


@MODELS.register_module()
class HRSAMViTNoSSM(BaseModule):

    """ No codebook """

    def __init__(self,

                 downsample_sizes,

                 window_size,
                 in_dim,
                 img_size,
                 patch_size,
                 depth,
                 embed_dim,
                 num_heads,
                 mlp_ratio,

                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,

                 out_indices,

                 use_checkpoint=False,
                 pretrained=None,
                 init_cfg=None,

                 rope_ratio=0.5,
                 use_xformer=True,

                 final_embed_dim=256,

                 flexible_resize=False):

        super(HRSAMViTNoSSM, self).__init__(init_cfg)

        self.downsample_sizes = downsample_sizes
        self.flexible_resize = flexible_resize

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_dim=in_dim, embed_dim=embed_dim)
        self.patch_shape = self.patch_embed.patch_shape

        self.patch_size = patch_size
        self.window_size = window_size
        assert window_size > 0, 'window_size must be larger than 0'

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        half_head_dim = embed_dim // num_heads // 2
        self.rope = VisionRotaryEmbedding(dim=half_head_dim, ratio=rope_ratio)

        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint

        if use_xformer and fmha is None:
            warnings.warn('xformers is not installed, using xformer=False')
            use_xformer = False

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=torch.linspace(0, drop_path_rate, depth)[i].item(),
                  use_xformer=use_xformer)
            for i in range(depth)])

        num_scales = len(downsample_sizes) + 1
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                Rearrange('(ns b) ... d -> b ... (ns d)', ns=num_scales)
            ) for _ in out_indices])

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_scales * embed_dim, final_embed_dim, bias=False),
                nn.LayerNorm(final_embed_dim),
                Rearrange('b h w c -> b c h w'))
            for _ in out_indices])
        self.out_conv = nn.Sequential(
            nn.Conv2d(final_embed_dim, final_embed_dim,
                      kernel_size=3, padding=1, bias=False,
                      groups=final_embed_dim),
            Transpose(1, -1),
            nn.LayerNorm(final_embed_dim),
            Transpose(1, -1))

        self.indices_cache = dict()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        ori_pos_embed = rearrange(
            self.pos_embed[:, 1:], '() (h w) c -> () c h w',
            h=self.patch_shape[0], w=self.patch_shape[1])

        xs, hw_shapes = [], []
        (H, W), ori_x = x.shape[-2:], x
        if self.flexible_resize:
            sizes = [(H, W)]
            Ps = self.patch_size
            for ds in self.downsample_sizes:
                ds1, ds2 = to_2tuple(ds)
                assert ds1 == ds2, 'Only support square downsample sizes'
                assert ds1 % Ps == 0, 'Downsample size must be divisible by patch size'
                if H >= W:
                    h = ds1
                    w = (int(W / (H / ds1)) + Ps - 1) // Ps * Ps
                else:
                    w = ds1
                    h = (int(H / (W / ds1)) + Ps - 1) // Ps * Ps
                sizes.append((h, w))
        else:
            sizes = [(H, W)] + list(map(to_2tuple, self.downsample_sizes))
        for h, w in sizes:
            if (h, w) != (H, W):
                x = F.interpolate(
                    ori_x, size=(h, w), mode='bilinear',
                    align_corners=False)
            else:
                x = ori_x
            x = self.patch_embed(x)
            *_, Hp, Wp = x.shape
            if ori_pos_embed.shape[-2:] != (Hp, Wp):
                pos_embed = F.interpolate(
                    ori_pos_embed, size=(Hp, Wp),
                    mode='bilinear', align_corners=False)
            else:
                pos_embed = ori_pos_embed
            x = self.pos_drop(x + pos_embed)
            xs.append(rearrange(x, 'b c h w -> b (h w) c'))
            hw_shapes.append((Hp, Wp))
        x = torch.cat(xs, dim=1)

        Bs = len(x)
        Ws = self.window_size
        window_size = to_2tuple(Ws)

        # make plain window arguments
        shift = 0
        plain_kwargs = self.make_indices(
            x,
            hw_shapes=hw_shapes,
            window_size=window_size,
            shift=shift,
            indices_cache=self.indices_cache)
        q_lengths = plain_kwargs['q_lengths']
        kv_lengths = plain_kwargs['kv_lengths']
        plain_window_kwargs = dict(
            rope=partial(self.rope,
                         batch_size=Bs,
                         window_size=window_size,
                         shift=shift,
                         hw_shapes=hw_shapes,
                         **plain_kwargs),
            attn_bias=(
                fmha.BlockDiagonalMask.from_seqlens(q_lengths, kv_lengths)  # noqa
                if fmha is not None else None
            ),
            window_size=window_size,
            **plain_kwargs
        )

        # make shift window arguments
        shift = Ws // 2
        shift_kwargs = self.make_indices(
            x,
            hw_shapes=hw_shapes,
            window_size=window_size,
            shift=shift,
            indices_cache=self.indices_cache)
        q_lengths = shift_kwargs['q_lengths']
        kv_lengths = shift_kwargs['kv_lengths']
        shift_window_kwargs = dict(
            rope=partial(self.rope,
                         batch_size=Bs,
                         window_size=window_size,
                         shift=shift,
                         hw_shapes=hw_shapes,
                         **shift_kwargs),
            attn_bias=(
                fmha.BlockDiagonalMask.from_seqlens(q_lengths, kv_lengths)  # noqa
                if fmha is not None else None
            ),
            window_size=window_size,
            **shift_kwargs
        )

        embeds = []
        x, out_index = rearrange(x, 'b l c -> (b l) c'), 0
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                kwargs = dict(plain_window_kwargs)
            else:
                kwargs = dict(shift_window_kwargs)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, **kwargs)
            else:
                x = block(x, **kwargs)
            if i in self.out_indices:
                outs = []
                for idx, out in enumerate(
                        rearrange(
                            x, '(b l) c -> b l c', b=Bs
                        ).split([h * w for h, w in hw_shapes], dim=1)
                ):
                    h, w = hw_shapes[idx]
                    if idx > 0:
                        out = rearrange(
                            out, 'b (h w) c -> b c h w', h=h, w=w)
                        out = F.interpolate(
                            out, size=hw_shapes[0], mode='bilinear',
                            align_corners=False)
                        out = rearrange(out, 'b c h w -> b h w c')
                    else:
                        out = rearrange(
                            out, 'b (h w) c -> b h w c', h=h, w=w)
                    outs.append(out)
                out = torch.cat(outs, dim=0)
                out = self.multi_scale_fusion[out_index](out)
                embeds.append(out)
                out_index += 1

        outs = [
            lateral_conv(embed) for lateral_conv, embed in
            zip(self.lateral_convs, embeds)]
        return (self.out_conv(sum(outs)), )

    @staticmethod
    def make_indices(x,
                     hw_shapes,
                     window_size,
                     shift,
                     indices_cache):
        B, device = len(x), x.device
        Ws1, Ws2 = to_2tuple(window_size)
        S1, S2 = to_2tuple(shift)
        key = f'{B}-{hw_shapes}-{Ws1}-{Ws2}-{S1}-{S2}'
        if key in indices_cache:
            return dict(**indices_cache[key])

        if Ws1 <= S1 or Ws2 <= S2:
            raise ValueError
        if len(indices_cache) >= 8:
            indices_cache.clear()
        padding_list = []
        start, base_indices, pad_hw_shapes = 0, [], []
        INF = 2 * sum(max(h + Ws1, w + Ws2) for h, w in hw_shapes) ** 2 * B
        for H, W in hw_shapes:
            if H > Ws1 or W > Ws2:
                s1, s2 = S1, S2
            else:
                s1, s2 = 0, 0
            Ph, Pw = (Ws1 - s1) % Ws1, (Ws2 - s2) % Ws2
            PH, _ = compute_padding_size(Ph + H, Ws1)
            hs = torch.full((PH, ), -INF, device=device).long()
            PW, _ = compute_padding_size(Pw + W, Ws2)
            ws = torch.full((PW, ), -INF, device=device).long()
            hs[Ph:Ph + H] = torch.arange(H, device=device)
            ws[Pw:Pw + W] = torch.arange(W, device=device)
            hs, ws = torch.meshgrid(hs, ws, indexing='ij')
            idxs = hs * W + ws + start
            base_indices.append(rearrange(
                idxs, '(h ws1) (w ws2) -> (h w) (ws1 ws2)', ws1=Ws1, ws2=Ws2))
            start += H * W
            pad_hw_shapes.append((PH, PW))
            padding_list.append((Ph, Pw))
        base_indices = torch.cat(base_indices, dim=0)
        indices = rearrange(torch.arange(B, device=device), 'b -> b () ()')
        indices = indices * (sum(h * w for h, w in hw_shapes)) + \
                  rearrange(base_indices, 'l w2 -> () l w2')
        indices = rearrange(indices, 'b l w2 -> (b l) w2')
        q_lengths = (indices >= 0).long().sum(dim=-1).tolist()
        kv_lengths = len(indices) * [indices.shape[-1]]
        indices = indices.flatten()
        q_indices = indices[indices >= 0]
        kv_indices = indices.clone()
        kv_indices[kv_indices < 0] = B * sum(h * w for h, w in hw_shapes)
        indices_cache[key] = dict(
            q_indices=q_indices,
            q_lengths=q_lengths,
            kv_indices=kv_indices,
            kv_lengths=kv_lengths,
            pad_hw_shapes=pad_hw_shapes,
            inv_indices=q_indices.argsort(),
            padding_list=padding_list
        )
        return dict(**indices_cache[key])


if __name__ == '__main__':
    model = HRSAMViTNoSSM(

        downsample_sizes=[256],
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
        drop_path_rate=0.2,
        use_checkpoint=False,
        out_indices=(2, 5, 8, 11),

        codebook_cfg=dict(
            codebook_size=256,
            embed_dim=256,
            num_heads=8,
            num_blocks_per_stage=2,
            mlp_ratio=4.
        )
    )

    container = nn.ModuleDict({'backbone': model})
    container.load_state_dict(torch.load(
        'pretrain/hrsam/mae_vit_base_sam_huge_dec.pth', map_location='cpu'),
        strict=False)

    count = 0
    for param in model.parameters():
        count += param.numel()
    if count > 1e6:
        count = count / 1e6
        print(f'Number of parameters: {count:.2f}M')
    elif count > 1e3:
        count = count / 1e3
        print(f'Number of parameters: {count:.2f}K')
    else:
        print(f'Number of parameters: {count}')

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        _x = torch.randn(2, 3, 1024, 1024).cuda()
        _xout = model(_x)
        for module in model.modules():
            if hasattr(module, 'use_xformer'):
                module.use_xformer = False
        _nxout = model(_x)
        for o, no in zip(_xout, _nxout):
            print(f'shape: {tuple(o.shape)}')
            print(f'diff between xformer and no-xformer: '
                  f'{(o - no).abs().mean().item()}')
