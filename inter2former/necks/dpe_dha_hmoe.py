import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.models.builder import NECKS

from engine.utils import rearrange, repeat

try:
    from xformers.ops import fmha  # noqa
except ImportError:
    warnings.warn('Cannot import xformers')
    fmha = None

try:
    from inter2former.cpp_extension.fast_moe import _fast_moe_linear  # noqa
except ImportError:
    warnings.warn('Cannot import cpp-based fast_moe, use PyTorch instead')
    _fast_moe_linear = None

from engine.utils.zoom_in import get_bbox_from_mask
try:
    from inter2former.cpp_extension.fast_mask_convert import _fast_mask_convert  # noqa
    get_bbox_from_mask_cpu = _fast_mask_convert.get_bbox_from_mask
except ImportError:
    warnings.warn('Cannot import cpp-based fast_mask_convert, use PyTorch instead')
    _fast_mask_convert = None


REF_DEFINITELY_BACKGROUND = 0
REF_POSSIBLY_BACKGROUND = 1
REF_UNKNOWN = 2
REF_POSSIBLY_FOREGROUND = 3
REF_DEFINITELY_FOREGROUND = 4


class BSQAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 code_size=8,
                 beta=0.25):
        super(BSQAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.code_size = code_size
        self.beta = beta

        self.code_proj = nn.Linear(embed_dim, code_size)

        self.register_buffer(
            'pos_one', torch.ones(1, code_size, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            'neg_one', torch.ones(1, code_size, dtype=torch.float32) * -1,
            persistent=False
        )
        self.codebook = nn.Parameter(
            nn.Embedding(2 * code_size, embed_dim).weight)
        self.register_buffer(
            'code_indices', torch.arange(2 ** code_size).long(),
            persistent=False
        )
        self.register_buffer(
            'code_base',
            (2 ** torch.arange(code_size - 1, -1, -1)).long(),
            persistent=False
        )

    def forward(self,
                q, k, v,
                lengths,
                mode='eval',
                inv_lengths=None):
        """
        :param q: L, C
        :param k: L, C
        :param v: L, C
        :param lengths: List[int], indicate length of tokens for each sample
        :param mode: 'eval' or 'train'
        :param inv_lengths: List[int]
        :return:  L, C
        """
        code = F.normalize(self.code_proj(k), p=2, dim=-1)
        sign = torch.where(  # noqa
            code >= 0,
            self.pos_one.expand_as(code),
            self.neg_one.expand_as(code))
        sign = sign / self.code_size ** 0.5
        if mode == 'train':
            loss = F.mse_loss(sign, code) * self.beta
            code = (sign - code).detach() + code
        else:
            code = sign
        code = code * self.code_size ** 0.5 / 2 + 0.5
        codebook = self.codebook
        if mode == 'train':
            code = torch.cat([code, 1 - code], dim=-1)
            k = code @ codebook
            if fmha is not None and q.is_cuda:
                q = rearrange(q, 'l (nh d) -> () l nh d', d=self.head_dim)
                k = rearrange(k, 'l (nh d) -> () l nh d', d=self.head_dim)
                v = rearrange(v, 'l (nh d) -> () l nh d', d=self.head_dim)
                attn_bias = fmha.BlockDiagonalMask.from_seqlens(lengths)  # noqa
                x = fmha.memory_efficient_attention(  # noqa
                    q, k, v, scale=self.scale, attn_bias=attn_bias)
                x = rearrange(x, '() l nh d -> l (nh d)')
            else:
                raise NotImplementedError
            return x, loss  # noqa
        else:
            if inv_lengths is None:
                raise ValueError(
                    '`inv_lengths` should be provided during `eval`')
            idxs = (code.long() * self.code_base).sum(-1, keepdim=True)
            base_code = ((self.code_indices[..., None] //
                          self.code_base) % 2)
            base_code = torch.cat([base_code, 1 - base_code], dim=-1)
            codebook = base_code.float() @ codebook
            codebook = rearrange(codebook, 'k c -> () k c')
            cod_k = codebook.expand(len(lengths), -1, -1)
            cod_v = torch.zeros_like(cod_k)
            cod_c = q.new_zeros(*cod_k.shape[:-1], 1)
            idxs_list = idxs.split(lengths, dim=0)
            v_list = v.split(lengths, dim=0)
            for batch_idx, (idxs, v) in enumerate(
                zip(idxs_list, v_list)
            ):
                cod_v[batch_idx].scatter_add_(
                    0, idxs.expand_as(v), v)
                cod_c[batch_idx].scatter_add_(
                    0, idxs, torch.ones_like(idxs).float())
            v = torch.cat([cod_v, cod_c], dim=-1)
            mask = (v[..., -1] > 0)
            k, v = cod_k[mask], v[mask]
            if fmha is not None and q.is_cuda:
                q = rearrange(
                    q, 'l (nh d) -> () l nh d', d=self.head_dim)
                k = rearrange(
                    k, 'n (nh d) -> () n nh d', d=self.head_dim)
                v, c = v[..., :-1], v[..., -1:]
                v = rearrange(
                    v, 'n (nh d) -> () n nh d', d=self.head_dim)
                c = rearrange(c, 'n () -> () n () ()')
                c = c.expand_as(v).contiguous()
                attn_bias = fmha.BlockDiagonalMask.from_seqlens(  # noqa
                    inv_lengths, mask.sum(dim=-1).tolist())
                x = fmha.memory_efficient_attention(  # noqa
                    q, k, v,
                    scale=self.scale, attn_bias=attn_bias) / \
                    fmha.memory_efficient_attention(  # noqa
                    q, k, c,
                    scale=self.scale, attn_bias=attn_bias)
                x = rearrange(x, '() l nh d -> l (nh d)')
                return x
            else:
                if len(lengths) == 1:
                    qs = [q]
                    ks = [k]
                    vs = [v]
                else:
                    qs = q.split(inv_lengths, dim=0)
                    kv_lengths = mask.sum(dim=-1).tolist()
                    ks = k.split(kv_lengths, dim=0)
                    vs = v.split(kv_lengths, dim=0)
                outs = []
                for q, k, v in zip(qs, ks, vs):
                    q = rearrange(q, 'l (nh d) -> nh l d', d=self.head_dim)
                    k = rearrange(k, 'l (nh d) -> nh d l', d=self.head_dim)
                    v, c = v[..., :-1], v[..., -1:]
                    v = rearrange(v, 'l (nh d) -> nh l d', d=self.head_dim)
                    c = rearrange(c, 'n () -> () n ()')
                    c = c.expand(v.shape[0], -1, -1)
                    attn = ((self.scale * q) @ k).softmax(dim=-1)
                    out = (attn @ v) / (attn @ c)
                    out = rearrange(out, 'nh l d -> l (nh d)')
                    outs.append(out)
                x = torch.cat(outs, dim=0)
                return x


class FullAttention(nn.Module):

    ROPE_CACHE_SIZE = 4

    def __init__(self,
                 embed_dim,
                 num_heads,
                 ratio=0.5,
                 theta=10000):
        super(FullAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        dim = self.head_dim // 2
        self.register_buffer('freqs', theta ** (
            -torch.arange(0, dim, 2)[:dim // 2].float() / dim
        ), persistent=False)
        self.ratio = ratio

        self.rope_cache = dict()

    def rope(self, x, idxs, hw_shape, batch_size):
        if x.is_cuda:
            key = str(hw_shape) + '-cuda'
        else:
            key = str(hw_shape) + '-cpu'
        if key in self.rope_cache:
            freqs_cos, freqs_sin = self.rope_cache[key]
        else:
            if len(self.rope_cache) >= self.ROPE_CACHE_SIZE:
                self.rope_cache.clear()
            ys, xs = torch.meshgrid(
                torch.arange(hw_shape[0],
                             device=x.device).float() * self.ratio,
                torch.arange(hw_shape[1],
                             device=x.device).float() * self.ratio,
                indexing='ij'
            )
            freqs_h = torch.einsum('..., f -> ... f', ys, self.freqs)
            freqs_w = torch.einsum('..., f -> ... f', xs, self.freqs)
            freqs = torch.cat([freqs_h, freqs_w], dim=-1)
            freqs = repeat(freqs, 'h w n -> (h w) () (n r)', r=2)
            freqs_cos, freqs_sin = freqs.cos(), freqs.sin()
            self.rope_cache[key] = freqs_cos, freqs_sin
        freqs_cos = freqs_cos.repeat(batch_size, 1, 1)
        freqs_sin = freqs_sin.repeat(batch_size, 1, 1)
        if idxs is not None:
            freqs_cos = freqs_cos[idxs]
            freqs_sin = freqs_sin[idxs]
        return x * freqs_cos + self.rotate_half(x) * freqs_sin

    @staticmethod
    def rotate_half(x):
        new_x = x.new_empty(x.shape)
        new_x[..., 0::2] = x[..., 1::2]
        new_x[..., 1::2] = -x[..., 0::2]
        return new_x

    def forward(self,
                q,
                k,
                v,
                lengths,
                idxs,
                hw_shape,
                mode='eval'):
        """
        :param q: L1, C
        :param k: L2, C
        :param v: L2, C
        :param lengths: List[int], indicate length of tokens for each sample
        :param idxs
        :param hw_shape: (H, W)
        :param mode: 'eval' or 'train'
        :return:  L1, C
        """
        H, W = hw_shape
        rope = partial(self.rope, hw_shape=hw_shape, batch_size=len(lengths))
        if fmha is not None and q.is_cuda:
            q = rearrange(q, 'l (nh d) -> () l nh d', d=self.head_dim)
            k = rearrange(k, 'l (nh d) -> () l nh d', d=self.head_dim)
            v = rearrange(v, 'l (nh d) -> () l nh d', d=self.head_dim)
            q, k = rope(q, idxs=idxs), rope(k, idxs=None)
            attn_bias = fmha.BlockDiagonalMask.from_seqlens(  # noqa
                lengths, len(lengths) * [H * W])
            x = fmha.memory_efficient_attention(  # noqa
                q, k, v, scale=self.scale, attn_bias=attn_bias)
            x = rearrange(x, '() l nh d -> l (nh d)')
        else:
            q = rearrange(q, 'l (nh d) -> l nh d', d=self.head_dim)
            k = rearrange(k, 'l (nh d) -> l nh d', d=self.head_dim)
            q, k = rope(q, idxs=idxs), rope(k, idxs=None)
            q = rearrange(q, 'l nh d -> nh l d', d=self.head_dim)
            k = rearrange(k, 'l nh d -> nh l d', d=self.head_dim)
            v = rearrange(v, 'l (nh d) -> nh l d', d=self.head_dim)
            if len(lengths) == 1:
                qs = [q]
                ks = [k]
                vs = [v]
            else:
                qs = q.split(lengths, dim=1)
                ks = k.split(len(lengths) * [H * W], dim=1)
                vs = v.split(len(lengths) * [H * W], dim=1)
            outs = []
            for q, k, v in zip(qs, ks, vs):
                attn = ((self.scale * q) @ k.transpose(-2, -1)).softmax(-1)
                outs.append(rearrange(attn @ v, 'nh l d -> l (nh d)'))
            x = torch.cat(outs, dim=0)
        return x


class SwiGLU(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.SiLU()
        self.w3 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, ffn_ln):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = ffn_ln(self.act(x1) * x2)
        x = self.w3(x)
        return x


class HybridMoE(nn.Module):

    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 num_experts,
                 lr=1e-3,
                 max_update_steps=100_000):
        super(HybridMoE, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.lr = lr

        # Shared expert components
        self.shared_ffn = SwiGLU(embed_dim, hidden_dim // num_experts)
        self.shared_ln = nn.LayerNorm(hidden_dim // num_experts)
        self.shared_gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Routed experts components
        select_ffn_list = nn.ModuleList([
            SwiGLU(embed_dim, hidden_dim // num_experts)
            for _ in range(num_experts)
        ])
        self.select_ffn_w1 = nn.Parameter(
            torch.stack([e.w1.weight for e in select_ffn_list]))
        self.select_ffn_b1 = nn.Parameter(
            torch.stack([e.w1.bias for e in select_ffn_list]))
        self.select_ffn_w2 = nn.Parameter(
            torch.stack([e.w2.weight for e in select_ffn_list]))
        self.select_ffn_b2 = nn.Parameter(
            torch.stack([e.w2.bias for e in select_ffn_list]))
        self.select_ffn_w3 = nn.Parameter(
            torch.stack([e.w3.weight for e in select_ffn_list]))
        self.select_ffn_b3 = nn.Parameter(
            torch.stack([e.w3.bias for e in select_ffn_list]))
        self.select_ln = nn.LayerNorm(hidden_dim // num_experts)
        self.expert_gate = nn.Sequential(
            nn.Linear(embed_dim, num_experts),
            nn.Sigmoid()
        )

        # Expert balancing parameters
        self.register_buffer('exp_bias', torch.zeros(num_experts))
        self.register_buffer('local_counts',
                             torch.zeros(num_experts, dtype=torch.long),
                             persistent=False)
        self.register_buffer('global_counts',
                             torch.zeros(num_experts, dtype=torch.long),
                             persistent=False)

        self.max_update_steps = max_update_steps
        self.register_buffer('steps', torch.tensor(0))

    @torch.no_grad()
    def _update_bias(self):
        """Update expert bias based on global statistics"""
        total = self.global_counts.sum().float()
        target = total / self.num_experts
        delta = (self.global_counts.float() - target).sign() * self.lr
        self.exp_bias -= delta  # Decrease bias for overloaded experts
        self.local_counts.zero_()
        self.global_counts.zero_()

    def _sync_counts(self):
        """Synchronize expert usage across devices"""
        if dist.is_initialized():
            counts = self.local_counts.float()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            self.global_counts = counts.long()
        else:
            self.global_counts.copy_(self.local_counts)

    def _cpp_fast_moe_linear(self, x, expert_indices):
        sorted_indices = torch.argsort(expert_indices)
        expert_indices_sorted = expert_indices[sorted_indices]
        x_sorted = x[sorted_indices].contiguous()

        unique_experts, counts = \
            torch.unique(expert_indices_sorted, return_counts=True)

        full_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device)
        full_counts[unique_experts] = counts

        expert_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            full_counts.cumsum(0)
        ])

        x1 = _fast_moe_linear.forward_sorted(  # noqa
            x_sorted,
            expert_offsets,
            self.select_ffn_w1,
            self.select_ffn_b1
        )
        x2 = _fast_moe_linear.forward_sorted(  # noqa
            x_sorted,
            expert_offsets,
            self.select_ffn_w2,
            self.select_ffn_b2
        )

        x = self.shared_ffn.act(x1) * x2
        x = self.select_ln(x)

        output_sorted = _fast_moe_linear.forward_sorted(  # noqa
            x.contiguous(),
            expert_offsets,
            self.select_ffn_w3,
            self.select_ffn_b3
        )

        return output_sorted[torch.argsort(sorted_indices)]

    def forward(self, x, index, mode='eval'):
        # Shared expert computation (always active)
        shared_out = self.shared_ffn(x, self.shared_ln)

        if index is None or index[0].numel() == 0:
            return shared_out

        # Process routed tokens
        x_moe = x[index]  # [S, C]

        # Compute gating scores
        shared_scores = self.shared_gate(x_moe)  # [S, 1]
        expert_scores = self.expert_gate(x_moe)  # [S, E]

        # Expert selection (top-1 routing)
        _, top1_idx = (expert_scores + self.exp_bias).topk(1, dim=-1)  # [S, 1]
        top1_score = expert_scores.gather(-1, top1_idx)  # [S, 1]
        expert_indices = top1_idx.squeeze(-1)  # [S]

        # Combine scores for normalization
        combined_scores = torch.cat([shared_scores, top1_score], dim=-1)
        weights = combined_scores.softmax(dim=-1)
        shared_weight, expert_weight = weights.unbind(dim=-1)

        # Compute shared expert contribution
        shared_moe_out = shared_out[index] * shared_weight.unsqueeze(-1)

        if mode == 'eval' and not x.is_cuda and _fast_moe_linear is not None:
            expert_moe_out = self._cpp_fast_moe_linear(x_moe, expert_indices)
            expert_moe_out = expert_moe_out * expert_weight.unsqueeze(-1)
        else:
            expert_moe_out = torch.zeros_like(x_moe)
            selected_experts = torch.unique(expert_indices)

            for expert_id in selected_experts:
                mask = expert_indices == expert_id
                if mask.sum() == 0:  # noqa
                    continue
                w1 = partial(
                    F.linear,
                    weight=self.select_ffn_w1[expert_id],
                    bias=self.select_ffn_b1[expert_id])
                w2 = partial(
                    F.linear,
                    weight=self.select_ffn_w2[expert_id],
                    bias=self.select_ffn_b2[expert_id])
                w3 = partial(
                    F.linear,
                    weight=self.select_ffn_w3[expert_id],
                    bias=self.select_ffn_b3[expert_id])
                z = x_moe[mask]
                z = self.shared_ffn.act(w1(z)) * w2(z)
                z = w3(self.select_ln(z))
                expert_moe_out[mask] = z * expert_weight[mask].unsqueeze(-1)

        # Update statistics
        if mode == 'train' and self.steps.item() < self.max_update_steps:
            unique, counts = torch.unique(expert_indices, return_counts=True)
            self.local_counts[unique] += counts
            self._sync_counts()
            self._update_bias()
            self.steps.data.add_(1)

        # Combine outputs
        final_out = shared_out.clone()
        final_out[index] = shared_moe_out + expert_moe_out

        return final_out


class Block(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 downscale,
                 code_size=8,
                 beta=0.25,
                 ratio=0.5,
                 theta=10000,
                 lr=1e-3,
                 max_update_steps=100_000,
                 num_ffn_experts=None):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        if downscale > num_heads:
            warnings.warn(
                f'downscale should be less than or equal to num_heads, '
                f'got {downscale} > {num_heads}, set to {num_heads}')
            downscale = num_heads
        self.downscale = downscale

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim // downscale)
        self.o_proj = nn.Linear(embed_dim // downscale, embed_dim)
        self.bsq_attn = BSQAttention(
            embed_dim // downscale,
            num_heads // downscale,
            code_size=code_size,
            beta=beta)
        self.loc_attn = FullAttention(
            embed_dim // downscale,
            num_heads // downscale,
            ratio=ratio,
            theta=theta)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(mlp_ratio * embed_dim)
        if num_ffn_experts is None:
            num_experts = downscale
        else:
            num_experts = num_ffn_experts
        self.ffn = HybridMoE(
            embed_dim, mlp_dim, num_experts,
            lr=lr, max_update_steps=max_update_steps)

        self.register_buffer(
            'gamma', torch.ones(embed_dim).float(), persistent=False)

    def forward(self, x, mode, idxs, bsq_kwargs, loc_kwargs, inv_idxs=None):
        q, k, v = self.qkv_proj(self.norm1(x)).chunk(3, dim=-1)
        if mode == 'train':
            attn, loss = self.bsq_attn(q, k, v, mode=mode, **bsq_kwargs)
            attn = self.o_proj(attn)
            attn = attn.clone()
            loc_attn = self.loc_attn(
                q[idxs], k, v, mode=mode, idxs=idxs, **loc_kwargs)
            loc_attn = self.o_proj(loc_attn)
            attn[idxs] = (1 - self.gamma) * attn[idxs] + self.gamma * loc_attn
            x = x + attn
            x = x + self.ffn(self.norm2(x), idxs, mode)
            return x, loss
        else:
            if inv_idxs is None:
                raise ValueError('`inv_idxs` should be provided during `eval`')
            attn = x.new_zeros(len(x), self.embed_dim // self.downscale)
            if inv_idxs[0].nelement() > 0:
                attn[inv_idxs] = self.bsq_attn(
                    q[inv_idxs], k, v, mode=mode, **bsq_kwargs)
            attn[idxs] = self.loc_attn(
                q[idxs], k, v, mode=mode, idxs=idxs, **loc_kwargs)
            x = x + self.o_proj(attn)
            x = x + self.ffn(self.norm2(x), idxs, mode)
            return x


class Rearrange(nn.Module):

    def __init__(self, pattern, **kwargs):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kwargs)


class PatchConv(nn.Module):

    def __init__(self, in_dim, dims, strides):
        super(PatchConv, self).__init__()
        self.stem = nn.Sequential()
        self.stride = 1
        for idim, odim, st in zip([in_dim] + list(dims[:-1]), dims, strides):
            self.stem.append(
                nn.Conv2d(idim, odim, (st - 1) * 2 + 1,
                          stride=st, padding=(st - 1))
            )
            self.stem.append(nn.ReLU())
            self.stride *= st

    def forward(self, x, target_hw=None):
        if target_hw is not None:
            h, w = target_hw
            h, w = h * self.stride, w * self.stride
            if x.shape[-2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode='area')
        return self.stem(x)


class UncertaintyMeasure(nn.Module):

    def __init__(self, kernel_size=3):
        super(UncertaintyMeasure, self).__init__()
        self.kernel_size = kernel_size
        self.register_buffer(
            'kernel',
            torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2,
            persistent=False)

    def forward(self, x):
        ks = self.kernel_size
        x = F.pad(
            x.float(), (ks // 2, ks // 2, ks // 2, ks // 2), mode='reflect')
        var = F.conv2d(x ** 2, self.kernel) - F.conv2d(x, self.kernel) ** 2
        return var


@NECKS.register_module()
class Inter2FormerDecoderNeck(nn.Module):

    NUM_REF_MODES = 5

    def __init__(self,
                 downscale,  # e.g., 4
                 depth,
                 embed_dim,
                 num_heads,
                 ref_dims,
                 ref_strides,
                 code_size=8,
                 mlp_ratio=4.0,
                 beta=0.25,
                 uncertainty_kernel_size=3,
                 lr=1e-3,
                 max_update_steps=100_000,
                 num_ffn_experts=None,
                 pre_downsample=1):
        super(Inter2FormerDecoderNeck, self).__init__()
        if pre_downsample > 1:
            warnings.warn(f'pre_downsample is ignored: {pre_downsample} > 1')
        self.pre_downsample = pre_downsample
        self.ref_embeds = nn.Embedding(self.NUM_REF_MODES, self.NUM_REF_MODES)
        self.ref_conv = PatchConv(self.NUM_REF_MODES, ref_dims, ref_strides)
        self.stride = self.ref_conv.stride
        self.uncertainty = UncertaintyMeasure(uncertainty_kernel_size)
        self.blocks = nn.ModuleList([
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  downscale,
                  code_size,
                  beta,
                  lr=lr,
                  max_update_steps=max_update_steps,
                  num_ffn_experts=num_ffn_experts)
            for _ in range(depth)
        ])

    def get_back_embed(self, x):
        """
        used in DPE
        """
        grid_size = 3
        stride = self.stride
        size = grid_size * stride
        back_embed = self.ref_embeds(
            torch.tensor(REF_POSSIBLY_BACKGROUND, device=x.device).long())
        back_embed = repeat(back_embed, 'c -> () c h w', h=size, w=size)
        back_embed = self.ref_conv(back_embed)[0, :, grid_size // 2, grid_size // 2]
        return back_embed

    def forward(self,
                x,
                ref_label,
                mode='eval',
                return_loss=None,
                gt_sem_seg=None):
        if return_loss is not None:
            warnings.warn(
                'return_loss is not used, please set `mode` to `train` instead'
            )
            mode = 'train'
        if mode == 'train' and gt_sem_seg is None:
            warnings.warn('`gt_sem_seg` is not provided during `train` mode')
        elif mode == 'eval' and gt_sem_seg is not None:
            warnings.warn('`gt_sem_seg` is not used during `eval` mode')
        if isinstance(x, (tuple, list)):
            x = x[-1]

        if ref_label.ndim == 3:
            ref_label = rearrange(ref_label, 'b h w -> b () h w')
        elif ref_label.ndim != 4:
            raise ValueError(f'Cannot handle `ref_label` of '
                             f'shape {tuple(ref_label.shape)}')

        # DPE process
        ori_ref_label = ref_label
        H, W = ref_label.shape[-2:]
        ref_embed = repeat(
            self.get_back_embed(x), 'c -> b c h w',
            b=len(ref_label), h=x.size(-2), w=x.size(-1))
        mask = torch.zeros_like(ref_label).float()
        st = self.stride
        if _fast_mask_convert is None:
            bboxes = get_bbox_from_mask(
                (ref_label != REF_POSSIBLY_BACKGROUND).to(ref_label),
                st
            ).view(-1, 4).long()
        else:
            bboxes = get_bbox_from_mask_cpu(
                (ref_label != REF_POSSIBLY_BACKGROUND).to(ref_label),
                st
            ).view(-1, 4).long()
        left, top, right, bottom = bboxes.unbind(dim=-1)
        bboxes = torch.stack([
            (left - 1 * st).clamp(0, W),
            (top - 1 * st).clamp(0, H),
            (right + 1 * st).clamp(0, W),
            (bottom + 1 * st).clamp(0, H)
        ], dim=-1)
        if len(ref_label) == 1:
            ref_label_list = ref_label  # [(1, H, W)]
        else:
            ref_label_list = ref_label.unbind(dim=0)  # [(1, H, W), ...]
        for i, (label, bbox) in enumerate(zip(ref_label_list, bboxes)):
            left, top, right, bottom = bbox
            label = label[..., top:bottom, left:right]
            raw_embed = self.ref_embeds(label.long())
            raw_embed = rearrange(raw_embed, '() h w c -> () c h w')
            mask[i:i+1, :, top:bottom, left:right] = \
                (self.uncertainty(label.float()) > 0.0).float()
            left, top, right, bottom = bbox // st
            ref_embed[i:i+1, :, top:bottom, left:right] = \
                self.ref_conv(raw_embed)
        x = x + ref_embed

        mask[(ori_ref_label == REF_UNKNOWN)] = 1.0
        mask = F.adaptive_max_pool2d(mask, x.shape[-2:])
        mask = rearrange(mask, 'b () h w -> b h w')
        mask_flat = mask.flatten()
        idxs = torch.nonzero(mask_flat, as_tuple=True)
        if mode == 'eval':
            inv_idxs = torch.nonzero(
                (mask_flat == 0).to(mask_flat), as_tuple=True)

        B, _, H, W = x.shape
        full_lengths = B * [H * W]
        part_lengths = mask.sum(dim=[-2, -1]).long().tolist()
        bsq_kwargs = dict(lengths=full_lengths,
                          inv_lengths=[
                              f - p for f, p in
                              zip(full_lengths, part_lengths)
                          ])
        loc_kwargs = dict(lengths=part_lengths, hw_shape=(H, W))
        x = rearrange(x, 'b c h w -> (b h w) c')
        losses = []
        for block in self.blocks:
            if mode == 'train':
                x, loss = block(x, mode, idxs, bsq_kwargs, loc_kwargs)
                losses.append(loss)
            else:
                x = block(x, mode, idxs, bsq_kwargs, loc_kwargs, inv_idxs)
        x = rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W)

        if mode == 'train':
            return x, dict(vq_loss=sum(losses) / len(losses))
        else:
            return x


if __name__ == '__main__':
    _dim = 256
    model = Inter2FormerDecoderNeck(
        downscale=16,
        depth=2,
        embed_dim=_dim,
        num_heads=8,
        ref_dims=(32, 32, 64, 128, 256),
        ref_strides=(1, 2, 2, 2, 2),
        code_size=8,
        mlp_ratio=4.0)
    model.eval()
    with torch.no_grad():
        _x = torch.randn(2, _dim, 64, 64)
        _ref_label = torch.ones(2, 1, 1024, 1024) * REF_POSSIBLY_BACKGROUND
        _ref_label[..., 100:200, 100:200] = REF_POSSIBLY_FOREGROUND

        _cpu_eout = model(_x, _ref_label, mode='eval')
        print(f'cpu out: {_cpu_eout.shape}')

        model = model.cuda()
        _x = _x.cuda()
        _ref_label = _ref_label.cuda()

        _eout = model(_x, _ref_label, mode='eval')
        print(_eout.shape)
        _tout, _loss = model(_x, _ref_label, mode='train')
        print(_tout.shape, _loss)
        print(f'diff: {(_tout - _eout).abs().mean().item()}')
        print(f'cpu diff: {(_cpu_eout - _eout.cpu()).abs().mean().item()}')
