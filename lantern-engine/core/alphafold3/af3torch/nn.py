# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Primitives shared by the AlphaFold 3 PyTorch port: Haiku-compatible Linear and
LayerNorm, gated linear units, attention, masked means, layout gathers and
chunked application. Each mirrors the function of the same name in
alphafold3.model.components / alphafold3.model.network."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

from .params import Scope

LN_EPS = 1e-5


# ── Linear ────────────────────────────────────────────────────────────────

def linear(x: torch.Tensor, scope: Scope, name: str, *, num_input_dims: int = 1, transpose: bool = False,
           bias: bool = False, out: torch.dtype | None = None) -> torch.Tensor:
    """hm.Linear: weights shaped (in..., out...) or, when transposed, (out..., in...).
    `num_input_dims=0` is the scalar mode (weights shaped like the output, x broadcast)."""
    w = scope.get(name, "weights", dtype=x.dtype)
    if num_input_dims == 0:
        y = x.unsqueeze(-1) * w if w.ndim == 1 else x.reshape(x.shape + (1,) * w.ndim) * w
    else:
        in_letters = "abcde"[:num_input_dims]
        n_out = w.ndim - num_input_dims
        out_letters = "hijkl"[:n_out]
        if transpose:
            eq = f"...{in_letters},{out_letters}{in_letters}->...{out_letters}"
        else:
            eq = f"...{in_letters},{in_letters}{out_letters}->...{out_letters}"
        y = torch.einsum(eq, x, w)
    if bias:
        y = y + scope.get(name, "bias", dtype=x.dtype)
    return y if out is None else y.to(out)


# ── LayerNorm ─────────────────────────────────────────────────────────────

def layer_norm(x: torch.Tensor, scope: Scope, name: str, *, scale: bool = True, offset: bool = True,
               axis: int = -1) -> torch.Tensor:
    """hm.LayerNorm with upcast: (b)float16 inputs are normalised in float32 and cast back.
    Parameters are vectors over `axis` (AF3 uses axis=0 for the triangle centre norm)."""
    dtype = x.dtype
    ax = axis % x.ndim
    if ax == x.ndim - 1:
        # the fused kernel normalises in float32 and returns the input dtype, like the explicit upcast below
        w = scope.get(name, "scale", dtype=torch.float32) if scale else None
        b = scope.get(name, "offset", dtype=torch.float32) if offset else None
        if dtype == torch.float32:
            return F.layer_norm(x, (x.shape[-1],), w, b, LN_EPS)
        return F.layer_norm(x.float(), (x.shape[-1],), w, b, LN_EPS).to(dtype)
    xf = x.float()
    mean = xf.mean(dim=axis, keepdim=True)
    var = (xf - mean).square().mean(dim=axis, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + LN_EPS)
    shape = [1] * x.ndim
    shape[ax] = x.shape[ax]
    if scale:
        y = y * scope.get(name, "scale", dtype=torch.float32).reshape(shape)
    if offset:
        y = y + scope.get(name, "offset", dtype=torch.float32).reshape(shape)
    return y.to(dtype)


# ── gated linear unit ─────────────────────────────────────────────────────

def glu(x: torch.Tensor, w: torch.Tensor, activation: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """tokamax.gated_linear_unit: w is (in, 2, out); out = act(x·w[:,0]) * (x·w[:,1])."""
    h = torch.einsum("...a,aso->...so", x, w.to(x.dtype))
    return activation(h[..., 0, :]) * h[..., 1, :]


def swish(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# ── attention ─────────────────────────────────────────────────────────────

def dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias: torch.Tensor | None = None,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """q, k, v: (..., S, H, D); bias broadcastable to (..., H, Sq, Sk); mask (..., 1, 1, Sk) bool.
    Scale 1/sqrt(D), like jax.nn / tokamax dot_product_attention. Returns (..., Sq, H, D)."""
    qh, kh, vh = (t.transpose(-3, -2) for t in (q, k, v))      # (..., H, S, D)
    attn_mask = None
    if bias is not None:
        attn_mask = bias.to(qh.dtype)
    if mask is not None:
        # jax.nn.dot_product_attention replaces masked logits (bias included) by -0.7·max, so a
        # fully masked row attends uniformly; adding that constant here rounds to the same value.
        large_negative = -0.7 * torch.finfo(qh.dtype).max
        base = attn_mask if attn_mask is not None else torch.zeros((), dtype=qh.dtype, device=qh.device)
        attn_mask = torch.where(mask, base, torch.full((), large_negative, dtype=qh.dtype, device=qh.device))
    if attn_mask is not None and attn_mask.stride(-1) != 1:
        # The fused kernels (flash / memory-efficient / cuDNN) need the mask's last dimension contiguous and
        # otherwise fall back to the math path, which materialises the logits in float32 (20 to 50× slower).
        # Batch and head dimensions may stay broadcast (size 1).
        attn_mask = attn_mask.contiguous()
    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, scale=1.0 / math.sqrt(q.shape[-1]))
    return o.transpose(-3, -2)


# ── masks and means ───────────────────────────────────────────────────────

def mask_mean(mask: torch.Tensor, value: torch.Tensor, dim, keepdim: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """utils.mask_mean: masked mean over `dim`, with broadcast factors for size-1 mask axes."""
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    factor = 1.0
    for d in dims:
        if mask.shape[d] == 1 and value.shape[d] != 1:
            factor *= value.shape[d]
    num = (mask * value).sum(dim=dims, keepdim=keepdim)
    den = torch.clamp(mask.sum(dim=dims, keepdim=keepdim) * factor, min=eps)
    return num / den


# ── atom layout gathers ───────────────────────────────────────────────────

class GatherInfo:
    """atom_layout.GatherInfo on the device: flat indices into a layout plus a mask."""

    __slots__ = ("gather_idxs", "gather_mask", "input_shape")

    def __init__(self, gather_idxs: torch.Tensor, gather_mask: torch.Tensor, input_shape):
        self.gather_idxs = gather_idxs.long()
        self.gather_mask = gather_mask.bool()
        self.input_shape = tuple(int(s) for s in input_shape)

    @property
    def shape(self):
        return tuple(self.gather_idxs.shape)


def convert(gi: GatherInfo, arr: torch.Tensor, *, layout_axes: tuple[int, ...]) -> torch.Tensor:
    """atom_layout.convert: flatten `layout_axes` of arr, gather, mask."""
    axes = tuple(a if a >= 0 else a + arr.ndim for a in layout_axes)
    begin, end = axes[0], axes[-1] + 1
    layout_shape = arr.shape[begin:end]
    batch_shape, feat_shape = arr.shape[:begin], arr.shape[end:]
    flat = arr.reshape(batch_shape + (math.prod(layout_shape),) + feat_shape)
    idx = gi.gather_idxs
    if begin == 0:
        out = flat[idx]
    else:
        out = flat.index_select(begin, idx.reshape(-1)).reshape(batch_shape + idx.shape + feat_shape)
    mask_shape = (1,) * len(batch_shape) + gi.gather_mask.shape + (1,) * len(feat_shape)
    return out * gi.gather_mask.reshape(mask_shape).to(out.dtype)


# ── chunking ──────────────────────────────────────────────────────────────

def chunked(fn: Callable, size: int | None, *batched: torch.Tensor, dim: int = 0, out_dim: int | None = None):
    """mapping.inference_subbatch: apply fn to slices of the batched tensors along `dim`
    and concatenate along `out_dim`; size None runs fn once."""
    if size is None or batched[0].shape[dim] <= size:
        return fn(*batched)
    out_dim = dim if out_dim is None else out_dim
    n = batched[0].shape[dim]
    parts = [fn(*(b.narrow(dim, s, min(size, n - s)) for b in batched)) for s in range(0, n, size)]
    return torch.cat(parts, dim=out_dim)


def shard_size(num_residues: int, spec) -> int | None:
    """modules.get_shard_size."""
    size = spec[0][-1]
    for upper, s in spec:
        size = s
        if upper is None or num_residues <= upper:
            break
    return size
