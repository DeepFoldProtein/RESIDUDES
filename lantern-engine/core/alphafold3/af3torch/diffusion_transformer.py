# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Diffusion transformer pieces (alphafold3.model.network.diffusion_transformer) in PyTorch.

Haiku built these with string prefixes rather than sub-scopes, so parameter
names are concatenations such as `transformerq_projection` or
`diffusion_atom_transformer_encoderffw_transition1`; `prefix` below is that
string and `s` the scope that owns it."""
from __future__ import annotations

import math

import torch

from . import nn
from .nn import Scope


def adaptive_layernorm(x: torch.Tensor, single_cond: torch.Tensor | None, s: Scope, prefix: str) -> torch.Tensor:
    if single_cond is None:
        return nn.layer_norm(x, s, f"{prefix}layer_norm")
    x = nn.layer_norm(x, s, f"{prefix}layer_norm", scale=False, offset=False)
    sc = nn.layer_norm(single_cond, s, f"{prefix}single_cond_layer_norm", offset=False)
    scale = nn.linear(sc, s, f"{prefix}single_cond_scale", bias=True)
    bias = nn.linear(sc, s, f"{prefix}single_cond_bias")
    return torch.sigmoid(scale) * x + bias


def adaptive_zero_init(x: torch.Tensor, single_cond: torch.Tensor | None, s: Scope, prefix: str) -> torch.Tensor:
    out = nn.linear(x, s, f"{prefix}transition2")
    if single_cond is None:
        return out
    cond = nn.linear(single_cond, s, f"{prefix}adaptive_zero_cond", bias=True)
    return torch.sigmoid(cond) * out


def transition_block(x: torch.Tensor, factor: int, s: Scope, prefix: str, single_cond: torch.Tensor | None = None) -> torch.Tensor:
    c = x.shape[-1]
    inter = factor * c
    x = adaptive_layernorm(x, single_cond, s, f"{prefix}ffw_")
    w = s.get(f"{prefix}ffw_transition1", "weights", dtype=x.dtype).reshape(c, 2, inter)
    h = nn.glu(x, w, nn.swish)
    return adaptive_zero_init(h, single_cond, s, f"{prefix}ffw_")


def self_attention(x: torch.Tensor, mask: torch.Tensor, pair_logits: torch.Tensor | None, s: Scope, *, prefix: str,
                   num_head: int, single_cond: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-head self-attention with an additive pair bias; logits in float32.
    x (..., N, C); mask (..., N); pair_logits (..., H, N, N)."""
    x = adaptive_layernorm(x, single_cond, s, prefix)
    bias = (1e9 * (mask.to(x.dtype) - 1.0))[..., None, None, :]                     # (..., 1, 1, N)
    if pair_logits is not None:
        bias = bias + pair_logits.to(x.dtype)                                       # (..., H, N, N)
    q = nn.linear(x, s, f"{prefix}q_projection", bias=True)
    k = nn.linear(x, s, f"{prefix}k_projection")
    v = nn.linear(x, s, f"{prefix}v_projection")
    o = nn.dot_product_attention(q, k, v, bias=bias)                                # softmax in float32 inside the kernel
    o = o.reshape(o.shape[:-2] + (-1,))
    o = o * torch.sigmoid(nn.linear(x, s, f"{prefix}gating_query"))
    return adaptive_zero_init(o, single_cond, s, prefix)


def transformer(act: torch.Tensor, mask: torch.Tensor, single_cond: torch.Tensor, pair_cond: torch.Tensor | None,
                s: Scope, *, num_blocks: int = 24, super_block_size: int = 4, factor: int = 2, num_head: int = 16) -> torch.Tensor:
    """Transformer: num_blocks blocks grouped in super blocks that share one pair-logit projection."""
    prefix = "transformer"
    pair_act = None if pair_cond is None else nn.layer_norm(pair_cond, s, "pair_input_layer_norm", offset=False)
    n_super = num_blocks // super_block_size
    for sb in range(n_super):
        outer = s.stack("__layer_stack_with_per_layer", sb)
        if pair_act is None:
            pair_logits = None
        else:
            pl = nn.linear(pair_act, outer, "pair_logits_projection")               # (N, N, sbs, H)
            pair_logits = pl.permute(2, 3, 0, 1)                                     # (sbs, H, N, N)
        for b in range(super_block_size):
            inner = outer.stack("__layer_stack_with_per_layer", b)
            pl_b = None if pair_logits is None else pair_logits[b]
            act = act + self_attention(act, mask, pl_b, inner, prefix=prefix, num_head=num_head, single_cond=single_cond)
            act = act + transition_block(act, factor, inner, prefix, single_cond)
    return act


def cross_attention(x_q: torch.Tensor, x_k: torch.Tensor, mask_q: torch.Tensor, mask_k: torch.Tensor, s: Scope, *,
                    prefix: str, num_head: int, key_dim: int, value_dim: int, pair_logits: torch.Tensor | None,
                    single_cond_q: torch.Tensor | None, single_cond_k: torch.Tensor | None) -> torch.Tensor:
    """cross_attention over atom subsets: x_q (..., Q, C), x_k (..., K, C)."""
    x_q = adaptive_layernorm(x_q, single_cond_q, s, f"{prefix}q")
    x_k = adaptive_layernorm(x_k, single_cond_k, s, f"{prefix}k")
    bias = 1e9 * (mask_q.to(x_q.dtype) - 1.0)[..., None, :, None] * (mask_k.to(x_q.dtype) - 1.0)[..., None, None, :]
    if pair_logits is not None:
        bias = bias + pair_logits.to(x_q.dtype)
    q = nn.linear(x_q, s, f"{prefix}q_projection", bias=True)
    k = nn.linear(x_k, s, f"{prefix}k_projection")
    v = nn.linear(x_k, s, f"{prefix}v_projection")
    o = nn.dot_product_attention(q, k, v, bias=bias)                                # softmax in float32 inside the kernel
    o = o.reshape(o.shape[:-2] + (-1,))
    o = o * torch.sigmoid(nn.linear(x_q, s, f"{prefix}gating_query"))
    return adaptive_zero_init(o, single_cond_q, s, prefix)


def cross_att_transformer(queries_act: torch.Tensor, queries_mask: torch.Tensor, queries_to_keys: nn.GatherInfo,
                          keys_mask: torch.Tensor, queries_single_cond: torch.Tensor, keys_single_cond: torch.Tensor,
                          pair_cond: torch.Tensor, s: Scope, *, prefix: str, num_blocks: int = 3, factor: int = 2,
                          num_head: int = 4, key_dim: int = 128, value_dim: int = 128) -> torch.Tensor:
    """CrossAttTransformer: queries (…, S, Q, C) attend to keys gathered from the queries."""
    pair_act = nn.layer_norm(pair_cond, s, "pair_input_layer_norm", offset=False)
    pl = nn.linear(pair_act, s, "pair_logits_projection")                            # (…, S, Q, K, blocks, H)
    nb = pl.ndim
    pair_logits = pl.permute(nb - 2, *range(nb - 5), nb - 5, nb - 1, nb - 4, nb - 3)  # (blocks, …, S, H, Q, K)
    for b in range(num_blocks):
        inner = s.stack("__layer_stack_with_per_layer", b)
        keys_act = nn.convert(queries_to_keys, queries_act, layout_axes=(-3, -2))
        queries_act = queries_act + cross_attention(
            queries_act, keys_act, queries_mask, keys_mask, inner, prefix=prefix, num_head=num_head, key_dim=key_dim,
            value_dim=value_dim, pair_logits=pair_logits[b], single_cond_q=queries_single_cond, single_cond_k=keys_single_cond)
        queries_act = queries_act + transition_block(queries_act, factor, inner, prefix, queries_single_cond)
    return queries_act
