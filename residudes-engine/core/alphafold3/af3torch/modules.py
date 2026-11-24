# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Pair / MSA stack blocks of AlphaFold 3 (alphafold3.model.network.modules) in PyTorch.

Every function takes the activations and the `Scope` whose Haiku name it owns, so
`PairFormerIteration(scope.stack('__layer_stack_no_per_layer_1', i).child('trunk_pairformer'), ...)`
reads layer i of the trunk straight from the published arrays."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import nn
from .nn import Scope
from .diffusion_transformer import self_attention

PAIR_ATTENTION_CHUNK = ((1536, 128), (None, 32))        # GlobalConfig.pair_attention_chunk_size
PAIR_TRANSITION_SHARD = ((2048, None), (None, 1024))     # GlobalConfig.pair_transition_shard_spec


def transition_block(act: torch.Tensor, s: Scope, *, factor: int = 4) -> torch.Tensor:
    """TransitionBlock: LN → GLU(swish) → Linear."""
    c = act.shape[-1]
    inter = int(c * factor)
    act = nn.layer_norm(act, s, "input_layer_norm")
    w = s.get("transition1", "weights", dtype=act.dtype).reshape(c, 2, inter)
    h = nn.glu(act, w, nn.swish)
    return nn.linear(h, s, "transition2")


def msa_attention(act: torch.Tensor, mask: torch.Tensor, pair_act: torch.Tensor, s: Scope, *, num_head: int = 8) -> torch.Tensor:
    """MSAAttention: attention weights come from the pair representation, values from the MSA."""
    act = nn.layer_norm(act, s, "act_norm")
    pair_act = nn.layer_norm(pair_act, s, "pair_norm")
    logits = nn.linear(pair_act, s, "pair_logits").permute(2, 0, 1)               # (H, N, N)
    logits = logits + 1e9 * (mask.max(dim=0).values.to(logits.dtype) - 1.0)
    weights = torch.softmax(logits.float(), dim=-1).to(act.dtype)
    c = act.shape[-1]
    v = nn.linear(act, s, "v_projection")                                          # (S, N, H, c/H)
    v_avg = torch.einsum("hqk,bkhc->bqhc", weights, v)
    v_avg = v_avg.reshape(v_avg.shape[:-2] + (-1,))
    gate = nn.linear(act, s, "gating_query")
    v_avg = v_avg * torch.sigmoid(gate)
    return nn.linear(v_avg, s, "output_projection")


def grid_self_attention(act: torch.Tensor, pair_mask: torch.Tensor, s: Scope, *, transpose: bool, num_head: int = 4) -> torch.Tensor:
    """GridSelfAttention over the rows (transpose=False) or columns (True) of the pair grid."""
    pair_mask = pair_mask.transpose(-1, -2)
    act = nn.layer_norm(act, s, "act_norm")
    bias = nn.linear(act, s, "pair_bias_projection").permute(2, 0, 1).unsqueeze(0)   # (1, H, N, N), a permuted view
    n = act.shape[0]
    if transpose:
        act = act.transpose(-2, -3)
    # pair_mask is an outer product of the token mask, so every valid row shares one key mask: that masked bias is
    # built once, contiguous, and broadcast over the row chunks inside the fused attention kernel. Chunks that hold
    # fully masked (padded) rows take the reference's row-dependent mask instead, which lets those rows attend
    # uniformly, so the output matches the JAX module everywhere.
    neg = torch.full((), -0.7 * torch.finfo(act.dtype).max, dtype=act.dtype, device=act.device)
    bias = bias.to(act.dtype).contiguous()
    key_mask = (pair_mask.sum(dim=0) > 0)[None, None, None, :]
    shared_bias = torch.where(key_mask, bias, neg)
    row_valid = pair_mask.sum(dim=1) > 0                                             # (N,) rows with any key

    def attention(a, rv):
        if bool(rv.all()):
            b = shared_bias
        else:
            b = torch.where(rv[:, None, None, None] & key_mask, bias, neg)         # (c, H, N, N)
        q = nn.linear(a, s, "q_projection", transpose=True)
        k = nn.linear(a, s, "k_projection", transpose=True)
        v = nn.linear(a, s, "v_projection")
        o = nn.dot_product_attention(q, k, v, bias=b)
        o = o.reshape(o.shape[:-2] + (-1,))
        o = o * torch.sigmoid(nn.linear(a, s, "gating_query", transpose=True))
        return nn.linear(o, s, "output_projection")

    act = nn.chunked(attention, nn.shard_size(n, PAIR_ATTENTION_CHUNK), act, row_valid)
    if transpose:
        act = act.transpose(-2, -3)
    return act


def triangle_multiplication(act: torch.Tensor, mask: torch.Tensor, s: Scope, *, outgoing: bool) -> torch.Tensor:
    """TriangleMultiplication (outgoing: ikc,jkc->ijc; incoming: kjc,kic->ijc)."""
    act = nn.layer_norm(act, s, "left_norm_input")
    input_act = act
    c = act.shape[-1]
    # AF3 interleaves the two halves by channel (2c + {a, b}); the weight columns are reordered into [a | b] so each
    # half is a contiguous slab of the projection and the transposition into (C, N, N) below is a plain copy.
    w_p, w_g = (s.get(n, "weights", dtype=act.dtype) for n in ("projection", "gate"))
    w_p = torch.cat([w_p[:, 0::2], w_p[:, 1::2]], dim=1)
    w_g = torch.cat([w_g[:, 0::2], w_g[:, 1::2]], dim=1)
    proj = (act @ w_p) * torch.sigmoid(act @ w_g) * mask[..., None]                # (N, N, 2C) = [a | b]
    a, b = proj[..., :c], proj[..., c:]
    eq = "ikc,jkc->cij" if outgoing else "kjc,kic->cij"
    act = torch.einsum(eq, a, b).permute(1, 2, 0).contiguous()                      # (N, N, C)
    act = nn.layer_norm(act, s, "center_norm")
    act = nn.linear(act, s, "output_projection")
    gate_out = nn.linear(input_act, s, "gating_linear")
    return act * torch.sigmoid(gate_out)


def outer_product_mean(act: torch.Tensor, mask: torch.Tensor, s: Scope, *, chunk: int = 128) -> torch.Tensor:
    """OuterProductMean: (S, N, C) MSA activations → (N, N, pair) update."""
    mask = mask[..., None]
    act = nn.layer_norm(act, s, "layer_norm_input")
    left = mask * nn.linear(act, s, "left_projection")
    right = mask * nn.linear(act, s, "right_projection")
    w = s.get("", "output_w", dtype=act.dtype)                                       # (c, e, F), directly in this scope
    b = s.get("", "output_b", dtype=act.dtype)

    def compute_chunk(left_chunk):
        l = left_chunk.permute(0, 2, 1)                                              # (S, c, n)
        o = torch.einsum("acb,ade->dceb", l, right)                                  # (N, c, e, n)
        o = torch.einsum("dceb,cef->dbf", o, w) + b                                  # (N, n, F)
        return o.permute(1, 0, 2)                                                    # (n, N, F)

    out = nn.chunked(compute_chunk, chunk, left, dim=1, out_dim=0)
    norm = torch.einsum("abc,adc->bdc", mask.to(out.dtype), mask.to(out.dtype))
    return out / (1e-3 + norm)


def pairformer_iteration(act: torch.Tensor, pair_mask: torch.Tensor, s: Scope, *, single_act: torch.Tensor | None = None,
                         seq_mask: torch.Tensor | None = None, pair_transition_factor: int = 4,
                         single_num_head: int = 16):
    """PairFormerIteration (with_single when single_act is given)."""
    n = act.shape[0]
    act = act + triangle_multiplication(act, pair_mask, s.child("triangle_multiplication_outgoing"), outgoing=True)
    act = act + triangle_multiplication(act, pair_mask, s.child("triangle_multiplication_incoming"), outgoing=False)
    act = act + grid_self_attention(act, pair_mask, s.child("pair_attention1"), transpose=False)
    act = act + grid_self_attention(act, pair_mask, s.child("pair_attention2"), transpose=True)
    ts = s.child("pair_transition")
    act = act + nn.chunked(lambda a: transition_block(a, ts, factor=pair_transition_factor),
                           nn.shard_size(n, PAIR_TRANSITION_SHARD), act)
    if single_act is None:
        return act
    pair_logits = nn.linear(nn.layer_norm(act, s, "single_pair_logits_norm"), s, "single_pair_logits_projection")
    pair_logits = pair_logits.permute(2, 0, 1)
    single_act = single_act + self_attention(single_act, seq_mask, pair_logits, s, prefix="single_attention_",
                                             num_head=single_num_head, single_cond=None)
    single_act = single_act + transition_block(single_act, s.child("single_transition"), factor=4)
    return act, single_act


def evoformer_iteration(msa_act: torch.Tensor, pair_act: torch.Tensor, msa_mask: torch.Tensor, pair_mask: torch.Tensor,
                        s: Scope) -> tuple[torch.Tensor, torch.Tensor]:
    """EvoformerIteration (the MSA stack block)."""
    n = pair_act.shape[0]
    pair_act = pair_act + outer_product_mean(msa_act, msa_mask, s.child("outer_product_mean"))
    msa_act = msa_act + msa_attention(msa_act, msa_mask, pair_act, s.child("msa_attention1"))
    msa_act = msa_act + transition_block(msa_act, s.child("msa_transition"))
    pair_act = pair_act + triangle_multiplication(pair_act, pair_mask, s.child("triangle_multiplication_outgoing"), outgoing=True)
    pair_act = pair_act + triangle_multiplication(pair_act, pair_mask, s.child("triangle_multiplication_incoming"), outgoing=False)
    pair_act = pair_act + grid_self_attention(pair_act, pair_mask, s.child("pair_attention1"), transpose=False)
    pair_act = pair_act + grid_self_attention(pair_act, pair_mask, s.child("pair_attention2"), transpose=True)
    ts = s.child("pair_transition")
    pair_act = pair_act + nn.chunked(lambda a: transition_block(a, ts), nn.shard_size(n, PAIR_TRANSITION_SHARD), pair_act)
    return msa_act, pair_act
