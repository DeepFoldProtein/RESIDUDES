#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sub-module comparison between the engine's torch reference layers (with the
converted AlphaFold 3 weights) and the port. Needs a CUDA device."""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch

CORE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE / "thirdparty"))
sys.path.insert(0, str(CORE / "alphafold3"))

from tensorrt_bionemo.configs import DiffusionTransformerConfig, PairformerConfig  # noqa: E402
from tensorrt_bionemo.models.openfold3.convert import (convert_hf_diffusion_transformer_torch,  # noqa: E402
                                                      convert_hf_pairformer_torch)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer  # noqa: E402
from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule  # noqa: E402

from af3torch import diffusion_transformer as tdt  # noqa: E402
from af3torch import modules as tm  # noqa: E402
from af3torch import nn as tnn  # noqa: E402
from af3torch.params import Params  # noqa: E402
from trt.af3_to_openfold3 import pairformer_state_dict, token_transformer_state_dict  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)
rng = np.random.default_rng(0)
N = 24
params = Params(pathlib.Path("/data/databases/af3"), device=DEV)
root = params.root().child("diffuser")


def rep(name, ref, port, valid=None):
    ref, port = ref.detach().float(), port.detach().float()
    if valid is not None:                      # compare the unpadded tokens only
        ref, port = ref[tuple(slice(0, valid) for _ in range(valid_dims(ref)))], port[tuple(slice(0, valid) for _ in range(valid_dims(port)))]
    rel = (ref - port).norm() / max(ref.norm().item(), 1e-6)
    print(f"{name:38s} rel {rel:.2e}  max|Δ| {(ref - port).abs().max():.2e}  |ref| {ref.abs().max():.2e}  {'ok' if rel < 1e-3 else 'MISMATCH'}", flush=True)


def valid_dims(x):
    return 2 if x.ndim >= 3 and x.shape[0] == x.shape[1] else 1


def t(shape):
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(DEV)


# ── pairformer layer 0 ────────────────────────────────────────────────────
cfg = PairformerConfig(token_s=384, token_z=128, pairwise_head_width=32, pairwise_num_heads=4, num_blocks=1, num_heads=16,
                       dtype="float32", norm_epsilon=1e-5, support_batch=False, attention_initial_norm=True, version="v1",
                       trimul_high_precision=False, tri_attn_end_bias_before_transpose=True, backend="torch")
ref = PairformerModule(cfg)
ref.load_weights(convert_hf_pairformer_torch(cfg, weights=pairformer_state_dict(params, num_blocks=1)))
ref.eval().to(DEV)
L = ref.layers[0]
sc = root.child("evoformer").stack("__layer_stack_no_per_layer_1", 0).child("trunk_pairformer")
z, s = t((N, N, 128)), t((N, 384))
mask = torch.ones(N, device=DEV); mask[-3:] = 0
pm = mask[:, None] * mask[None, :]
with torch.no_grad():
    rep("tri_mul_out", L.tri_mul_out(z[None], mask=pm[None])[0], tm.triangle_multiplication(z, pm, sc.child("triangle_multiplication_outgoing"), outgoing=True))
    rep("tri_mul_in", L.tri_mul_in(z[None], mask=pm[None])[0], tm.triangle_multiplication(z, pm, sc.child("triangle_multiplication_incoming"), outgoing=False))
    rep("tri_attn_start (valid tokens)", L.tri_attn_start(z[None], mask=pm[None], attn_metadata=None)[0], tm.grid_self_attention(z, pm, sc.child("pair_attention1"), transpose=False), valid=N - 3)
    rep("tri_attn_end (valid tokens)", L.tri_attn_end(z[None], mask=pm[None], attn_metadata=None)[0], tm.grid_self_attention(z, pm, sc.child("pair_attention2"), transpose=True), valid=N - 3)
    ones = torch.ones_like(pm)
    rep("tri_attn_start (no padding)", L.tri_attn_start(z[None], mask=ones[None], attn_metadata=None)[0], tm.grid_self_attention(z, ones, sc.child("pair_attention1"), transpose=False))
    rep("tri_attn_end (no padding)", L.tri_attn_end(z[None], mask=ones[None], attn_metadata=None)[0], tm.grid_self_attention(z, ones, sc.child("pair_attention2"), transpose=True))
    rep("transition_z", L.transition_z(z[None])[0], tm.transition_block(z, sc.child("pair_transition")))
    rep("transition_s", L.transition_s(s[None])[0], tm.transition_block(s, sc.child("single_transition")))
    pl = tnn.linear(tnn.layer_norm(z, sc, "single_pair_logits_norm"), sc, "single_pair_logits_projection").permute(2, 0, 1)
    rep("single attention (pair bias)", L.attention(s[None], z[None], mask[None], attn_metadata=None)[0],
        tdt.self_attention(s, mask, pl, sc, prefix="single_attention_", num_head=16, single_cond=None))

# ── token transformer layer 0 ─────────────────────────────────────────────
tcfg = DiffusionTransformerConfig(num_blocks=1, num_heads=16, dim=768, dim_single_cond=384, dim_pairwise=128, expansion_factor=2,
                                  bias_proj=True, conditioned_transition_using_silu=True, attention_initial_norm=False,
                                  post_layer_norm=False, version="v1", dtype="float32", norm_epsilon=1e-5, support_batch=False,
                                  backend="torch")
tref = OpenFold3DiffusionTransformer(tcfg)
tref.load_weights(convert_hf_diffusion_transformer_torch(tcfg, weights=token_transformer_state_dict(params, num_blocks=1),
                                                         prefix="sample_diffusion.diffusion_module.diffusion_transformer.blocks"))
tref.eval().to(DEV)
T = tref.layers[0]
tr = root.child("~").child("diffusion_head").child("transformer")
inner = tr.stack("__layer_stack_with_per_layer", 0).stack("__layer_stack_with_per_layer", 0)
a, cond = t((N, 768)), t((N, 384))
with torch.no_grad():
    rep("adaln", T.adaln(a[None], cond[None])[0], tdt.adaptive_layernorm(a, cond, inner, "transformer"))
    rep("conditioned transition", T.transition(a[None], cond[None])[0], tdt.transition_block(a, 2, inner, "transformer", cond))
    # attention core: engine pair_bias_attn on the adaptively normed input, no output gate
    x = tdt.adaptive_layernorm(a, cond, inner, "transformer")
    q = tnn.linear(x, inner, "transformerq_projection", bias=True).float()
    k = tnn.linear(x, inner, "transformerk_projection").float()
    pl0 = tnn.linear(tnn.layer_norm(z, tr, "pair_input_layer_norm", offset=False), tr.stack("__layer_stack_with_per_layer", 0), "pair_logits_projection")
    pl0 = pl0.permute(2, 3, 0, 1)[0]                                   # (H, N, N) for block 0
    bias = (1e9 * (mask - 1.0))[None, None, :]
    logits = torch.einsum("qhc,khc->hqk", q * 48 ** -0.5, k) + bias + pl0
    w = torch.softmax(logits, dim=-1)
    v = tnn.linear(x, inner, "transformerv_projection")
    o = torch.einsum("hqk,khc->qhc", w, v).reshape(N, -1) * torch.sigmoid(tnn.linear(x, inner, "transformergating_query"))
    o = tnn.linear(o, inner, "transformertransition2")
    rep("pair-bias attention core", T.pair_bias_attn(s=x[None], z=z[None], mask=mask[None], attn_metadata=None)[0], o)
    rep("full block", T(a[None], cond[None], z[None], mask[None])[0],
        a + tdt.self_attention(a, mask, pl0, inner, prefix="transformer", num_head=16, single_cond=cond)
        + tdt.transition_block(a + tdt.self_attention(a, mask, pl0, inner, prefix="transformer", num_head=16, single_cond=cond), 2, inner, "transformer", cond))
