#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Module-by-module check of the PyTorch port against the Haiku modules.

Both sides get the published parameters of one layer and the same random
float32 inputs; everything runs on the CPU. Reports the largest absolute
difference and the relative error of each module's output.

    JAX_PLATFORMS=cpu core/jaxenv/.venv/bin/python core/alphafold3/tests/compare_jax.py [--n 48] [--model-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models" / "alphafold3"))

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from alphafold3.model import model as jmodel  # noqa: E402
from alphafold3.model import model_config  # noqa: E402
from alphafold3.model.atom_layout import atom_layout  # noqa: E402
from alphafold3.model.network import diffusion_transformer as jdt  # noqa: E402
from alphafold3.model.network import modules as jm  # noqa: E402

from af3torch import diffusion_transformer as tdt  # noqa: E402
from af3torch import modules as tm  # noqa: E402
from af3torch import nn as tnn  # noqa: E402
from af3torch.params import Params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="/data/databases/af3")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--only", default=None, help="comma-separated test names")
args = ap.parse_args()

torch.set_default_dtype(torch.float32)
rng = np.random.default_rng(0)
params = Params(pathlib.Path(args.model_dir), device="cpu")
root = params.root().child("diffuser")
GC = model_config.GlobalConfig()
GC.flash_attention_implementation = "xla"
CFG = jmodel.Model.Config()
N = args.n


def hk_params(prefix: str, idx: tuple[int, ...] = ()) -> dict:
    """Published params under `prefix` re-rooted for a transform, sliced for a stacked layer."""
    out: dict = {}
    for (scope, name) in params.keys():
        if scope == prefix or scope.startswith(prefix + "/"):
            key = scope[len(prefix) + 1:] if scope != prefix else ""
            arr = np.asarray(params.get(scope, name).float().numpy())
            for i in idx:
                arr = arr[i]
            out.setdefault(key, {})[name] = jnp.asarray(arr)
    assert out, prefix
    return out


def report(name: str, j, t, t0: float) -> None:
    j = np.asarray(j, dtype=np.float32)
    t = np.asarray(t.detach().float().cpu().numpy(), dtype=np.float32)
    assert j.shape == t.shape, (name, j.shape, t.shape)
    diff = np.abs(j - t)
    rel = np.linalg.norm(j - t) / max(np.linalg.norm(j), 1e-6)
    flag = "ok" if rel < 1e-3 else ("CHECK" if rel < 1e-2 else "MISMATCH")
    print(f"{name:40s} max|Δ| {diff.max():.3e}  rel {rel:.2e}  |jax| {np.abs(j).max():.2e}  {flag}  ({time.time() - t0:.1f}s)", flush=True)


def rand(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def tt(x):
    return torch.from_numpy(np.asarray(x))


TRUNK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer"
MSA = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"
CONF = "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer"
TPL = "diffuser/evoformer/template_embedding/single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration"
DIFF = "diffuser/~/diffusion_head"

trunk_t = root.child("evoformer").stack("__layer_stack_no_per_layer_1", 0).child("trunk_pairformer")
msa_t = root.child("evoformer").stack("__layer_stack_no_per_layer", 0).child("msa_stack")
conf_t = root.child("confidence_head").stack("__layer_stack_no_per_layer", 0).child("confidence_pairformer")
tpl_t = root.child("evoformer").child("template_embedding").child("single_template_embedding").stack("__layer_stack_no_per_layer", 0).child("template_embedding_iteration")
diff_t = root.child("~").child("diffusion_head")

pair = rand(N, N, 128)
pair_mask = np.ones((N, N), np.float32)
pair_mask[-3:, :] = 0
pair_mask[:, -3:] = 0
seq_mask = np.ones((N,), np.float32)
seq_mask[-3:] = 0
single = rand(N, 384)
tests = {}


def test(name):
    def deco(fn):
        tests[name] = fn
        return fn
    return deco


@test("pair_transition")
def _():
    t0 = time.time()
    f = hk.transform(lambda a: jm.TransitionBlock(CFG.evoformer.pairformer.pair_transition, GC, name="pair_transition")(a))
    j = f.apply(hk_params(TRUNK + "/pair_transition", (0,)) and _reroot(TRUNK, (0,), "pair_transition"), None, pair)
    t = tm.transition_block(tt(pair), trunk_t.child("pair_transition"))
    report("pair_transition", j, t, t0)


def _reroot(prefix: str, idx, module: str) -> dict:
    """Params of `prefix/module/...` keyed as `module/...` (what a transform of that module expects)."""
    return hk_params(prefix + "/" + module, idx) and {(module + "/" + k if k else module): v for k, v in hk_params(prefix + "/" + module, idx).items()}


@test("triangle_multiplication")
def _():
    for kind, eq in (("outgoing", "ikc,jkc->ijc"), ("incoming", "kjc,kic->ijc")):
        t0 = time.time()
        name = f"triangle_multiplication_{kind}"
        cfg = jm.TriangleMultiplication.Config(equation=eq)
        f = hk.transform(lambda a, m: jm.TriangleMultiplication(cfg, GC, name=name)(a, m))
        j = f.apply(_reroot(TRUNK, (0,), name), None, pair, pair_mask)
        t = tm.triangle_multiplication(tt(pair), tt(pair_mask), trunk_t.child(name), outgoing=(kind == "outgoing"))
        report(name, j, t, t0)


@test("grid_attention")
def _():
    for name, transpose in (("pair_attention1", False), ("pair_attention2", True)):
        t0 = time.time()
        f = hk.transform(lambda a, m: jm.GridSelfAttention(jm.GridSelfAttention.Config(), GC, transpose=transpose, name=name)(a, m))
        j = f.apply(_reroot(TRUNK, (0,), name), None, pair, pair_mask)
        t = tm.grid_self_attention(tt(pair), tt(pair_mask), trunk_t.child(name), transpose=transpose)
        report(name, j, t, t0)


@test("outer_product_mean")
def _():
    t0 = time.time()
    S = 40
    msa = rand(S, N, 64)
    msa_mask = np.ones((S, N), np.float32)
    msa_mask[-5:, :] = 0
    msa_mask[:, -3:] = 0
    f = hk.transform(lambda a, m: jm.OuterProductMean(jm.OuterProductMean.Config(), GC, 128, name="outer_product_mean")(a, m))
    j = f.apply(_reroot(MSA, (0,), "outer_product_mean"), None, msa, msa_mask)
    t = tm.outer_product_mean(tt(msa), tt(msa_mask), msa_t.child("outer_product_mean"))
    report("outer_product_mean", j, t, t0)
    t0 = time.time()
    f = hk.transform(lambda a, m, p: jm.MSAAttention(jm.MSAAttention.Config(), GC, name="msa_attention1")(a, m, p))
    j = f.apply(_reroot(MSA, (0,), "msa_attention1"), None, msa, msa_mask, pair)
    t = tm.msa_attention(tt(msa), tt(msa_mask), tt(pair), msa_t.child("msa_attention1"))
    report("msa_attention1", j, t, t0)
    t0 = time.time()
    f = hk.transform(lambda x: jm.EvoformerIteration(CFG.evoformer.msa_stack, GC, name="msa_stack")(
        activations={"msa": x[0], "pair": x[1]}, masks={"msa": msa_mask, "pair": pair_mask}))
    j = f.apply(_reroot("diffuser/evoformer/__layer_stack_no_per_layer", (0,), "msa_stack"), None, (msa, pair))
    tm_, tp_ = tm.evoformer_iteration(tt(msa), tt(pair), tt(msa_mask), tt(pair_mask), msa_t)
    report("evoformer_iteration.msa", j["msa"], tm_, t0)
    report("evoformer_iteration.pair", j["pair"], tp_, t0)


@test("pairformer")
def _():
    t0 = time.time()
    f = hk.transform(lambda a, s: jm.PairFormerIteration(CFG.evoformer.pairformer, GC, with_single=True, name="trunk_pairformer")(
        act=a, single_act=s, pair_mask=pair_mask, seq_mask=seq_mask))
    j = f.apply(_reroot("diffuser/evoformer/__layer_stack_no_per_layer_1", (0,), "trunk_pairformer"), None, pair, single)
    tp, ts = tm.pairformer_iteration(tt(pair), tt(pair_mask), trunk_t, single_act=tt(single), seq_mask=tt(seq_mask))
    report("trunk_pairformer.pair", j[0], tp, t0)
    report("trunk_pairformer.single", j[1], ts, t0)
    t0 = time.time()
    f = hk.transform(lambda a, s: jm.PairFormerIteration(CFG.heads.confidence.pairformer, GC, with_single=True, name="confidence_pairformer")(
        act=a, single_act=s, pair_mask=pair_mask, seq_mask=seq_mask))
    j = f.apply(_reroot("diffuser/confidence_head/__layer_stack_no_per_layer", (0,), "confidence_pairformer"), None, pair, single)
    tp, ts = tm.pairformer_iteration(tt(pair), tt(pair_mask), conf_t, single_act=tt(single), seq_mask=tt(seq_mask))
    report("confidence_pairformer.pair", j[0], tp, t0)
    report("confidence_pairformer.single", j[1], ts, t0)
    t0 = time.time()
    pair64 = rand(N, N, 64)
    f = hk.transform(lambda a: jm.PairFormerIteration(CFG.evoformer.template.template_stack, GC, name="template_embedding_iteration")(act=a, pair_mask=pair_mask))
    j = f.apply(_reroot(TPL.rsplit("/", 1)[0], (0,), "template_embedding_iteration"), None, pair64)
    t = tm.pairformer_iteration(tt(pair64), tt(pair_mask), tpl_t, pair_transition_factor=2)
    report("template_embedding_iteration", j, t, t0)


@test("diffusion_transformer")
def _():
    t0 = time.time()
    act = rand(N, 768)
    cond = rand(N, 384)
    f = hk.transform(lambda a, c, p: jdt.Transformer(CFG.heads.diffusion.transformer, GC, name="transformer")(act=a, mask=seq_mask, single_cond=c, pair_cond=p))
    j = f.apply(_reroot(DIFF, (), "transformer"), None, act, cond, pair)
    t = tdt.transformer(tt(act), tt(seq_mask), tt(cond), tt(pair), diff_t.child("transformer"))
    report("diffusion transformer (24 blocks)", j, t, t0)


@test("cross_att_transformer")
def _():
    t0 = time.time()
    S, Q, K, C = 3, 8, 12, 128
    q_act, q_cond, k_cond = rand(S, Q, C), rand(S, Q, C), rand(S, K, C)
    q_mask = np.ones((S, Q), np.float32)
    q_mask[-1, -2:] = 0
    k_mask = np.ones((S, K), np.float32)
    k_mask[:, -1] = 0
    idx = rng.integers(0, S * Q, size=(S, K))
    gmask = np.ones((S, K), bool)
    gmask[0, -1] = False
    gi_j = atom_layout.GatherInfo(gather_idxs=idx, gather_mask=gmask, input_shape=np.array([S, Q]))
    gi_t = tnn.GatherInfo(torch.from_numpy(idx), torch.from_numpy(gmask), (S, Q))
    pair_cond = rand(S, Q, K, 16)
    name = "diffusion_atom_transformer_encoder"
    f = hk.transform(lambda a, pc: jdt.CrossAttTransformer(CFG.heads.diffusion.atom_transformer, GC, name=name)(
        queries_act=a, queries_mask=q_mask, queries_to_keys=gi_j, keys_mask=k_mask, queries_single_cond=q_cond,
        keys_single_cond=k_cond, pair_cond=pc))
    j = f.apply(_reroot(DIFF, (), name), None, q_act, pair_cond)
    t = tdt.cross_att_transformer(tt(q_act), tt(q_mask), gi_t, tt(k_mask), tt(q_cond), tt(k_cond), tt(pair_cond),
                                  diff_t.child(name), prefix=name)
    report("atom cross-attention transformer", j, t, t0)


names = args.only.split(",") if args.only else list(tests)
for nm in names:
    try:
        tests[nm]()
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"{nm}: FAILED {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
