#!/usr/bin/env python3
"""What does the compiled JAX pairformer actually contain?

Builds the Boltz-2 / OF3-width PairformerStack (lantern_engine_jax.haiku) with random
bf16 parameters, lowers it for the given N and input dtype, and reports:
  * how many cuDNN fused-attention custom calls the optimised HLO contains
    (0 = triangle attention fell back to XLA's materialised softmax),
  * XLA's own peak temp buffer size (compiled.memory_analysis),
  * measured device peak + runtime for one call.

    lantern-engine run --jax --model bt2 -- bench/micro/pairformer_hlo.py --n 604 --in-dtype fp32
"""
import argparse, os, sys, time
sys.path.insert(0, os.environ.get("LANTERN_ENGINE_CORE", "core"))
import jax, jax.numpy as jnp, haiku as hk
from lantern_engine_jax.haiku.pairformer import PairformerStack, PairformerVariant

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=604)
ap.add_argument("--blocks", type=int, default=1)
ap.add_argument("--in-dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--variant", default="boltz2", choices=["boltz2", "openfold3", "protenix_v2"])
ns = ap.parse_args()

dev = jax.devices()[0]
N, dt = ns.n, (jnp.float32 if ns.in_dtype == "fp32" else jnp.bfloat16)
variant = getattr(PairformerVariant, ns.variant)(num_blocks=ns.blocks)

def fn(z, pm, s, sm):
    return PairformerStack(variant=variant, with_single=True)(z, pm, s, sm)

tr = hk.transform(fn)
key = jax.random.PRNGKey(0)
z = jax.random.normal(key, (N, N, variant.token_z), dt)
s = jax.random.normal(key, (N, variant.token_s), dt)
pm = jnp.ones((N, N), jnp.float32)
sm = jnp.ones((N,), jnp.float32)
params = tr.init(key, z, pm, s, sm)
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)

lowered = jax.jit(tr.apply).lower(params, None, z, pm, s, sm)
compiled = lowered.compile()
txt = compiled.as_text()
n_fmha = txt.count("__cudnn$fmha") + txt.count("cudnn$fmha") + txt.count("__cudnn$fused")
n_custom = txt.count("custom_call_target")
ma = compiled.memory_analysis()
temp_gb = getattr(ma, "temp_size_in_bytes", 0) / 1e9
print(f"variant={ns.variant} blocks={ns.blocks} N={N} in_dtype={ns.in_dtype} "
      f"env: FOLD_MASK={os.environ.get('BIO_TRT_JAX_FOLD_MASK','1')} SCAN={os.environ.get('BIO_TRT_JAX_PAIRFORMER_SCAN','?')} "
      f"TOKAMAX={os.environ.get('BIO_TRT_JAX_TOKAMAX','?')} CUEQ={os.environ.get('BIO_TRT_JAX_CUEQ','?')}", flush=True)
print(f"  cuDNN fused-attention custom calls in HLO: {n_fmha}   (all custom calls: {n_custom})", flush=True)
print(f"  XLA temp buffers (program peak scratch): {temp_gb:.2f} GB", flush=True)
# largest fp32 arrays mentioned in the HLO, as a hint of what is materialised
import re
shapes = re.findall(r"(f32|bf16)\[([0-9,]+)\]", txt)
big = sorted({(d, s) for d, s in shapes}, key=lambda ds: -eval("*".join(ds[1].split(","))) if ds[1] else 0)[:4]
for d, shp in big:
    nbytes = eval("*".join(shp.split(","))) * (4 if d == "f32" else 2)
    print(f"  largest tensors: {d}[{shp}] = {nbytes/1e9:.2f} GB", flush=True)
    # which instructions produce this shape? (first 3, truncated)
    pat = re.compile(r"^\s*(%?[\w.\-]+) = " + re.escape(f"{d}[{shp}]") + r"[^ ]* (\w+)\(", re.M)
    seen = 0
    for mm in pat.finditer(txt):
        line = txt[mm.start():txt.find("\n", mm.start())]
        print("      ", line.strip()[:170], flush=True)
        seen += 1
        if seen >= 3: break
# every custom call and every attention-looking dot, with the module path that produced it
for mm in re.finditer(r"custom_call_target=\"([^\"]+)\".*?op_name=\"([^\"]*)\"", txt):
    print(f"  custom-call {mm.group(1)[:40]:<40s} @ {mm.group(2)[-90:]}", flush=True)
for mm in re.finditer(r"= (f32|bf16)\[([0-9,]+)\][^\n]*? dot\([^\n]*?op_name=\"([^\"]*attention[^\"]*)\"", txt):
    print(f"  attention dot {mm.group(1)}[{mm.group(2)}] @ {mm.group(3)[-90:]}", flush=True)
t0 = time.perf_counter(); out = compiled(params, None, z, pm, s, sm); jax.block_until_ready(out); t1 = time.perf_counter()
out = compiled(params, None, z, pm, s, sm); jax.block_until_ready(out); t2 = time.perf_counter()
peak = dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
print(f"  run: first {t1-t0:.2f} s, second {t2-t1:.3f} s per {ns.blocks} block(s); device peak {peak:.2f} GB", flush=True)
