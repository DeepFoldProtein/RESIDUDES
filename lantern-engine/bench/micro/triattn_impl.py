#!/usr/bin/env python3
"""Which attention implementation does the JAX triangle attention actually get?

Runs the exact shapes lantern_engine_jax uses for triangle attention
(q/k/v [N, N, H, D] bf16, bias [1, H, N, N], mask [N, 1, 1, N] bool) through
jax.nn.dot_product_attention with implementation = cudnn / xla, and through
cuequivariance_jax.triangle_attention when importable, and reports time and
peak device memory per N. Prints the exception text when an implementation is
rejected, which is what decides the fallback inside lantern_engine_jax.haiku._tokamax.

    lantern-engine run --jax -- bench/micro/triattn_impl.py [--sizes 202,604,1002] [--heads 4] [--dim 32]
"""
import argparse, time
import jax, jax.numpy as jnp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="202,604,1002")
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--dim", type=int, default=32)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--variants", default="xla,cudnn,cudnn-bias,cudnn-folded,cuex", help="comma list; run one per process for a true peak-memory reading")
ns = ap.parse_args()

dev = jax.devices()[0]
print("jax", jax.__version__, "device", dev.device_kind, flush=True)
try:
    import cuequivariance_jax as cuex  # noqa
    print("cuequivariance_jax", getattr(cuex, "__version__", "?"), flush=True)
except Exception as e:
    cuex = None
    print("cuequivariance_jax: not importable:", type(e).__name__, str(e)[:100], flush=True)


def peak_gb():
    try:
        return dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        return float("nan")


def run(name, fn, *args):
    try:
        f = jax.jit(fn)
        t0 = time.perf_counter(); out = f(*args); out.block_until_ready(); t_compile = time.perf_counter() - t0
        ts = []
        for _ in range(ns.reps):
            t0 = time.perf_counter(); out = f(*args); out.block_until_ready(); ts.append(time.perf_counter() - t0)
        print(f"  {name:<28s} ok   run {min(ts)*1e3:8.1f} ms  (first call incl. compile {t_compile:6.2f} s)  peak {peak_gb():6.2f} GB", flush=True)
        return np.asarray(out.astype(jnp.float32))
    except Exception as e:
        msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
        print(f"  {name:<28s} FAIL {type(e).__name__}: {msg}", flush=True)
        return None


for N in [int(s) for s in ns.sizes.split(",")]:
    H, D = ns.heads, ns.dim
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (N, N, H, D), jnp.bfloat16)
    k = jax.random.normal(k2, (N, N, H, D), jnp.bfloat16)
    v = jax.random.normal(k3, (N, N, H, D), jnp.bfloat16)
    bias = jax.random.normal(k4, (1, H, N, N), jnp.bfloat16)
    mask = jnp.ones((N, 1, 1, N), jnp.bool_)
    print(f"N={N}  q/k/v [{N},{N},{H},{D}] bf16  bias [1,{H},{N},{N}]  mask [{N},1,1,{N}]", flush=True)
    V = set(ns.variants.split(","))
    ref = None
    if "xla" in V:
        ref = run("xla (jax.nn default)", lambda q, k, v, b, m: jax.nn.dot_product_attention(q, k, v, bias=b, mask=m), q, k, v, bias, mask)
    out = None
    if "cudnn" in V:
        out = run("cudnn (bias+mask)", lambda q, k, v, b, m: jax.nn.dot_product_attention(q, k, v, bias=b, mask=m, implementation="cudnn"), q, k, v, bias, mask)
    if out is not None and ref is not None:
        print(f"    max|cudnn-xla| = {np.abs(out-ref).max():.4f}", flush=True)
    if "cudnn-bias" in V:
        out = run("cudnn (bias only)", lambda q, k, v, b: jax.nn.dot_product_attention(q, k, v, bias=b, implementation="cudnn"), q, k, v, bias)
    # bias folded with mask into one additive fp32/bf16 bias (what a pad-to-64 shim would do)
    big = jnp.where(mask, 0.0, -1e9).astype(jnp.bfloat16)  # [N,1,1,N]
    if "cudnn-folded" in V:
        out = run("cudnn (bias+mask folded)", lambda q, k, v, b, m: jax.nn.dot_product_attention(q, k, v, bias=b + m, implementation="cudnn"), q, k, v, bias, big)
    if cuex is not None and "cuex" in V:
        def cx(q, k, v, b, m):
            q_c = jnp.transpose(q, (0, 2, 1, 3))[None]; k_c = jnp.transpose(k, (0, 2, 1, 3))[None]; v_c = jnp.transpose(v, (0, 2, 1, 3))[None]
            b_c = b[None].astype(jnp.float32); m_c = m[None]
            o, _, _ = cuex.triangle_attention(q_c, k_c, v_c, b_c, m_c, scale=float(D ** -0.5))
            return jnp.transpose(jnp.squeeze(o, 0), (0, 2, 1, 3))
        out = run("cuex.triangle_attention", cx, q, k, v, bias, mask)
        if out is not None and ref is not None:
            print(f"    max|cuex-xla| = {np.abs(out-ref).max():.4f}", flush=True)
