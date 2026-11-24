"""Boltz-2 (bt2) — wraps core/scripts/boltz_inference{,_jax}.py.

Reproduces the CASP17 production invocation. Accepts an AF3-dialect
input.json (converted at runtime by core/scripts/af3_input_adapter.py) or a
Boltz YAML.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import runtime_env, venv_python
from ._common import (Plan, Request, checkpoint_path, flatten_seed_dirs, input_seeds,
                      remove_files, resolve_engines, stage_input)

FAMILY = "boltz"


def plan(req: Request) -> Plan:
    site, root = req.site, req.site.core
    jax = req.backend == "jax"
    venv = site.jax_venv_for(req.model.tag) if jax else site.venv
    script = root / "scripts" / ("boltz_inference_jax.py" if jax else "boltz_inference.py")
    staged = stage_input(req)
    ckpt = checkpoint_path(req)
    ds = req.diffusion_samples if req.diffusion_samples is not None else 5

    cmd = [str(venv_python(venv)), str(script)]
    notes: dict = {"diffusion_samples": ds}
    if req.backend == "trt":
        e = resolve_engines(req)
        cmd += ["--engines_dir", str(e)]
        notes["engines_dir"] = str(e)
    elif req.backend == "torch":
        cmd += ["--no_trt"]
    if jax:
        cmd += ["--jax_dtype", str(site.jax.get("dtype", "bf16")),
                "--jax_cache_dir", str(site.cache_dir / "jax_boltz")]
        # Hybrid trunk/diffusion split: JAX pairformers + TRT token transformer.
        # RESIDUDES_ENGINE_JAX_TRT_DIFFUSION=1 (site [jax.env] or environment) and engines
        # for this GPU → pass the engine set; the JAX entry swaps only the
        # diffusion transformer. Without engines the JAX path runs as before.
        jenv = site.jax.get("env", {}) if isinstance(site.jax.get("env"), dict) else {}
        want = os.environ.get("RESIDUDES_ENGINE_JAX_TRT_DIFFUSION", jenv.get("RESIDUDES_ENGINE_JAX_TRT_DIFFUSION", "0"))
        if str(want) == "1":
            try:
                e = resolve_engines(req)
            except Exception as exc:  # no engines → plain JAX
                notes["jax_trt_diffusion"] = f"off ({exc.__class__.__name__})"
            else:
                cmd += ["--engines_dir", str(e)]
                notes["jax_trt_diffusion"] = str(e)
    cmd += ["predict", str(staged), "--out_dir", str(req.out.resolve()),
            "--checkpoint", str(ckpt), "--model", "boltz2", "--output_format", "mmcif",
            "--diffusion_samples", str(ds), "--num_workers", "0" if jax else "2"]
    seed = req.seed
    if seed is None:
        seeds = input_seeds(req.input)
        seed = seeds[0] if len(seeds) == 1 else None
    if seed is not None:
        cmd += ["--seed", str(seed)]
    notes["seed"] = seed
    # Always explicit: Boltz's own default is 3 recycles and only the torch/TRT
    # entry script re-defaults it to 10 — the JAX entry does not, so without this
    # the backends would not do the same work. 10 matches the other models.
    n_rec = req.num_recycles if req.num_recycles is not None else 10
    cmd += ["--recycling_steps", str(n_rec)]
    notes["num_recycles"] = n_rec
    cmd += ["--override"] + req.extra_args

    extra_env: dict = {"BOLTZ_CACHE": str(site.boltz_cache or (site.checkpoints / "boltz2"))}
    if seed is not None:
        # `--seed` only drives sampling; the unified writer takes summary.seed and
        # its native seed_<N>/ name from BT2_SEED (set via the BT2_SEEDS wrapper).
        extra_env["BT2_SEED"] = seed
        extra_env["BT2_SEEDS"] = seed
    if req.template_weight is not None:
        extra_env["BT2_TPL_WEIGHT"] = req.template_weight
    if os.environ.get("RESIDUDES_ENGINE_PROFILE") == "1":
        # per-stage timing of the forward pass → <out>/profile.json (see docs/benchmark.md)
        extra_env["RESIDUDES_ENGINE_PROFILE_OUT"] = str(req.out.resolve() / "profile.json")
    env = runtime_env(
        site, venv, pythonpath=[root / "thirdparty"],
        extra=extra_env, jax=jax, model_tag=req.model.tag, family=FAMILY,
    )
    return Plan(cmd=cmd, env=env, cwd=root, venv=venv, staged_input=staged, notes=notes)


def post(req: Request, plan: Plan) -> list[Path]:
    # Boltz writes <out>/boltz_results_<stem>/predictions/<stem>/seed_<S>/ …
    nested = sorted(p for p in req.out.glob("boltz_results_*") if p.is_dir())
    seed_dirs = sorted(p for p in req.out.glob("boltz_results_*/predictions/*/seed_*") if p.is_dir())
    if not seed_dirs:
        # legacy flat layout (no seed dir): lift the prediction dir itself
        seed_dirs = sorted(p for p in req.out.glob("boltz_results_*/predictions/*") if p.is_dir())
    moved = flatten_seed_dirs(req.out, seed_dirs)
    remove_files(nested)
    return moved
