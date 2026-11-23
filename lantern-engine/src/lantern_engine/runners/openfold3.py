"""OpenFold 3 (of3) — wraps core/openfold3/inference{,_jax}.py.

Mirrors casp17 inferences/openfold3_p2/run_of3{,_jax}.sbatch.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..config import runtime_env, venv_python
from ._common import (META_DIRS, Plan, Request, checkpoint_path, flatten_seed_dirs, remove_files,
                      resolve_engines, stage_input)

FAMILY = "openfold3"
_CLUTTER = ("experiment_config.json", "model_config.json", "inference_query_set.json", "summary.txt")


def _pairing_mode(req: Request) -> str:
    if req.pairing_mode:
        return req.pairing_mode
    env = os.environ.get("OF3_PAIRING_MODE")
    if env in ("precomputed", "online"):
        return env
    # casp17 convention: pairing_mode.txt next to input.json or one level up (variant dir)
    for d in (req.input.resolve().parent, req.input.resolve().parent.parent):
        f = d / "pairing_mode.txt"
        if f.is_file():
            v = f.read_text().strip()
            if v in ("precomputed", "online"):
                return v
    return "precomputed"


def plan(req: Request) -> Plan:
    site, root = req.site, req.site.core
    jax = req.backend == "jax"
    venv = site.jax_venv_for(req.model.tag) if jax else site.venv
    script = root / "openfold3" / ("inference_jax.py" if jax else "inference.py")
    staged = stage_input(req, also_seeds_key=True)
    ckpt = checkpoint_path(req)
    ds = req.diffusion_samples if req.diffusion_samples is not None else 5

    cmd = [str(venv_python(venv)), str(script)]
    if jax:
        cmd += ["--jax_dtype", str(site.jax.get("dtype", "bf16")),
                "--jax_cache_dir", str(site.cache_dir / "jax_openfold3")]
    cmd += ["--query", str(staged), "--checkpoint", str(ckpt),
            "--output-dir", str(req.out.resolve()), "--num-diffusion-samples", str(ds)]
    notes: dict = {"diffusion_samples": ds}
    if req.backend == "trt":
        e = resolve_engines(req)
        cmd += ["--engines-dir", str(e)]
        notes["engines_dir"] = str(e)
    elif req.backend == "torch":
        cmd += ["--no-trt"]
    cmd += req.extra_args

    pm = _pairing_mode(req)
    notes["pairing_mode"] = pm
    extra_env: dict = {"OF3_PAIRING_MODE": pm, "DS_BUILD_OPS": "0"}
    if req.num_recycles is not None:
        extra_env["OF3_NUM_RECYCLES"] = req.num_recycles
    if req.template_weight is not None:
        extra_env["OF3_TPL_WEIGHT"] = req.template_weight
    # aux_heads pairformer runs in PyTorch over the full sample batch; 24 GB cards need per-sample.
    if req.gpu and req.gpu.tag == "rtx4090" and "OF3_PER_SAMPLE_TOKEN_CUTOFF" not in os.environ:
        extra_env["OF3_PER_SAMPLE_TOKEN_CUTOFF"] = "0"
    # Short per-run TMPDIR (unix-socket path limit) so concurrent jobs don't collide.
    tmp = Path(tempfile.mkdtemp(prefix="of3-", dir="/tmp"))
    extra_env["TMPDIR"] = str(tmp)
    if os.environ.get("LANTERN_ENGINE_PROFILE") == "1":
        extra_env["LANTERN_ENGINE_PROFILE_OUT"] = str(req.out.resolve() / "profile.json")

    env = runtime_env(
        site, venv,
        pythonpath=[root / "models" / "openfold-3", root / "thirdparty", root],
        extra=extra_env, jax=jax, model_tag=req.model.tag, family=FAMILY,
    )
    return Plan(cmd=cmd, env=env, cwd=root, venv=venv, staged_input=staged, notes=notes, cleanup=[tmp])


def post(req: Request, plan: Plan) -> list[Path]:
    # OF3 writes <out>/<query_id>/seed_<S>/sample_<i>/ ; flatten and drop native clutter.
    seed_dirs = sorted(p for p in req.out.glob("*/seed_*") if p.is_dir() and p.parent.name not in META_DIRS)
    moved = flatten_seed_dirs(req.out, seed_dirs)
    remove_files([req.out / c for c in _CLUTTER])
    remove_files(plan.cleanup)
    return moved
