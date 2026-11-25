"""Protenix (pt2) — wraps core/protenix/inference{,_jax}.py.

Command lines reproduce the CASP17 production invocations for the TRT, JAX
and plain-PyTorch paths.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import runtime_env, venv_python
from ._common import META_DIRS, Plan, Request, checkpoint_path, flatten_seed_dirs, resolve_engines, stage_input

FAMILY = "protenix"


def plan(req: Request) -> Plan:
    site, root = req.site, req.site.core
    jax = req.backend == "jax"
    venv = site.jax_venv_for(req.model.tag) if jax else site.venv
    script = root / "protenix" / ("inference_jax.py" if jax else "inference.py")
    staged = stage_input(req)
    checkpoint_path(req)  # fail early if the .pt is missing
    n_cycle = req.num_recycles if req.num_recycles is not None else 10

    cmd = [str(venv_python(venv)), str(script)]
    if jax:
        cmd += ["--jax_dtype", str(site.jax.get("dtype", "bf16")), "--jax_variant", "v2",
                "--jax_cache_dir", str(site.cache_dir / "jax_protenix")]
    cmd += [
        "--model_name", req.model.extra["model_name"],
        "--input_json_path", str(staged),
        "--dump_dir", str(req.out.resolve()),
        "--load_checkpoint_dir", str(site.checkpoints),
        "--use_template", "true", "--use_msa", "true", "--use_seeds_in_json", "true",
        "--model.N_cycle", str(n_cycle),
    ]
    if site.kalign:
        cmd += ["--data.template.kalign_binary_path", str(site.kalign)]
    notes: dict = {"num_recycles": n_cycle}
    if req.backend == "trt":
        e = resolve_engines(req)
        cmd += ["--engines_dir", str(e)]
        notes["engines_dir"] = str(e)
    elif req.backend == "torch":
        cmd += ["--no_trt"]
    # Triangle modules at the Python layer: torch unless the JAX path owns them (cueq-jax).
    jax_env = (site.jax.get("env") or {})
    per_model = jax_env.get(req.model.tag) or {}
    cueq = str(per_model.get("RESIDUDES_ENGINE_JAX_CUEQ", per_model.get("RESIDUDES_ENGINE_JAX_CUEQ",
               jax_env.get("RESIDUDES_ENGINE_JAX_CUEQ", jax_env.get("RESIDUDES_ENGINE_JAX_CUEQ", "0")))))
    if not (jax and cueq == "1"):
        cmd += ["--triangle_attention", "torch", "--triangle_multiplicative", "torch"]
    cmd += req.extra_args

    extra_env = {}
    if req.template_weight is not None:
        extra_env["PT2_TPL_WEIGHT"] = req.template_weight
    if os.environ.get("RESIDUDES_ENGINE_PROFILE") == "1":
        extra_env["RESIDUDES_ENGINE_PROFILE_OUT"] = str(req.out.resolve() / "profile.json")
    env = runtime_env(
        site, venv,
        pythonpath=[root / "models" / "protenix", root / "thirdparty", root],
        extra=extra_env, jax=jax, model_tag=req.model.tag, family=FAMILY,
    )
    return Plan(cmd=cmd, env=env, cwd=root, venv=venv, staged_input=staged, notes=notes)


def post(req: Request, plan: Plan) -> list[Path]:
    # Protenix dumps <out>/<job name>/seed_<N>/… ; flatten to the unified layout.
    seed_dirs = sorted(p for p in req.out.glob("*/seed_*") if p.is_dir() and p.parent.name not in META_DIRS)
    return flatten_seed_dirs(req.out, seed_dirs)
