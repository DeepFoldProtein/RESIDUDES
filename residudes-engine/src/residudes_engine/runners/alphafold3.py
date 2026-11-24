"""AlphaFold 3 (af3): wraps core/models/alphafold3/run_alphafold.py, DeepMind's AF3 3.0.2
with the CASP17 fork's unified writer, cross-chain templates and template weight.

Backends: `jax` runs DeepMind's Haiku model, `torch` the PyTorch port in
core/alphafold3/af3torch (same featurisation, seeds and writer; only the forward
pass differs). Both run in the JAX environment (`venv_jax`, core/jaxenv), which
carries torch as well. TensorRT engines for the port are the next step. Mirrors
inferences/af3/run_af3.sbatch from the CASP17 pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..config import SiteError, runtime_env, venv_python
from ._common import META_DIRS, Plan, Request, flatten_seed_dirs, remove_files, resolve_engines, stage_input

FAMILY = "alphafold3"
WEIGHT_FILES = ("af3.bin", "af3.bin.zst")
WEIGHTS_HELP = ("AlphaFold 3 parameters are licensed to each user by Google DeepMind and are not "
                "downloaded by residudes-engine: request access at https://github.com/google-deepmind/alphafold3 "
                "(WEIGHTS_TERMS_OF_USE.md), then put af3.bin (or af3.bin.zst) in <checkpoints>/af3/ "
                "or point [site] af3_weights at the directory that holds it.")


def weights_dir(req: Request) -> Path:
    site = req.site
    d = site.af3_weights or (site.checkpoints / "af3")
    if not any((d / f).is_file() for f in WEIGHT_FILES):
        raise SiteError(f"AlphaFold 3 parameters not found under {d}. {WEIGHTS_HELP}")
    return d


def _with_flag(existing: str | None, flag: str) -> str:
    return existing if existing and flag in existing else " ".join(x for x in (existing, flag) if x)


def plan(req: Request) -> Plan:
    site, root = req.site, req.site.core
    if req.backend not in ("jax", "torch", "trt"):
        raise SiteError(f"AlphaFold 3 runs on jax, torch or trt; --backend {req.backend} is not available for af3.")
    torch = req.backend in ("torch", "trt")
    venv = site.jax_venv_for(req.model.tag)
    script = (root / "alphafold3" / "inference.py") if torch else (root / "models" / "alphafold3" / "run_alphafold.py")
    if not script.is_file():
        raise SiteError(f"AlphaFold 3 entry point missing: {script}")
    staged = stage_input(req)
    wdir = weights_dir(req)
    ds = req.diffusion_samples if req.diffusion_samples is not None else 5
    nr = req.num_recycles if req.num_recycles is not None else 10
    jcfg = site.jax
    cache = site.cache_dir / "jax_alphafold3"
    cache.mkdir(parents=True, exist_ok=True)

    cmd = [str(venv_python(venv)), str(script),
           "--json_path", str(staged), "--model_dir", str(wdir), "--output_dir", str(req.out.resolve()),
           "--run_data_pipeline=false", "--run_inference=true", "--save_distogram=true", "--force_output_dir=true",
           "--num_recycles", str(nr), "--num_diffusion_samples", str(ds),
           "--jax_compilation_cache_dir", str(cache),
           "--flash_attention_implementation", str(jcfg.get("af3_attention", "triton"))]
    notes: dict = {"diffusion_samples": ds, "num_recycles": nr, "weights": str(wdir)}
    if req.backend == "trt":
        e = resolve_engines(req)
        cmd += ["--engines_dir", str(e)]
        notes["engines_dir"] = str(e)
    cmd += req.extra_args

    if torch:
        # JAX only runs the fork's device check here; the card belongs to torch (and TensorRT).
        extra_env: dict = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_CLIENT_MEM_FRACTION": "0.02"}
        notes["port"] = "core/alphafold3/af3torch"
    else:
        # The upstream container's XLA defaults: AF3 owns the whole card, so it preallocates.
        extra_env = {
            "XLA_FLAGS": _with_flag(os.environ.get("XLA_FLAGS"), "--xla_gpu_enable_triton_gemm=false"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": str(jcfg.get("af3_mem_fraction", 0.95)),
        }
    if req.template_weight is not None:
        extra_env["AF3_TPL_WEIGHT"] = req.template_weight
        notes["template_weight"] = req.template_weight
    # XLA assembles PTX with the ptxas under CUDA_ROOT; the Triton kernels (tokamax, via jax-triton) are assembled by
    # a second search that only walks PATH and /usr/local/cuda. A node toolkit older than the one the JAX plugin and
    # Triton were built with fails with "Unsupported .version"; the venv ships a matching ptxas, so both see it first.
    nvcc_dir = next((nv for nv in sorted(venv.glob("lib/python3.*/site-packages/nvidia/cu13")) if (nv / "bin" / "ptxas").is_file()), None)
    if nvcc_dir is not None:
        extra_env["CUDA_ROOT"] = str(nvcc_dir)
    env = runtime_env(site, venv, pythonpath=[root / "thirdparty", root] if torch else [], extra=extra_env, jax=True,
                      model_tag=req.model.tag, family=FAMILY)
    if nvcc_dir is not None:
        env["PATH"] = f"{nvcc_dir / 'bin'}:{env['PATH']}"
    # JAX 0.11 reads XLA_CLIENT_MEM_FRACTION and refuses to start the CUDA plugin when the
    # deprecated XLA_PYTHON_CLIENT_MEM_FRACTION (set by runtime_env for the older JAX, or by a shell) is also present.
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    return Plan(cmd=cmd, env=env, cwd=script.parent, venv=venv, staged_input=staged, notes=notes)


def post(req: Request, plan: Plan) -> list[Path]:
    """AF3 writes <out>/<name>/seed_<S>/sample_<i>/ (the unified writer) plus a
    top-ranked duplicate and ranking_scores.csv under <out>/<name>/. Lift the seed
    directories to the residudes-engine layout and drop the duplicate."""
    out = req.out.resolve()
    name_dirs = [p for p in out.iterdir()
                 if p.is_dir() and p.name not in META_DIRS and not p.name.startswith(("seed_", "sample_"))]
    seed_dirs = sorted(p for nd in name_dirs for p in nd.glob("seed_*") if p.is_dir())
    moved = flatten_seed_dirs(out, seed_dirs) if seed_dirs else []
    # Only clutter may remain in <name>/ now; never delete a directory that still holds samples.
    leftovers = [nd for nd in name_dirs if nd.exists()
                 and not any(p.is_dir() and p.name.startswith(("seed_", "sample_")) for p in nd.iterdir())]
    remove_files(leftovers + [out / "TERMS_OF_USE.md"])
    return moved
