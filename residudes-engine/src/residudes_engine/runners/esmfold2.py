"""ESMFold2 (ef2) — wraps the esm checkout's inferences/esmfold2/run_inference.py.

ESMFold2 (EvolutionaryScale / Biohub, on the ESMC-6B language model) is the one
model here with no TensorRT engine and no template embedder: it runs in PyTorch
on a single GPU, from a sequence alone or with an unpaired MSA. Mirrors
inferences/esmfold2/run_ef2.sbatch from the CASP17 pipeline.

The runner python lives with the esm checkout because it imports ``esm.*``, so a site
points ``[site] esm`` at that checkout and the runner uses its conda env (python 3.12;
the esm package pins >=3.12,<3.13, which the main venv does not satisfy).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import SiteError, runtime_env
from ._common import Plan, Request, flatten_seed_dirs, input_seeds, remove_files, stage_input

FAMILY = "esmfold2"
RUNNER_REL = Path("inferences") / "esmfold2" / "run_inference.py"
SETUP_HELP = ("ESMFold2 runs from an esm checkout carrying inferences/esmfold2/ and its own python 3.12 "
              "conda env: run that directory's setup_env.sh, then set [site] esm to the checkout "
              "(and [site] esm_python if the interpreter is not <esm>/.conda/bin/python).")


def esm_dir(req: Request) -> Path:
    d = req.site.esm
    if d is None:
        raise SiteError(f"{req.site.path.name}: [site].esm is not set. {SETUP_HELP}")
    if not (d / RUNNER_REL).is_file():
        raise SiteError(f"ESMFold2 entry point missing: {d / RUNNER_REL}. {SETUP_HELP}")
    return d


def esm_python(req: Request) -> Path:
    """The esm env's interpreter: [site] esm_python, else <esm>/.conda/bin/python."""
    py = req.site.esm_python or (esm_dir(req) / ".conda" / "bin" / "python")
    if not py.is_file():
        raise SiteError(f"esm env python missing: {py}. {SETUP_HELP}")
    return py


def weights_home(req: Request) -> Path:
    """HF cache holding biohub/ESMFold2: [site] esmfold2_weights, else <checkpoints>/esmfold2_hf."""
    return req.site.esmfold2_weights or (req.site.checkpoints / "esmfold2_hf")


def templated_chains(path: Path) -> list[str]:
    """Chain ids of an AF3-dialect input that carry templates, '<multimerTemplates>' included.

    ESMFold2 has no template embedder. build_input.py refuses such an input, but
    that is after the job has a GPU, so the same check runs here while planning."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("dialect") != "alphafold3":
        return []
    out: list[str] = []
    for entry in data.get("sequences") or []:
        if not isinstance(entry, dict):
            continue
        for kind, body in entry.items():
            if kind == "protein" and isinstance(body, dict) and body.get("templates"):
                ids = body.get("id")
                out += [str(i) for i in ids] if isinstance(ids, list) else [str(ids)]
    if data.get("multimerTemplates"):
        out.append("<multimerTemplates>")
    return out


def plan(req: Request) -> Plan:
    site = req.site
    if req.backend != "torch":
        raise SiteError("ESMFold2 has no TensorRT engine and no JAX port: it runs in PyTorch "
                        f"(--backend torch), not {req.backend}.")
    script = esm_dir(req) / RUNNER_REL
    python = esm_python(req)
    tmpl = templated_chains(req.input)
    if tmpl:
        raise SiteError(f"ESMFold2 has no template embedder, but {req.input} carries templates for "
                        f"chain(s) {', '.join(tmpl)}. Build a template-free input variant "
                        "(single_seq, or <variant>_msa_only, which keeps the unpaired MSA).")

    staged = stage_input(req)
    seed = req.seed
    if seed is None:
        seeds = input_seeds(req.input)
        seed = seeds[0] if len(seeds) == 1 else 1
    ds = req.diffusion_samples if req.diffusion_samples is not None else 5

    # One seed per run, samples flat under <out>: <out> is itself the seed dir, as for bt2 and af3.
    cmd = [str(python), str(script), str(staged), str(req.out.resolve()),
           "--seeds", str(seed), "--layout", "flat", "--diffusion-samples", str(ds)]
    notes: dict = {"seed": seed, "diffusion_samples": ds, "esm": str(esm_dir(req))}
    if req.num_recycles is not None:
        # ESMFold2 calls its recycles "loops" and defaults to 3; only an explicit --num-recycles moves it.
        cmd += ["--num-loops", str(req.num_recycles)]
        notes["num_loops"] = req.num_recycles
    cmd += req.extra_args

    env = runtime_env(site, python.parent.parent, pythonpath=[], extra={"HF_HOME": str(weights_home(req))},
                      model_tag=req.model.tag, family=FAMILY)
    # No TensorRT plugin here, so the site's libstdc++ must not be forced ahead of the conda env's own.
    env.pop("LD_PRELOAD", None)
    return Plan(cmd=cmd, env=env, cwd=script.parent, venv=python.parent.parent, staged_input=staged, notes=notes)


def post(req: Request, plan: Plan) -> list[Path]:
    """`--layout flat` writes sample_<i>/ straight into <out>, so there is nothing to lift;
    a nested run (--layout nested through extra args) leaves seed_<S>/ to flatten. The
    runner also drops a copy of the input beside them, which the manifest already records."""
    out = req.out.resolve()
    seed_dirs = sorted(p for p in out.glob("seed_*") if p.is_dir())
    moved = flatten_seed_dirs(out, seed_dirs) if seed_dirs else []
    remove_files([out / "input.json"])
    return moved
