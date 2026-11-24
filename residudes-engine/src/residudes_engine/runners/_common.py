"""Shared pieces: request/plan types, input staging, subprocess runner,
native→unified output flattening, run manifest."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import RESIDUDES_ENGINE_ROOT, GPU, Site, SiteError
from ..registry import ModelSpec, engine_max_seq_len, find_engine_dir


# The run-metadata folder in every output. Outputs written under an earlier name have ".lantern-engine" or ".foldmux".
META_DIR = ".residudes-engine"
META_DIRS = (META_DIR, ".lantern-engine", ".foldmux")


def meta_of(seed_dir: Path) -> Path:
    """The run-metadata folder of an output, old name included."""
    for name in META_DIRS:
        if (Path(seed_dir) / name).is_dir():
            return Path(seed_dir) / name
    return Path(seed_dir) / META_DIR


@dataclass
class Request:
    model: ModelSpec
    backend: str
    input: Path
    out: Path
    site: Site
    gpu: GPU | None
    seed: int | None = None
    num_recycles: int | None = None
    diffusion_samples: int | None = None
    template_weight: float | None = None
    pairing_mode: str | None = None
    engines_dir: Path | None = None
    extra_args: list[str] = field(default_factory=list)
    dry_run: bool = False
    keep_native: bool = False

    @property
    def meta_dir(self) -> Path:
        return self.out / META_DIR


@dataclass
class Plan:
    cmd: list[str]
    env: dict[str, str]
    cwd: Path
    venv: Path
    staged_input: Path
    notes: dict[str, Any] = field(default_factory=dict)
    cleanup: list[Path] = field(default_factory=list)


# ── input ──────────────────────────────────────────────────────────────────

def load_af3_dialect(path: Path) -> dict | None:
    if path.suffix != ".json" or not path.is_file():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("dialect") == "alphafold3" else None


def input_seeds(path: Path) -> list[int]:
    d = load_af3_dialect(path)
    if not d:
        return []
    return [int(s) for s in (d.get("modelSeeds") or [])]


def stage_input(req: Request, *, also_seeds_key: bool = False) -> Path:
    """Place the input under <out>/.residudes-engine/. For AF3-dialect JSON with
    ``--seed``, rewrite modelSeeds (and ``seeds`` for OF3) like the CASP17
    submit scripts did. Other formats (Boltz YAML, native JSON) pass through."""
    src = req.input.resolve()
    data = load_af3_dialect(src)
    if data is None:
        return src
    if req.seed is not None:
        data["modelSeeds"] = [req.seed]
        if also_seeds_key:
            data["seeds"] = [req.seed]
    req.meta_dir.mkdir(parents=True, exist_ok=True)
    dst = req.meta_dir / "input.json"
    dst.write_text(json.dumps(data, indent=2) + "\n")
    return dst


# ── resolution helpers ─────────────────────────────────────────────────────

def checkpoint_path(req: Request) -> Path:
    p = req.site.checkpoints / req.model.checkpoint
    if not p.exists():
        raise SiteError(f"checkpoint missing: {p}")
    return p


def estimate_tokens(input_path: Path) -> int | None:
    """Rough token count of an AF3-dialect input, for choosing an engine size.

    Polymers contribute one token per residue per copy; ligands count their
    heavy atoms from the SMILES (or 64 per CCD code) since AF3-style models
    tokenise ligands per atom. 1 % headroom (polymer counts are exact). None for non-AF3 inputs."""
    d = load_af3_dialect(input_path)
    if not d:
        return None
    n = 0
    for ent in d.get("sequences", []):
        for kind, body in ent.items():
            if not isinstance(body, dict):
                continue
            ids = body.get("id", [])
            copies = len(ids) if isinstance(ids, list) else 1
            if kind in ("protein", "rna", "dna"):
                n += len(body.get("sequence", "")) * copies
            elif kind == "ligand":
                smiles = body.get("smiles")
                if smiles:
                    import re
                    n += len(re.findall(r"Cl|Br|[BCNOSPFI]|Se|Si", smiles)) * copies
                else:
                    n += 64 * len(body.get("ccdCodes", [])) * copies
    return int(n * 1.01) + 1 if n else None


def resolve_engines(req: Request) -> Path:
    if req.engines_dir is not None:
        e = req.engines_dir
    else:
        if req.gpu is None or req.gpu.tag is None:
            cc = req.gpu.compute_cap if req.gpu else "no GPU detected"
            raise SiteError(
                f"cannot pick TRT engines for {req.model.tag}: {cc}. "
                f"Pass --engines-dir, or use --backend torch/jax."
            )
        n_tokens = estimate_tokens(req.input)
        e = find_engine_dir(req.site.engines, req.gpu.tag, req.model, n_tokens=n_tokens)
        if e is None:
            raise SiteError(
                f"no TRT engines for {req.model.tag} on {req.gpu.tag} under {req.site.engines}. "
                f"Build them (`residudes-engine engines build --model {req.model.tag}`), pass --engines-dir, "
                f"or use --backend torch."
            )
    if not (e / "pairformer_engine" / "trt" / "rank0.engine").is_file():
        raise SiteError(f"no pairformer_engine/trt/rank0.engine under {e}")
    return e


# ── execution ──────────────────────────────────────────────────────────────

def _shell_join(cmd: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(c) for c in cmd)


def run_plan(req: Request, plan: Plan) -> int:
    """Run the inference process, tee-ing output to <out>/inference.log."""
    req.out.mkdir(parents=True, exist_ok=True)
    log = req.out / "inference.log"
    print(f"▶ cwd : {plan.cwd}")
    print(f"▶ cmd : {_shell_join(plan.cmd)}")
    print(f"▶ log : {log}")
    sys.stdout.flush()
    with log.open("a") as fh:
        fh.write(f"# residudes-engine {__version__}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n# {_shell_join(plan.cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(
            plan.cmd, cwd=str(plan.cwd), env=plan.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        fh.write(f"# exit {proc.returncode}\n")
    return proc.returncode


# ── output ─────────────────────────────────────────────────────────────────

def flatten_seed_dirs(out: Path, seed_dirs: list[Path]) -> list[Path]:
    """Move native ``…/seed_<N>/`` trees up: a single seed lands directly in
    <out>/, several land in <out>/seed_<N>/. Empty native parents are removed."""
    out = out.resolve()
    moved: list[Path] = []
    for sd in seed_dirs:
        sd = sd.resolve()
        dst = out if len(seed_dirs) == 1 else out / sd.name
        if sd == dst:
            moved.append(dst)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for item in sorted(sd.iterdir()):
            target = dst / item.name
            if target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        moved.append(dst)
        p = sd
        while p != out and p.exists() and out in p.parents:
            try:
                p.rmdir()
            except OSError:
                break
            p = p.parent
    return moved


def remove_files(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif p.exists() or p.is_symlink():
                p.unlink()
        except OSError:
            pass


def write_manifest(req: Request, plan: Plan, rc: int, t0: float, t1: float) -> Path:
    req.meta_dir.mkdir(parents=True, exist_ok=True)
    engines = plan.notes.get("engines_dir")
    man = {
        "residudes-engine": __version__,
        "model": req.model.tag,
        "family": req.model.family,
        "backend": req.backend,
        "seed": req.seed,
        "input": str(req.input.resolve()),
        "staged_input": str(plan.staged_input),
        "out": str(req.out.resolve()),
        "site": req.site.name,
        "site_file": str(req.site.path),
        "core": str(req.site.core),
        "residudes_engine_commit": _git_head(RESIDUDES_ENGINE_ROOT),
        "venv": str(plan.venv),
        "engines_dir": engines,
        "engine_max_seq_len": engine_max_seq_len(Path(engines)) if engines else None,
        "gpu": req.gpu.__dict__ if req.gpu else None,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cmd": plan.cmd,
        "env_overrides": {k: plan.env[k] for k in sorted(plan.env)
                          if k.startswith(("RESIDUDES_ENGINE_", "BIO_TRT_", "PT2_", "OF3_", "BT2_", "BOLTZ_", "XLA_", "JAX_",
                                           "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "LAYERNORM_TYPE",
                                           "PYTORCH_CUDA_ALLOC_CONF", "DS_BUILD_OPS", "CUDA_"))},
        "exit_code": rc,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "wall_s": round(t1 - t0, 1),
        "notes": {k: v for k, v in plan.notes.items() if k != "engines_dir"},
    }
    p = req.meta_dir / "run.json"
    p.write_text(json.dumps(man, indent=2, default=str) + "\n")
    return p


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None
