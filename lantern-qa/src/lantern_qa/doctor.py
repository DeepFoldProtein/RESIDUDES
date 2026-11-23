"""`lantern-qa doctor`: is this environment able to reproduce the CASP17 QA numbers?

Checks, each reported as ok / warn / fail:
  python + pinned packages   versions equal to pyproject pins (the CASP17 production venv)
  torch / GPU                torch build, CUDA visible, device names
  data                       every manifest file present with the right size (sha256: `data verify`)
  env                        the EMA_* / LANTERN_QA_* variables the scorers will read
  optional                   ANARCI venv (antibody CDR / aADARS), TCR Pierce tools
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import tomllib
from pathlib import Path

from lantern_qa.data import load_manifest, status
from lantern_qa.paths import PROJECT_DIR, configure_env, data_root


def _pins() -> dict[str, str]:
    with open(PROJECT_DIR / "pyproject.toml", "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    return {d.split("==")[0].strip(): d.split("==")[1].strip() for d in deps if "==" in d}


def run(as_json: bool = False) -> int:
    rows: list[tuple[str, str, str]] = []           # (status, check, detail)
    add = lambda st, name, detail="": rows.append((st, name, detail))

    v = sys.version.split()[0]
    add("ok" if v.startswith("3.12.") else "fail", "python", f"{v} ({sys.executable})")
    mism = []
    for name, want in _pins().items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            mism.append(f"{name} missing")
            continue
        if have != want:
            mism.append(f"{name} {have} != {want}")
    add("ok" if not mism else "fail", "pinned packages", "all match pyproject" if not mism else "; ".join(mism))

    try:
        import torch
        cuda = torch.cuda.is_available()
        devs = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
        add("ok" if cuda else "warn", "torch", f"{torch.__version__} (CUDA {torch.version.cuda}); "
            + (f"{len(devs)} GPU: {', '.join(sorted(set(devs)))}" if cuda else "no GPU visible: MPNN/DFIRE/clustering fall back to CPU (slow)"))
    except Exception as exc:                        # noqa: BLE001
        add("fail", "torch", f"not importable: {exc} (install with `uv sync --extra cu130` or `--extra cu124`)")

    root = data_root()
    files = load_manifest()
    st = [status(e, root, full=False) for e in files]
    missing = [e["path"] for e, s in zip(files, st) if s != "ok"]
    add("ok" if not missing else "fail", "data", f"{root}: {len(files) - len(missing)}/{len(files)} files"
        + ("" if not missing else f"; missing/mismatched e.g. {missing[0]} -> `lantern-qa data fetch`"))

    env = configure_env()
    bad_env = [k for k, p in env.items() if not Path(p).exists()]
    add("ok" if not bad_env else "warn", "env", ", ".join(f"{k}={p}" for k, p in env.items() if k in bad_env)
        or "EMA_* / LANTERN_QA_* point at existing paths")

    cdr_bin = os.environ.get("LANTERN_QA_CDR_BIN", "/gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin")
    has_anarci = Path(cdr_bin, "ANARCI").exists() or shutil.which("ANARCI")
    add("ok" if has_anarci else "warn", "optional: ANARCI", cdr_bin if has_anarci else
        "not found: antibody CDR masking / aADARS features unavailable (set LANTERN_QA_CDR_BIN)")
    tcr = os.environ.get("LANTERN_QA_TCR_TOOLS", "/gpfs/deepfold/casp/casp17/target_notes/_tcr_tools")
    add("ok" if Path(tcr, "tcr3d_pierce.py").exists() else "warn", "optional: TCR Pierce angles",
        tcr if Path(tcr, "tcr3d_pierce.py").exists() else "not found (set LANTERN_QA_TCR_TOOLS); only --method tcr-angle needs it")

    if as_json:
        print(json.dumps({"host": platform.node(), "checks": [dict(status=s, check=c, detail=d) for s, c, d in rows]}, indent=1))
    else:
        print(f"lantern-qa doctor on {platform.node()}")
        for s, c, d in rows:
            print(f"  [{s:4s}] {c:28s} {d}")
    return 1 if any(s == "fail" for s, _, _ in rows) else 0
