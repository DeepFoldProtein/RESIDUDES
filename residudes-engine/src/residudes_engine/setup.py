"""`residudes-engine setup …` and `residudes-engine engines …` helpers: model weights, site files,
TensorRT engine builds. Everything heavy runs inside the runtime venv."""
from __future__ import annotations

import hashlib
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import registry
from .config import RESIDUDES_ENGINE_ROOT, SITES_DIR, GPU, Site, SiteError, runtime_env, venv_python


# ── model weights ──────────────────────────────────────────────────────────

def _download_any(urls: list[str], dest: Path, expected_size: int | None) -> str:
    """Try each URL in order; return the one that worked. Content is verified by the caller."""
    errors = []
    for url in urls:
        try:
            print(f"  trying {url}")
            _download_url(url, dest, expected_size)
            return url
        except Exception as e:  # noqa: BLE001 — fall through to the next source
            errors.append(f"{url}: {type(e).__name__}: {e}")
            print(f"  ✗ {errors[-1]}")
    raise SystemExit("all sources failed:\n  " + "\n  ".join(errors)
                     + f"\nDownload the file by hand and place it at {dest}")


def _download_url(url: str, dest: Path, expected_size: int | None) -> None:
    """Fetch with curl when available (follows redirects, resumes, works even when the
    CLI's Python was built without ssl — pyenv builds often are); else urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        cmd = [curl, "-L", "--fail", "--retry", "3", "-C", "-", "--progress-bar", "-o", str(tmp), url]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"curl exit {rc}")
        if expected_size and tmp.stat().st_size != expected_size:
            raise RuntimeError(f"size {tmp.stat().st_size} != expected {expected_size}")
        tmp.replace(dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "residudes-engine"})
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or expected_size or 0)
        done, t0, last = 0, time.time(), 0.0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                pct = f"{100 * done / total:5.1f}%" if total else ""
                rate = done / max(time.time() - t0, 1e-6) / 2**20
                print(f"\r  {dest.name}: {done / 2**30:6.2f} GB {pct} {rate:6.0f} MB/s", end="", flush=True)
                last = time.time()
    print()
    if expected_size and tmp.stat().st_size != expected_size:
        raise SystemExit(f"{dest.name}: size {tmp.stat().st_size} != expected {expected_size}; download corrupt?")
    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _venv_python_snippet(site: Site, family_pythonpath: list[Path], code: str, dry_run: bool,
                         cwd: Path | None = None) -> int:
    """Run a short Python program inside the runtime venv. cwd must NOT be core/: its
    per-family backend dirs (core/boltz, core/protenix, …) would shadow the installed
    packages when '' is first on sys.path."""
    env = runtime_env(site, site.venv, pythonpath=family_pythonpath)
    cmd = [str(venv_python(site.venv)), "-c", code]
    print("▶", shlex.quote(cmd[0]), "-c", shlex.quote(code.strip().splitlines()[0] + " …"))
    if dry_run:
        return 0
    workdir = cwd or site.checkpoints
    workdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, env=env, cwd=str(workdir)).returncode


def setup_checkpoints(site: Site, tags: list[str], *, dry_run: bool = False, force: bool = False) -> int:
    """Fetch model weights into <site.checkpoints> using each publisher's channel."""
    rc = 0
    for tag in tags:
        spec = registry.get_model(tag)
        dest = site.checkpoints / spec.checkpoint
        w = spec.weights
        if dest.exists() and not force:
            size = dest.stat().st_size
            flag = "" if not w.get("size") or size == w["size"] else f"  (size {size} ≠ expected {w['size']}!)"
            print(f"▶ {tag}: present {dest}{flag}")
            continue
        print(f"▶ {tag}: fetching → {dest}")
        if dry_run:
            print(f"  method={w.get('method')} {' | '.join(w.get('urls', [])) or w.get('name', '')}")
            continue
        method = w.get("method")
        dest.parent.mkdir(parents=True, exist_ok=True)   # publishers' downloaders assume it exists
        if method == "url":
            _download_any(w["urls"], dest, w.get("size"))
        elif method == "openfold3":
            code = (
                "from pathlib import Path\n"
                "from openfold3.entry_points.parameters import download_model_parameters\n"
                f"download_model_parameters(Path({str(dest.parent)!r}), {w['name']!r}, skip_confirmation=True)\n"
            )
            root = site.core
            rc |= _venv_python_snippet(site, [root / "models" / "openfold-3", root / "thirdparty", root], code, dry_run)
        elif method == "boltz2":
            code = (
                "from pathlib import Path\n"
                "from boltz.main import download_boltz2\n"
                f"download_boltz2(Path({str(dest.parent)!r}))\n"
            )
            rc |= _venv_python_snippet(site, [site.core / "thirdparty"], code, dry_run)
        elif method == "manual":
            alt = getattr(site, "af3_weights", None)
            where = f" (or {alt}, from [site] af3_weights)" if alt else ""
            print(f"  not downloadable: {w.get('note')}{where}")
            if not (dest.exists() or (alt and any((alt / f).is_file() for f in ("af3.bin", "af3.bin.zst")))):
                rc = 1
            continue
        else:
            print(f"  ✗ no download method for {tag}", file=sys.stderr)
            rc = 1
        if dest.exists():
            if w.get("sha256"):
                print("  verifying sha256 …", end="", flush=True)
                got = sha256_of(dest)
                if got != w["sha256"]:
                    print(f"\n  ✗ sha256 mismatch: {got} (expected {w['sha256']}); file kept as {dest.name}.bad",
                          file=sys.stderr)
                    dest.rename(dest.with_suffix(dest.suffix + ".bad"))
                    rc = 1
                    continue
                print(" OK")
            print(f"  ✓ {dest} ({dest.stat().st_size / 2**30:.2f} GiB)")
        elif not dry_run:
            print(f"  ✗ {dest} not produced", file=sys.stderr)
            rc = 1
    return rc


# ── site file ──────────────────────────────────────────────────────────────

def write_site(name: str, *, checkpoints: Path, engines: Path, scheduler: str, partition: str | None,
               hosts: list[str] | None, force: bool) -> Path:
    """Create sites/<name>.toml from the template with the given paths."""
    template = (SITES_DIR / "local.example.toml").read_text()
    out = SITES_DIR / f"{name}.toml"
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    hosts = hosts if hosts is not None else [socket.gethostname()]
    lines = []
    for line in template.splitlines():
        key = line.split("=", 1)[0].strip()
        if line.startswith("#   residudes-engine") or line.startswith("# Template") or line.startswith("# writes") or line.startswith("# [site]") or line.startswith("# override"):
            continue
        if key == "name":
            line = f'name = "{name}"'
        elif key == "hosts":
            line = "hosts = [" + ", ".join(f'"{h}"' for h in hosts) + "]"
        elif key == "scheduler":
            line = f'scheduler = "{scheduler}"'
        elif key == "checkpoints":
            line = f'checkpoints = "{checkpoints}"'
        elif key == "engines":
            line = f'engines = "{engines}"'
        elif key == "partition" and partition:
            line = f'partition = "{partition}"'
        lines.append(line)
    out.write_text(f"# Site file written by `residudes-engine setup site` on {time.strftime('%Y-%m-%d')} ({socket.gethostname()}).\n"
                   + "\n".join(lines) + "\n")
    return out


# ── TensorRT engines ───────────────────────────────────────────────────────

_FAMILY_SCRIPT = {"protenix": "protenix", "openfold3": "openfold3", "boltz": "boltz", "alphafold3": "alphafold3"}
_FAMILY_FORK = {"protenix": "models/protenix", "openfold3": "models/openfold-3", "boltz": None, "alphafold3": None}


def engine_build_plan(site: Site, spec: registry.ModelSpec, gpu: GPU, *, kernel: str, max_seqlen: int,
                      min_seqlen: int, extra: list[str]) -> tuple[list[str], dict[str, str], Path]:
    """Command + env to build TRT engines for one model on the GPU of this machine."""
    if gpu.tag is None:
        raise SiteError(f"unknown GPU class (compute capability {gpu.compute_cap}); add it to registry.GPUS")
    if spec.family not in _FAMILY_SCRIPT:
        raise SiteError(f"{spec.name} has no TensorRT engine build; it runs on {', '.join(spec.backends)}")
    root = site.core
    out = spec.engine_dir_for_build(site.engines, gpu.tag, kernel, max_seqlen)
    ckpt = site.checkpoints / spec.checkpoint
    if spec.family == "alphafold3":
        # the build script takes the directory holding af3.bin (licensed weights; see [site] af3_weights)
        ckpt = site.af3_weights or (site.checkpoints / "af3")
        if not any((ckpt / f).is_file() for f in ("af3.bin", "af3.bin.zst")):
            raise SiteError(f"AlphaFold 3 parameters not found under {ckpt} (set [site] af3_weights)")
    elif not ckpt.exists():
        raise SiteError(f"checkpoint missing: {ckpt} (run `residudes-engine setup checkpoints`)")
    script = root / _FAMILY_SCRIPT[spec.family] / "trt" / "build_engines.sh"
    cmd = ["bash", str(script), "--checkpoint", str(ckpt), "--engines-dir", str(out),
           "--min-seqlen", str(min_seqlen), "--max-seqlen", str(max_seqlen)]
    if spec.family == "protenix":
        cmd += ["--model-name", spec.extra["model_name"]]
    cmd += extra
    pythonpath = [root / "thirdparty", root]
    fork = _FAMILY_FORK[spec.family]
    if fork:
        pythonpath.insert(0, root / fork)
    env = runtime_env(site, site.venv, pythonpath=pythonpath,
                      extra={"TRIANGLE_ATTN_BACKEND": kernel.upper(), "PYTHON": str(venv_python(site.venv)),
                             "ENGINES_DIR": str(out)})
    return cmd, env, out


def list_engines(site: Site, gpu: GPU | None) -> list[tuple[str, str, Path | None, int | None]]:
    rows = []
    tags = [gpu.tag] if gpu and gpu.tag else sorted({t for m in registry.MODELS.values() for t in m.engines})
    for g in tags:
        for spec in registry.MODELS.values():
            d = registry.find_engine_dir(site.engines, g, spec)
            rows.append((g, spec.tag, d, registry.engine_max_seq_len(d) if d else None))
    return rows


def slurm_wrap(site: Site, args: list[str], *, name: str, log_dir: Path, time_limit: str | None,
               gpu_constraint: str | None, dry_run: bool) -> int:
    """Submit `residudes-engine <args>` as one Slurm job via scripts/slurm/residudes-engine.sbatch."""
    sl = site.slurm
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["sbatch", "-J", name, "-p", str(sl.get("partition", "normal")),
           "--gres", str(sl.get("gres", "gpu:1")), "-c", str(sl.get("cpus", 4)),
           "-t", str(time_limit or sl.get("time", "02:00:00")),
           "-o", str(log_dir / f"{name}.%j.out"), "-e", str(log_dir / f"{name}.%j.err"),
           f"--export=ALL,RESIDUDES_ENGINE_ROOT={RESIDUDES_ENGINE_ROOT},RESIDUDES_ENGINE_SITE_FILE={site.path}"]
    if gpu_constraint:
        cmd += ["-C", gpu_constraint]
    cmd += [str(RESIDUDES_ENGINE_ROOT / "scripts" / "slurm" / "residudes-engine.sbatch"), *args, "--site", str(site.path)]
    print("▶", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    return res.returncode
