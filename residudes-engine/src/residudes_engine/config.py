"""Site configuration — the only place machine-specific paths are allowed.

A site is a TOML file under ``sites/`` (or any path). Resolution order:
explicit ``--site``, ``$RESIDUDES_ENGINE_SITE_FILE``, ``$RESIDUDES_ENGINE_SITE`` (name or path),
then hostname match against each site's ``hosts`` patterns. Every ``[site]``
key can be overridden by ``RESIDUDES_ENGINE_<KEY>``.
"""
from __future__ import annotations

import fnmatch
import os
import socket
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry

RESIDUDES_ENGINE_ROOT = Path(os.environ.get("RESIDUDES_ENGINE_ROOT") or Path(__file__).resolve().parents[2])
SITES_DIR = RESIDUDES_ENGINE_ROOT / "sites"

_GLIBCXX_NEEDED = b"GLIBCXX_3.4.32"
_LIBSTDCXX_DEFAULTS = (
    "/opt/conda/lib/libstdc++.so.6",
    "/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so",
    "/usr/lib/gcc/x86_64-linux-gnu/12/libstdc++.so",
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib64/libstdc++.so.6",
)


class SiteError(RuntimeError):
    pass


@dataclass
class Site:
    name: str
    path: Path
    scheduler: str
    core: Path
    venv: Path
    venv_jax: Path | None
    checkpoints: Path
    engines: Path
    libstdcxx: list[Path]
    kalign: Path | None
    cache_dir: Path
    boltz_cache: Path | None
    af3_weights: Path | None = None        # directory holding af3.bin (default <checkpoints>/af3)
    slurm: dict[str, Any] = field(default_factory=dict)
    jax: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def venv_by_key(self, key: str) -> Path:
        if key == "venv":
            return self.venv
        if key == "venv_jax":
            if self.venv_jax is None:
                raise SiteError(f"{self.path.name}: venv_jax not set but required")
            return self.venv_jax
        raise SiteError(f"unknown venv key {key!r}")

    def jax_venv_for(self, tag: str) -> Path:
        key = (self.jax.get("venv") or {}).get(tag, "venv")
        return self.venv_by_key(key)


@dataclass
class GPU:
    compute_cap: str
    name: str
    mem_mib: int
    tag: str | None


def available_sites() -> list[str]:
    return sorted(p.stem for p in SITES_DIR.glob("*.toml") if not p.name.endswith(".example.toml"))


def _expand(value: str, ctx: dict[str, Any]) -> str:
    s = os.path.expandvars(str(value))
    for k, v in ctx.items():
        s = s.replace("{" + k + "}", str(v))
    return os.path.expanduser(s)


def resolve_site_path(name_or_path: str | None = None) -> Path:
    cand = name_or_path or os.environ.get("RESIDUDES_ENGINE_SITE_FILE") or os.environ.get("RESIDUDES_ENGINE_SITE")
    if cand:
        p = Path(cand).expanduser()
        if p.suffix == ".toml" and p.is_file():
            return p.resolve()
        p2 = SITES_DIR / f"{cand}.toml"
        if p2.is_file():
            return p2
        raise SiteError(f"site {cand!r} not found (available: {', '.join(available_sites()) or 'none'})")
    host = socket.gethostname()
    for p in sorted(SITES_DIR.glob("*.toml")):
        if p.name.endswith(".example.toml"):
            continue
        try:
            d = tomllib.loads(p.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for pat in d.get("site", {}).get("hosts", []) or []:
            if fnmatch.fnmatch(host, pat):
                return p
    local = SITES_DIR / "local.toml"
    if local.is_file():
        return local
    raise SiteError(
        f"no site matches host {host!r}. Set RESIDUDES_ENGINE_SITE=<name|path> or copy "
        f"sites/local.example.toml to sites/local.toml (available: {', '.join(available_sites()) or 'none'})"
    )


def load_site(name_or_path: str | None = None) -> Site:
    path = resolve_site_path(name_or_path)
    try:
        d = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SiteError(f"{path}: {e}") from None
    s = d.get("site", {})
    ctx = {"residudes_engine_root": RESIDUDES_ENGINE_ROOT, "home": Path.home(), "site_dir": path.parent}

    def P(key: str, default: str | None = None, required: bool = True) -> Path | None:
        env = os.environ.get(f"RESIDUDES_ENGINE_{key.upper()}")
        v = env if env else s.get(key, default)
        if v in (None, ""):
            if required:
                raise SiteError(f"{path.name}: [site].{key} is required")
            return None
        return Path(_expand(v, ctx))

    libs_env = os.environ.get("RESIDUDES_ENGINE_LIBSTDCXX")
    libs = libs_env.split(":") if libs_env else (s.get("libstdcxx") or [])
    return Site(
        name=s.get("name", path.stem),
        path=path,
        scheduler=s.get("scheduler", "none"),
        core=P("core", "{residudes_engine_root}/core"),
        venv=P("venv"),
        venv_jax=P("venv_jax", required=False),
        checkpoints=P("checkpoints"),
        engines=P("engines"),
        libstdcxx=[Path(_expand(x, ctx)) for x in libs],
        kalign=P("kalign", required=False),
        cache_dir=P("cache_dir", "{home}/.cache/residudes-engine"),
        boltz_cache=P("boltz_cache", required=False),
        af3_weights=P("af3_weights", required=False),
        slurm=d.get("slurm", {}) or {},
        jax=d.get("jax", {}) or {},
        raw=d,
    )


# ── Runtime discovery ──────────────────────────────────────────────────────

def detect_gpu() -> GPU | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    parts = [x.strip() for x in first.split(",")]
    if len(parts) < 3:
        return None
    cc, name, mem = parts[0], parts[1], parts[2]
    try:
        mem_i = int(float(mem))
    except ValueError:
        mem_i = 0
    return GPU(compute_cap=cc, name=name, mem_mib=mem_i, tag=registry.gpu_tag_for(cc))


def find_libstdcxx(candidates: list[Path]) -> Path | None:
    for c in list(candidates) + [Path(p) for p in _LIBSTDCXX_DEFAULTS]:
        try:
            if c.is_file() and _GLIBCXX_NEEDED in c.read_bytes():
                return c
        except OSError:
            continue
    return None


def site_packages(venv: Path) -> Path:
    hits = sorted((venv / "lib").glob("python3.*/site-packages"))
    if not hits:
        raise SiteError(f"no site-packages under {venv} (is the venv built? see sites/local.example.toml)")
    return hits[-1]


def venv_python(venv: Path) -> Path:
    py = venv / "bin" / "python"
    if not py.exists():
        raise SiteError(f"venv python missing: {py}")
    return py


def runtime_env(
    site: Site,
    venv: Path,
    *,
    pythonpath: list[Path],
    extra: dict[str, Any] | None = None,
    jax: bool = False,
    model_tag: str | None = None,
    family: str | None = None,
) -> dict[str, str]:
    """Environment for the inference subprocess: LD_PRELOAD / LD_LIBRARY_PATH /
    PYTHONPATH / cache dirs / backend knobs, assembled in one place."""
    env = dict(os.environ)
    sp = site_packages(venv)

    # Native libs that pip drops outside the linker's default search path.
    ld: list[Path] = [sp / "cuequivariance_ops" / "lib"]
    cu_pinned = [p for p in (sp / "nvidia" / "cu13" / "lib", sp / "nvidia" / "cu12" / "lib") if p.is_dir()]
    if cu_pinned:
        ld += cu_pinned
    elif (sp / "nvidia").is_dir():
        ld += sorted(p for p in (sp / "nvidia").glob("*/lib") if p.is_dir())
    ld.append(sp / "tensorrt_libs")
    ld = [p for p in ld if p.is_dir()]
    prev_ld = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join([str(p) for p in ld] + ([prev_ld] if prev_ld else []))

    lib = find_libstdcxx(site.libstdcxx)
    if lib is not None:
        prev = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = str(lib) + (f":{prev}" if prev else "")

    # PYTHONPATH: drop residudes-engine's own src (from the launcher), prepend the family's paths.
    own_src = str(RESIDUDES_ENGINE_ROOT / "src")
    inherited = [p for p in env.get("PYTHONPATH", "").split(":") if p and p != own_src]
    merged: list[str] = []
    for p in [str(p) for p in pythonpath] + inherited:
        if p not in merged:
            merged.append(p)
    env["PYTHONPATH"] = ":".join(merged)

    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
    env["LAYERNORM_TYPE"] = "torch_layernorm"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cache = site.cache_dir
    for sub, var in (("cuequivariance-triton", "CUEQ_TRITON_CACHE_DIR"), ("triton", "TRITON_CACHE_DIR"),
                     ("torchinductor", "TORCHINDUCTOR_CACHE_DIR")):
        d = cache / sub
        d.mkdir(parents=True, exist_ok=True)
        env.setdefault(var, str(d))

    if jax:
        jcfg = site.jax
        jc = cache / f"jax_{family or 'models'}"
        jc.mkdir(parents=True, exist_ok=True)
        env.setdefault("JAX_COMPILATION_CACHE_DIR", str(jc))
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(jcfg.get("mem_fraction", 0.65)))
        if jcfg.get("allocator"):
            env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", str(jcfg["allocator"]))
        # Site files use the RESIDUDES_ENGINE_JAX_* names; the runtime modules still read
        # the legacy BIO_TRT_JAX_* names, so every knob is exported under both.
        def put(k: str, v: Any, force: bool) -> None:
            names = [k]
            if k.startswith("RESIDUDES_ENGINE_JAX_"):
                names.append("BIO_TRT_JAX_" + k[len("RESIDUDES_ENGINE_JAX_"):])
            elif k.startswith("BIO_TRT_JAX_"):
                names.append("RESIDUDES_ENGINE_JAX_" + k[len("BIO_TRT_JAX_"):])
            for n in names:
                if force or n not in env:
                    env[n] = str(v)

        for k, v in (jcfg.get("env") or {}).items():
            if isinstance(v, dict):
                continue
            put(k, v, force=False)
        if model_tag:
            for k, v in ((jcfg.get("env") or {}).get(model_tag) or {}).items():
                put(k, v, force=True)
        # A user-set RESIDUDES_ENGINE_JAX_* in the shell wins and is mirrored too.
        for k in list(env):
            if k.startswith("RESIDUDES_ENGINE_JAX_"):
                env.setdefault("BIO_TRT_JAX_" + k[len("RESIDUDES_ENGINE_JAX_"):], env[k])

    if extra:
        for k, v in extra.items():
            if v is not None:
                env[k] = str(v)
    return env
