"""Stable import path for the model-agnostic I/O helpers.

v0 loads the implementations under ``core/scripts/`` (``cif_normalize.py``,
``logit_codec.py``, ``inputs_meta.py``) by file. Phase 0 of the port moves
the code here; callers keep ``residudes_engine.schema.<name>()``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_VENDORED = {"cif": "cif_normalize", "logits": "logit_codec", "provenance": "inputs_meta"}
_cache: dict[str, object] = {}


def _scripts_dir(core: Path | None) -> Path:
    if core is None:
        from ..config import load_site
        core = load_site().core
    return Path(core) / "scripts"


def load(name: str, core: Path | None = None):
    """Return the module behind schema name ``cif`` | ``logits`` | ``provenance``."""
    if name in _cache:
        return _cache[name]
    fname = _VENDORED[name]
    path = _scripts_dir(core) / f"{fname}.py"
    spec = importlib.util.spec_from_file_location(f"residudes_engine.schema._vendored_{fname}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _cache[name] = mod
    return mod


def cif(core: Path | None = None):
    return load("cif", core)


def logits(core: Path | None = None):
    return load("logits", core)


def provenance(core: Path | None = None):
    return load("provenance", core)
