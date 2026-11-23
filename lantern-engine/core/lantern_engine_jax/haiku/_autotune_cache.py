# SPDX-License-Identifier: Apache-2.0
"""Persist the in-process tokamax autotune cache across runs.

Tokamax's `_AUTOTUNING_CACHE` is a module-level dict keyed by
`(Op, device_kind, abstract_arg_key)` -> `AutotuningData`. The packaged
JSON files only cover shapes DeepMind benchmarked; on user shapes the
runtime triggers an autotune cache miss + microbenchmark on first use.

For the Bio-TRT Protenix v2 pipeline, the warm pairformer hits ~288
unique call signatures per inference. Each microbenchmark costs
50-100 ms, so warm runs spend ~10-15 s re-tuning identical shapes
across processes. Pickling `_AUTOTUNING_CACHE` after the first run and
restoring it on subsequent runs eliminates this overhead.

Activate via env var `BIO_TRT_JAX_AUTOTUNE_CACHE_DIR` (default
`~/.cache/jax_protenix/tokamax_autotune`). Files are pickled per-device
to keep cross-GPU runs isolated.
"""

from __future__ import annotations

import atexit
import logging
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("lantern_engine_jax.autotune")


def _info(msg: str) -> None:
    """Log to logger AND stderr — Protenix's root logger filters us."""
    logger.info(msg)
    print(f"[lantern_engine_jax.autotune] {msg}", file=sys.stderr, flush=True)


def _cache_dir() -> Path:
    p = os.environ.get(
        "BIO_TRT_JAX_AUTOTUNE_CACHE_DIR",
        os.path.expanduser("~/.cache/jax_protenix/tokamax_autotune"),
    )
    return Path(p)


def _device_id() -> str:
    """Stable filesystem-safe device identifier."""
    try:
        import jax
        from jax.extend import backend
        kind = backend.get_default_device().device_kind
    except Exception:
        kind = "unknown"
    return kind.replace(" ", "_").replace("/", "_")


def _cache_path() -> Path:
    return _cache_dir() / f"{_device_id()}.pkl"


_loaded = False
_loaded_lock = threading.Lock()
_initial_size = 0


def _import_tokamax_op_module():
    """Return tokamax._src.ops.op or None if unavailable."""
    try:
        from tokamax._src.ops import op as op_mod  # type: ignore
        return op_mod
    except Exception:
        return None


def _force_autotune_fallback() -> None:
    """Switch tokamax's cache-miss fallback from heuristics → autotune.

    With `heuristics` (the default) tokamax never benchmarks at runtime,
    so `_AUTOTUNING_CACHE` stays empty and there is nothing to persist.
    With `autotune` the first call on a new shape runs a microbenchmark
    over multiple kernel tile configs and picks the fastest — and that
    result lands in the in-memory cache.

    Set BIO_TRT_JAX_AUTOTUNE_FALLBACK=heuristics to opt out.
    """
    # Default 'heuristics' — real autotune microbenchmarks each shape over
    # multiple kernel configs (~20-30s/shape on a 4090), and with ~288 unique
    # call sites that's >2 hours of cold time for any non-trivial pipeline.
    # Set BIO_TRT_JAX_AUTOTUNE_FALLBACK=autotune to opt in for offline
    # population (recommended only if you'll cache the result).
    fb = os.environ.get("BIO_TRT_JAX_AUTOTUNE_FALLBACK", "heuristics")
    try:
        from tokamax._src import config as _tk_config  # type: ignore
        # _ConfigOption only exposes a ``__call__(value)`` context manager;
        # we want a permanent override, so write straight to the underlying
        # threading.local state used by `value`.
        from tokamax._src.config import _STATE  # type: ignore
        flag = _tk_config.autotuning_cache_miss_fallback.flag
        # Initial parse (with empty argv) so flag.parser is wired.
        from absl import flags as _absl_flags  # type: ignore
        if not _absl_flags.FLAGS.is_parsed():
            _absl_flags.FLAGS([__file__])
        # Validate + assign.
        flagval = _absl_flags.FLAGS[flag.name]
        flagval.value = fb
        setattr(_STATE, flag.name, fb)
        # Only emit if user opted into the slow autotune path, since
        # the heuristics default is a no-op message.
        if fb != "heuristics":
            _info(f"cache-miss fallback = {fb}")
    except Exception as e:
        _info(f"cache-miss fallback override failed: {e}")


def install() -> bool:
    """Load saved autotune cache (if any) and arm save-on-exit.

    Idempotent. Returns True if a saved cache was loaded.
    """
    global _loaded, _initial_size
    with _loaded_lock:
        if _loaded:
            return False
        _loaded = True

    _force_autotune_fallback()

    op_mod = _import_tokamax_op_module()
    if op_mod is None:
        return False

    path = _cache_path()
    loaded = False
    if path.exists() and path.stat().st_size > 0:
        try:
            with open(path, "rb") as f:
                saved: dict[Any, dict[str, dict[Any, Any]]] = pickle.load(f)
        except Exception as e:
            logger.warning("Autotune cache: failed to load %s (%s)", path, e)
            saved = {}
        # Merge into the live `_AUTOTUNING_CACHE`, treating saved data as
        # authoritative for any (op, device, key) triple.
        live = op_mod._AUTOTUNING_CACHE
        for op, by_dev in saved.items():
            existing = live.get(op)
            if existing is None:
                # Use a real AutotuningCache so that the lazy-load
                # `__missing__` path still fires for untouched device kinds.
                from tokamax._src.autotuning import cache as cache_lib  # type: ignore
                existing = cache_lib.AutotuningCache(op)
                live[op] = existing
            for dev, kvmap in by_dev.items():
                bucket = existing.get(dev)
                if bucket is None:
                    existing[dev] = dict(kvmap)
                else:
                    bucket.update(kvmap)
        n = sum(len(d) for od in live.values() for d in od.values())
        _initial_size = n
        _info(f"loaded {n} entries from {path}")
        loaded = True

    atexit.register(_save_if_changed)
    return loaded


def save_now() -> bool:
    """Save the live `_AUTOTUNING_CACHE` to disk; return True if anything written."""
    op_mod = _import_tokamax_op_module()
    if op_mod is None:
        _info("save_now: tokamax not importable, skipping")
        return False
    live = op_mod._AUTOTUNING_CACHE
    n = sum(len(d) for od in live.values() for d in od.values())
    if n <= _initial_size:
        # Quiet path: with the default `heuristics` fallback, nothing ever
        # lands in `_AUTOTUNING_CACHE`, so this is the common case.
        return False
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {op: {d: dict(kv) for d, kv in by_dev.items()}
                for op, by_dev in live.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        _info(f"saved {n} entries (added {n - _initial_size}) to {path}")
        return True
    except Exception as e:
        _info(f"save failed: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _save_if_changed() -> None:
    save_now()
