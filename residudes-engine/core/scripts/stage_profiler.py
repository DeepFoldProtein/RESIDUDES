# SPDX-License-Identifier: Apache-2.0
"""Per-stage wall time / peak memory of a model forward, for any torch model.

    from stage_profiler import install_class_profiler
    install_class_profiler(OpenFold3, ("input_embedder", "template_embedder", "msa_module",
                                       "pairformer_stack", "sample_diffusion", "aux_heads"), label="of3")

Enabled by RESIDUDES_ENGINE_PROFILE=1 (no-op otherwise). On the first call of
``model_cls.forward`` the named submodules get synchronising forward hooks;
each top-level forward is timed as one "step". Totals go to
$RESIDUDES_ENGINE_PROFILE_OUT (default ./profile.json) and are printed as a table on
stdout+stderr. Hot-swapped modules (TRT / JAX wrappers) are nn.Modules whose
forward returns torch tensors, so the synchronise after each call covers them.
The Boltz-2 variant lives in boltz_unified_jax.install_stage_profiler (it has
to hook Lightning's predict_step and a non-module ``sample`` method)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def install_class_profiler(model_cls, stage_attrs: tuple[str, ...], *, label: str,
                           method: str = "forward") -> bool:
    if os.environ.get("RESIDUDES_ENGINE_PROFILE", "0") != "1":
        return False
    import torch

    out_path = os.environ.get("RESIDUDES_ENGINE_PROFILE_OUT") or "profile.json"
    state = {"installed": False, "stages": {}, "steps": []}

    def _sync():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    def _acc(name, dt):
        st = state["stages"].setdefault(name, {"calls": 0, "time_s": 0.0, "peak_alloc_gb": 0.0})
        st["calls"] += 1
        st["time_s"] += dt
        try:
            st["peak_alloc_gb"] = max(st["peak_alloc_gb"], torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass

    def _pre(name):
        def hook(mod, args):
            _sync()
            mod.__dict__["_residudes_t0"] = time.perf_counter()
        return hook

    def _post(name):
        def hook(mod, args, output):
            _sync()
            t0 = mod.__dict__.pop("_residudes_t0", None)
            if t0 is not None:
                _acc(name, time.perf_counter() - t0)
        return hook

    def _install_on(model):
        for name in stage_attrs:
            mod = getattr(model, name, None)
            if isinstance(mod, torch.nn.Module):
                mod.register_forward_pre_hook(_pre(name))
                mod.register_forward_hook(_post(name))
        state["installed"] = True

    def _dump():
        stages = state["stages"]
        total = sum(s["time_s"] for s in stages.values())
        step_total = sum(state["steps"]) if state["steps"] else None
        rec = {"model": label, "stages": stages, "stages_total_s": total, "steps": state["steps"],
               "unattributed_s": (step_total - total) if step_total is not None else None,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(rec, indent=2) + "\n")
        except Exception as e:  # pragma: no cover
            print(f"[{label} profile] could not write {out_path}: {e}", file=sys.stderr)
        lines = [f"[{label} profile] stage                 calls    time_s   share  peak_alloc_gb"]
        for name, s in sorted(stages.items(), key=lambda kv: -kv[1]["time_s"]):
            share = (s["time_s"] / total * 100) if total else 0.0
            lines.append(f"[{label} profile] {name:<20s} {s['calls']:5d} {s['time_s']:9.2f} {share:6.1f}%  {s['peak_alloc_gb']:7.2f}")
        if step_total is not None:
            lines.append(f"[{label} profile] forward total {step_total:.2f}s; unattributed {step_total - total:.2f}s")
        msg = "\n".join(lines) + "\n"
        sys.stdout.write(msg); sys.stdout.flush()
        sys.stderr.write(msg); sys.stderr.flush()

    orig = getattr(model_cls, method)

    def profiled(self, *a, **kw):
        if not state["installed"]:
            _install_on(self)
        _sync()
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **kw)
        finally:
            _sync()
            state["steps"].append(time.perf_counter() - t0)
            _dump()

    setattr(model_cls, method, profiled)
    print(f"[{label} profile] RESIDUDES_ENGINE_PROFILE=1: stage hooks armed on {model_cls.__name__}.{method}; output → {out_path}",
          flush=True)
    return True
