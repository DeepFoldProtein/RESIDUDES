#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Boltz-2 inference with JAX hot-swap.

Replaces the structure pairformer (and confidence pairformer when present)
with Haiku/JAX modules sharing weights from the same checkpoint. The
diffusion token transformer is NOT swapped (Boltz's per-block call signature
differs from AF3 / Protenix and the Haiku port for it is incomplete in
``lantern_engine_jax/boltz.py``).

The unified output layout (``seed_<S>/sample_<i>/{model.cif, summary.json,
meta.json, arrays/}``) is produced post-hoc by ``run_bt2_jax.sbatch`` from
Boltz's native flat outputs (``<name>_model_<i>.cif`` + side-car npz/json).
We don't import ``scripts/boltz_inference.py`` here because its module-load
side effects (eager ``boltz.model.modules.confidencev2`` import, etc.) make
``torch.cuda.device_count()`` fail under Slurm ``--gres=gpu:1`` cgroups.

Usage::

    python scripts/boltz_inference_jax.py predict examples/boltz2/ubiquitin.yaml \\
        --out_dir results/boltz2_jax_ubiquitin \\
        --use_msa_server  # or pass an MSA file via the boltz CLI

JAX-specific flags (consumed before boltz's parser):
    --no_jax           Run vanilla PyTorch (skip the swap; for baselines).
    --jax_dtype fp32|bf16   Engine dtype (default bf16).
    --jax_cache_dir PATH    Persistent XLA compilation cache directory.

All other flags are forwarded to boltz's CLI.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("boltz-jax")

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Scrub _REPO_ROOT from sys.path so `from boltz.model...` resolves to the
# pip-installed boltz package, not our local `boltz/` adapter.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _REPO_ROOT]
sys.path.insert(0, str(_REPO_ROOT / "thirdparty"))

# ── Pre-parse JAX-only flags ──────────────────────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no_jax", action="store_true")
_pre.add_argument("--jax_dtype", choices=["fp32", "bf16"], default="bf16")
_pre.add_argument("--jax_cache_dir", default=None)
# Hybrid: keep the JAX pairformers and run the diffusion token transformer
# (24 blocks × ~200 sampling steps, PyTorch otherwise) on its TensorRT engine.
_pre.add_argument("--engines_dir", default=None,
                  help="TRT engine set; only its token_transformer_engine is used (BT2_JAX_TRT_TOKEN_TRANSFORMER=0 disables)")
_jax_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# ── JAX cache directory (via env vars only — defer `import jax` itself) ───
# JAX-CUDA initialization races with PyTorch-CUDA init in Lightning's
# `_AcceleratorConnector` (NVML init failure → "No supported gpu backend
# found"). We set the cache dir via env-only here so JAX picks it up later
# *if* it's used; the actual `import jax` is deferred to _inject_jax(),
# which fires after Boltz has fully set up its torch model.
_cache_dir = _jax_args.jax_cache_dir or os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser("~/.cache/jax_boltz"),
)
os.makedirs(_cache_dir, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _cache_dir
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

# ── AF3 input.json → bt2 yaml passthrough (parity with scripts/boltz_inference.py) ─
def _maybe_convert_af3_predict_arg(argv):
    import json as _json, tempfile, importlib.util as _ilu
    for i, a in enumerate(argv):
        if a == "predict" and i + 1 < len(argv):
            arg = argv[i + 1]
            if not os.path.exists(arg):
                return
            spec = _ilu.spec_from_file_location(
                "biotrt_af3_adapter",
                str(_REPO_ROOT / "scripts" / "af3_input_adapter.py"),
            )
            adapter = _ilu.module_from_spec(spec)
            spec.loader.exec_module(adapter)

            def _is_af3(p):
                if not p.endswith(".json") or not os.path.isfile(p):
                    return False
                try:
                    with open(p) as f:
                        d = _json.load(f)
                    return isinstance(d, dict) and d.get("dialect") == "alphafold3"
                except (OSError, _json.JSONDecodeError):
                    return False

            def _convert_one(json_path: str, out_dir: Path) -> Path | None:
                with open(json_path) as f:
                    data = _json.load(f)
                yaml_str = adapter.af3_to_bt2_yaml(data, sidecar_dir=out_dir)
                name = data.get("name") or Path(json_path).stem
                out = out_dir / f"{name}.yaml"
                out.write_text(yaml_str)
                return out

            if os.path.isdir(arg):
                jsons = sorted(p for p in Path(arg).iterdir() if _is_af3(str(p)))
                if not jsons:
                    return
                tmp_dir = Path(tempfile.mkdtemp(prefix="biotrt_bt2_jax_af3_"))
                for jp in jsons:
                    _convert_one(str(jp), tmp_dir)
                logger.info(
                    "AF3 dialect dir → bt2 yamls: %s (%d files)", tmp_dir, len(jsons))
                argv[i + 1] = str(tmp_dir)
            elif _is_af3(arg):
                tmp_dir = Path(tempfile.mkdtemp(prefix="biotrt_bt2_jax_af3_"))
                out = _convert_one(arg, tmp_dir)
                logger.info("AF3 dialect → bt2 yaml: %s", out)
                argv[i + 1] = str(out)
            return

_maybe_convert_af3_predict_arg(sys.argv)


# When given a directory, the temp dir contains alignment-override JSONs and
# MSA CSV sidecars boltz.main.check_inputs would reject. Filter to yaml/fasta.
def _install_check_inputs_filter():
    try:
        import boltz.main as _bmain  # type: ignore
    except Exception:
        return
    _orig = _bmain.check_inputs
    def _filtered(data):
        from pathlib import Path as _P
        d = _P(data)
        if d.is_dir():
            files = []
            for ext in ("*.yaml", "*.yml", "*.fasta", "*.fa", "*.fas"):
                files.extend(sorted(d.glob(ext)))
            return files
        return _orig(d)
    _bmain.check_inputs = _filtered

_install_check_inputs_filter()


# ── ContactConditioning amplification (BIO_TRT_CONTACT_AMP / NEG_PAIRS) ───
# Two knobs on the ContactConditioning embedding (the +z_init bias the model
# adds based on user contact constraints):
#
#   BIO_TRT_CONTACT_AMP=N (default 1.0)
#     Scales the SPECIFIED contact signal added to z_init. UNSPECIFIED and
#     UNSELECTED regions stay at the trained baseline.
#
#   BIO_TRT_CONTACT_NEG_PAIRS="<a1>,<r1>,<a2>,<r2>,<thr>;..."
#     Per-pair "anti-contact" channel. (a, r) are asym_id (int, 0-based:
#     chain A=0, B=1, ...) and residue number (int, 1-based to match
#     input.json). For each entry, we synthesize a CONTACT class at that
#     (i, j) token pair and SUBTRACT BIO_TRT_CONTACT_NEG_AMP * encoder(...)
#     from z_init — flipping the attractive bias into a repulsive one.
#     The pair is NOT registered in the input JSON / featurizer, so the
#     ContactPotential layer never sees it (no attractive guidance fight).
#
#   BIO_TRT_CONTACT_NEG_AMP=N (default -1.0)
#     Sign-flip magnitude. -1.0 = cancel the attractive signal; -2.0 = same
#     magnitude repulsion as a positive contact; <-2.0 = stronger push.
def _install_contact_amp():
    import torch

    try:
        amp_pos = float(os.environ.get("BIO_TRT_CONTACT_AMP", "1.0"))
    except ValueError:
        logger.warning("[bt2_jax] BIO_TRT_CONTACT_AMP not a float; disabling amp")
        return
    try:
        neg_amp = float(os.environ.get("BIO_TRT_CONTACT_NEG_AMP", "-1.0"))
    except ValueError:
        logger.warning("[bt2_jax] BIO_TRT_CONTACT_NEG_AMP not a float; ignoring")
        neg_amp = -1.0

    neg_pairs_str = os.environ.get("BIO_TRT_CONTACT_NEG_PAIRS", "").strip()
    neg_pairs: list[tuple[int, int, int, int, float]] = []
    # Entry separator can be `;` (intuitive) or `:` (safe through shell).
    # Inside an entry, fields are comma-separated.
    for entry in neg_pairs_str.replace(":", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 5:
            print(
                f"[bt2_jax] BIO_TRT_CONTACT_NEG_PAIRS malformed entry {entry!r} "
                f"(want a1,r1,a2,r2,thr)",
                flush=True,
            )
            continue
        try:
            a1, r1, a2, r2 = (int(p) for p in parts[:4])
            thr = float(parts[4])
        except ValueError:
            print(
                f"[bt2_jax] BIO_TRT_CONTACT_NEG_PAIRS parse error in {entry!r}",
                flush=True,
            )
            continue
        # User passes 1-based residue numbers (matches input.json); featurizer
        # stores 0-based res_idx (schema.py token_spec_to_ids does `- 1`).
        neg_pairs.append((a1, r1 - 1, a2, r2 - 1, thr))

    if amp_pos == 1.0 and not neg_pairs:
        return

    try:
        from boltz.model.modules.trunkv2 import ContactConditioning  # type: ignore
        from boltz.data import const  # type: ignore
    except Exception as _e:
        logger.warning(
            f"[bt2_jax] contact amp install failed: {type(_e).__name__}: {_e}"
        )
        return

    contact_class_idx = const.contact_conditioning_info["CONTACT"]
    n_classes = len(const.contact_conditioning_info)

    _orig_forward = ContactConditioning.forward

    def _amped_forward(self, feats):
        out = _orig_forward(self, feats)
        cc = feats["contact_conditioning"]  # (B, N, N, 5)
        bg = (
            self.encoding_unspecified * cc[..., 0:1]
            + self.encoding_unselected * cc[..., 1:2]
        )
        out_amp = bg + amp_pos * (out - bg)

        if not neg_pairs:
            return out_amp

        # Resolve token indices for each (asym_id, res_idx) anti-contact pair.
        asym_id = feats["asym_id"][0]  # (N,)
        res_idx = feats["residue_index"][0]  # (N,)
        N = cc.shape[1]
        device = cc.device

        neg_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
        cc_neg = cc.clone()
        thr_neg = feats["contact_threshold"].clone()

        # One-hot CONTACT row used to overwrite cc_neg at neg positions.
        contact_onehot = torch.zeros(n_classes, dtype=cc.dtype, device=device)
        contact_onehot[contact_class_idx] = 1
        # Match cc's dtype (long after featurizer one_hot, but cast to bf16
        # inside pairformer; we just write into the existing buffer).
        contact_onehot = contact_onehot.to(cc.dtype)

        for (a1, r1, a2, r2, thr) in neg_pairs:
            t1 = torch.nonzero(
                (asym_id == a1) & (res_idx == r1), as_tuple=False
            ).flatten()
            t2 = torch.nonzero(
                (asym_id == a2) & (res_idx == r2), as_tuple=False
            ).flatten()
            if t1.numel() == 0 or t2.numel() == 0:
                print(
                    f"[bt2_jax] NEG_PAIR (a={a1},r={r1+1})-(a={a2},r={r2+1}) "
                    f"unresolved (t1={t1.numel()}, t2={t2.numel()}); skipping",
                    flush=True,
                )
                continue
            ii, jj = torch.meshgrid(t1, t2, indexing="ij")
            ii, jj = ii.flatten(), jj.flatten()
            for i, j in zip(ii.tolist(), jj.tolist()):
                cc_neg[0, i, j] = contact_onehot
                cc_neg[0, j, i] = contact_onehot
                thr_neg[0, i, j] = thr
                thr_neg[0, j, i] = thr
                neg_mask[i, j] = True
                neg_mask[j, i] = True

        if not neg_mask.any():
            return out_amp

        feats_neg = dict(feats)
        feats_neg["contact_conditioning"] = cc_neg
        feats_neg["contact_threshold"] = thr_neg
        out_with_neg = _orig_forward(self, feats_neg)
        # (out_with_neg - out) at neg positions = pure encoder contribution
        # from the synthetic CONTACT one-hot (zero elsewhere because cc is
        # otherwise identical).
        delta = out_with_neg - out  # (B, N, N, token_z)
        gate = neg_mask.to(delta.dtype).unsqueeze(0).unsqueeze(-1)  # (1,N,N,1)
        return out_amp + neg_amp * delta * gate

    ContactConditioning.forward = _amped_forward
    logger.info(
        "ContactConditioning amplification: pos=x%.2f neg=x%.2f neg_pairs=%d",
        amp_pos,
        neg_amp,
        len(neg_pairs),
    )
    print(
        f"[bt2_jax] ContactConditioning amplification: "
        f"pos=x{amp_pos:.2f} neg=x{neg_amp:.2f} neg_pairs={len(neg_pairs)}",
        flush=True,
    )


_install_contact_amp()


# ── Inject JAX into Boltz2 model load ─────────────────────────────────────
if not _jax_args.no_jax:
    from boltz.model.models.boltz2 import Boltz2  # type: ignore

    def _inject_jax(model, ckpt_path: str) -> None:
        """Replace pairformer_module (+ confidence pairformer) with JAX wrappers.

        The score_model.token_transformer swap is OFF here — Boltz's per-block
        ``bias`` kwarg differs from AF3 / Protenix and the Haiku port in
        ``lantern_engine_jax/boltz.py`` is incomplete (causes a downstream shape
        mismatch in atom_attention_decoder). Only structure + confidence
        pairformers are swapped.

        JAX is imported lazily here so its CUDA init can't race with
        Lightning's ``_AcceleratorConnector`` (which queries
        ``torch.cuda.device_count()`` via NVML and fails when JAX has
        already grabbed the device).
        """
        # Lazy: keep _REPO_ROOT off sys.path until just before lantern_engine_jax
        # imports — we scrubbed it earlier so the pip `boltz` resolves.
        sys.path.insert(0, str(_REPO_ROOT))
        try:
            import jax  # noqa: F401  # first-time CUDA grab happens here
            jax.config.update("jax_compilation_cache_dir", _cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
            import jax.numpy as jnp
            from lantern_engine_jax.boltz import BoltzJAX, BoltzJAXConfig
        finally:
            try:
                sys.path.remove(str(_REPO_ROOT))
            except ValueError:
                pass

        cfg = BoltzJAXConfig.from_checkpoint(ckpt_path)
        cfg.dtype = jnp.bfloat16 if _jax_args.jax_dtype == "bf16" else jnp.float32
        # Default OFF: per-block bias kwarg differs from AF3/Protenix and may
        # cause shape mismatch in atom_attention_decoder. Toggle ON via env
        # to test on large complexes where the extra memory savings matter.
        cfg.swap_token_transformer = (
            os.environ.get("BIO_TRT_JAX_SWAP_TOKEN_TRANSFORMER", "0") == "1"
        )
        # Optional disables for debugging.
        cfg.swap_template_pairformer = (
            os.environ.get("BIO_TRT_JAX_SWAP_TEMPLATE", "1") == "1"
        )
        cfg.swap_msa_pairformer = (
            os.environ.get("BIO_TRT_JAX_SWAP_MSA", "1") == "1"
        )
        logger.info(
            "JAX hot-swap: dtype=%s structure=%d confidence=%d template=%s msa=%s token_transformer=%s",
            _jax_args.jax_dtype,
            cfg.structure_pairformer_num_layers,
            cfg.confidence_pairformer_num_layers,
            f"{cfg.template_pairformer_num_layers}L" if cfg.swap_template_pairformer else "PT",
            f"{cfg.msa_num_layers}L" if cfg.swap_msa_pairformer else "PT",
            "JAX" if cfg.swap_token_transformer else "PT",
        )
        wrapper = BoltzJAX(model, cfg, ckpt_path=ckpt_path)
        wrapper.optimize()

    def _inject_trt_token_transformer(model, engines_dir: str) -> bool:
        """Swap score_model.token_transformer for its TRT engine (pairformers stay JAX).

        Same adapter as the TRT backend (boltz/compat.py::CompatTokenTransformerTRT);
        the engine takes Boltz's precomputed per-block pair bias, so it is
        independent of how the trunk was computed."""
        import importlib.util
        from pathlib import Path as _P
        tt_dir = _P(engines_dir) / "token_transformer_engine"
        if not (tt_dir / "trt" / "rank0.engine").is_file():
            logger.warning("no token_transformer_engine under %s; diffusion stays in PyTorch", engines_dir)
            return False
        spec = importlib.util.spec_from_file_location("_biotrt_boltz_compat", str(_REPO_ROOT / "boltz" / "compat.py"))
        compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat)
        from tensorrt_bionemo.models.helper import AcceleratedConfig
        from tensorrt_bionemo.runtime.allocator import OnDemandContextMemoryManager
        configs = {"token_transformer": AcceleratedConfig(checkpoint=str(tt_dir), backend="trt")}
        compat.BoltzTRT(model).optimize(configs, context_memory_allocator=OnDemandContextMemoryManager())
        logger.info("TRT token transformer injected from %s (hybrid JAX trunk + TRT diffusion)", tt_dir)
        print(f"[bt2_jax] hybrid: TRT token transformer from {tt_dir}", flush=True)
        return True

    # Stack our JAX patch on top of whatever boltz_inference already
    # installed (only the TRT patch, and only if --engines_dir was set).
    _orig_load = Boltz2.load_from_checkpoint

    def _patched_load(cls, *args, **kwargs):
        model = _orig_load(*args, **kwargs)
        ckpt_path = args[0] if args else kwargs.get("checkpoint_path")
        if ckpt_path is None:
            raise RuntimeError("JAX hot-swap requires checkpoint_path argument.")
        logger.info("Injecting JAX from %s", ckpt_path)
        _inject_jax(model, str(ckpt_path))
        if _jax_args.engines_dir and os.environ.get("BT2_JAX_TRT_TOKEN_TRANSFORMER", "1") == "1":
            _inject_trt_token_transformer(model, _jax_args.engines_dir)
        return model

    Boltz2.load_from_checkpoint = classmethod(_patched_load)
    logger.info("Compilation cache: %s", _cache_dir)


# ── Unified output layout + multi-seed loop ───────────────────────────────
# Installs the BoltzWriter monkey-patch (seed_<S>/sample_<i>/{model.cif,
# summary.json, meta.json, arrays/}) and the Trainer.predict multi-seed
# wrapper (loops BT2_SEEDS inside one model load). Implementations live in
# scripts/boltz_unified_jax.py so this file stays focused on JAX hot-swap.
# Both installs are idempotent / no-op when their precondition (BoltzWriter
# present / BT2_SEEDS unset) isn't met.
try:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from boltz_unified_jax import (  # type: ignore
        install_af3_alignment_override,
        install_unified_writer,
        install_multiseed_predict,
        install_constraint_logger,
        install_runtime_feats_logger,
        install_debug_oom,
        install_template_chunking,
        install_stage_profiler,
        install_torch_compile,
    )
    install_torch_compile()    # BT2_TORCH_COMPILE=… (no-op otherwise); runs before the JAX swap patch stacks on load
    install_af3_alignment_override()
    install_constraint_logger()
    install_runtime_feats_logger()
    install_unified_writer()
    install_stage_profiler()   # LANTERN_ENGINE_PROFILE=1 → per-stage timing JSON (no-op otherwise)
    install_multiseed_predict()
    # Chunk TemplateV2Module over the template axis so the per-template pair
    # stack isn't materialized all at once (the large-N OOM root cause).
    install_template_chunking()
    # Last: when BOLTZ_DEBUG_OOM=1, override predict_step with a re-raising
    # reimplementation so the real OOM (torch or JAX) surfaces. No-op otherwise.
    install_debug_oom()
except Exception as _e:
    logger.warning(f"[bt2_jax] unified-writer install failed: {type(_e).__name__}: {_e}")


# ── Defer to boltz CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    from boltz.main import cli  # type: ignore
    cli()
