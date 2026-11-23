"""Unified output writer + multi-seed loop for the bt2_jax path.

Goal: produce the cross-model layout (seed_<S>/sample_<i>/{model.cif,
summary.json, meta.json, arrays/}) that boltz_inference.py emits on the
torch path, without inheriting the torch-only bookkeeping (MSA hooks,
head-logit capture, csv-dedup bypass, paired-MSA accounting) which
require boltz_inference.py's eager imports that break Lightning's GPU
init under cgroups (see boltz_inference_jax.py docstring).

What this module installs:
  * AF3 alignment override (same as torch path; pure schema patch with no
    eager torch imports).
  * BoltzWriter monkey-patch: writes per-target/per-seed/per-sample
    artefacts and a seed-level timing.json (watcher's completion marker).
  * pl.Trainer.predict multi-seed wrapper: loops BT2_SEEDS so a single
    model load handles N seeds; each pass updates BT2_SEED env and the
    writer reads it freshly.

Layout (parity with boltz_inference.py's torch unified writer minus the
torch-only MSA/logit dumps):
  <output_dir>/<record.id>/seed_<S>/
      distogram_logits.npz                       (if pdistogram in pred)
      timing.json                                (completion marker)
      sample_<rank>/
          model.cif
          summary.json
          meta.json
          arrays/
              plddt_atom.npy
              plddt_token.npy
              pae.npy / pde.npy                  (raw expected-value)
              contact_probs.npy                  (from distogram, AF3 rule)
"""
from __future__ import annotations

import json
import logging
import os
import string
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

logger = logging.getLogger("boltz-unified-jax")


# ── Helpers ──────────────────────────────────────────────────────────────

def _chain_letter(idx: int) -> str:
    if idx < 26:
        return string.ascii_uppercase[idx]
    return string.ascii_uppercase[(idx // 26) - 1] + string.ascii_uppercase[idx % 26]


def _save_npy(p: Path, arr) -> None:
    if arr is None:
        return
    import torch as _torch
    if isinstance(arr, _torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype.kind == "f" and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def _save_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2,
                            default=lambda x: float(x) if hasattr(x, "item") else None))


def _f(x):
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        return None


def _compute_gpde(pde, contact_probs):
    """AF3 SI 5.7 Eq.16 — Σ pde·cp / Σ cp."""
    import torch as _torch
    if pde is None or contact_probs is None:
        return None
    if isinstance(pde, _torch.Tensor):
        pde = pde.detach().cpu().float().numpy()
    if isinstance(contact_probs, _torch.Tensor):
        contact_probs = contact_probs.detach().cpu().float().numpy()
    pde = np.asarray(pde, dtype=np.float32)
    contact_probs = np.asarray(contact_probs, dtype=np.float32)
    if pde.shape != contact_probs.shape or pde.ndim < 2:
        return None
    s = float(contact_probs.sum())
    if s <= 0:
        return None
    return float((pde * contact_probs).sum() / s)


def _round3(x):
    if isinstance(x, dict):  return {k: _round3(v) for k, v in x.items()}
    if isinstance(x, list):  return [_round3(v) for v in x]
    if isinstance(x, bool):  return x
    if isinstance(x, float): return round(x, 3)
    return x


# ── Contact / pocket constraint logger ──────────────────────────────────

def install_constraint_logger() -> None:
    """Monkey-patch boltz's schema parser to log parsed constraints once.

    boltz parses contact_constraints into Record.inference_options but never
    logs them, so a misconfigured YAML (wrong chain id, off-by-one residue)
    is silent — the model just sees a different constraint than intended.
    This wrapper prints to stderr right after parsing so the constraints
    show up in inference.log.

    Format: "[bt2 constraints] N contacts: (chainA-resA <-> chainB-resB, max=D.D, force=T/F) ..."
    """
    try:
        from boltz.data.parse import schema as _bschema  # type: ignore
    except Exception as e:
        logger.warning(f"[bt2 constraint-log] schema import failed: {e}")
        return

    _orig_parse = _bschema.parse_boltz_schema

    def _wrap(*args, **kwargs):
        target = _orig_parse(*args, **kwargs)
        try:
            record = getattr(target, "record", None)
            chains = getattr(record, "chains", None) if record is not None else None
            opts = getattr(record, "inference_options", None) if record is not None else None
            if opts is None:
                return target

            # Build chain_idx → chain_name lookup. Record.chains is a list of
            # ChainInfo with .chain_name (single letter post-boltz parse).
            def _chain_name(i: int) -> str:
                try:
                    return chains[i].chain_name
                except Exception:
                    return f"<chain{i}>"

            cc = getattr(opts, "contact_constraints", None) or []
            pc = getattr(opts, "pocket_constraints", None) or []
            sys.stderr.write(
                f"[bt2 constraints] {len(cc)} contact, {len(pc)} pocket "
                f"parsed for {getattr(target, 'name', getattr(record, 'id', '?'))}\n"
            )
            for i, item in enumerate(cc):
                # (token1, token2, max_distance, force) where each token is
                # (chain_idx, residue_idx_0based) or atom-level tuple.
                try:
                    t1, t2, max_d, force = item
                    c1, r1 = t1[0], t1[1]
                    c2, r2 = t2[0], t2[1]
                    sys.stderr.write(
                        f"  [{i:2d}] {_chain_name(c1)}-{int(r1)+1} <-> "
                        f"{_chain_name(c2)}-{int(r2)+1}  "
                        f"max={float(max_d):.1f}Å  force={bool(force)}\n"
                    )
                except Exception as e:
                    sys.stderr.write(f"  [{i:2d}] (unparsable: {e}) raw={item!r}\n")
            for i, item in enumerate(pc):
                try:
                    binder, contacts, max_d, force = item
                    sys.stderr.write(
                        f"  pocket[{i}] binder={_chain_name(binder)} "
                        f"n_contacts={len(contacts)} max={float(max_d):.1f}Å "
                        f"force={bool(force)}\n"
                    )
                except Exception as e:
                    sys.stderr.write(f"  pocket[{i}] (unparsable: {e}) raw={item!r}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[bt2 constraint-log] dump failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
        return target

    _bschema.parse_boltz_schema = _wrap
    # Also patch downstream modules that did `from ... import parse_boltz_schema`
    # at import time — they captured the original into their own namespace and
    # won't see our patch on the schema module.
    for _mod_name in ("boltz.data.parse.yaml", "boltz.data.parse.fasta"):
        try:
            import importlib
            _m = importlib.import_module(_mod_name)
            if hasattr(_m, "parse_boltz_schema"):
                _m.parse_boltz_schema = _wrap
        except Exception:
            pass
    logger.info("[bt2_jax] contact/pocket constraint logger installed")


def install_runtime_feats_logger() -> None:
    """Verify constraints actually reach the contact embedder + TFG potential.

    Parse-time logging (install_constraint_logger) only proves the YAML was
    parsed into Record.inference_options. The featurizer can still drop
    them (e.g. token1/token2 lookup mismatch → silent skip). This patch
    wraps PredictionDataset.__getitem__ to inspect the featurized tensors
    AFTER process() builds them, on the first call only:

      - features['contact_conditioning'][N,N,5] — contact embedder input.
        Count cells with class==CONTACT (=4) → should equal 2·N_contacts
        (symmetric, idx1↔idx2 and idx2↔idx1).
      - features['contact_pair_index'] — TFG contact-potential index pairs
        (rows = count of `force=True` contacts).
      - features['contact_thresholds'] — TFG max_distance per pair.

    Mismatch between parsed contacts and these counts ⇒ featurizer
    dropped them ⇒ contact embedder / TFG potential are no-ops.
    """
    try:
        from boltz.data.module import inferencev2 as _inv2  # type: ignore
    except Exception as e:
        logger.warning(f"[bt2 feats-log] inferencev2 import failed: {e}")
        return

    _logged = [False]
    _orig_get = _inv2.PredictionDataset.__getitem__

    def _wrap(self, idx):
        features = _orig_get(self, idx)
        if _logged[0]:
            return features
        try:
            import torch as _torch
            import numpy as _np
            CC_CONTACT = 4   # const.contact_conditioning_info["CONTACT"]
            sys.stderr.write("[bt2 feats] runtime constraint check (first batch):\n")
            cc = features.get("contact_conditioning")
            if cc is not None:
                arr = cc.numpy() if isinstance(cc, _torch.Tensor) else _np.asarray(cc)
                # contact_conditioning is one-hot encoded as class IDs (not yet
                # converted to one-hot tensor). Inspect raw class values; if it's
                # already one-hot ([..., 5]), reduce via argmax.
                if arr.ndim >= 3 and arr.shape[-1] == len(_inv2.const.contact_conditioning_info):
                    cls = arr.argmax(axis=-1)
                else:
                    cls = arr
                n_contact = int((cls == CC_CONTACT).sum())
                n_total = int(cls.size)
                sys.stderr.write(
                    f"  contact_conditioning  shape={tuple(arr.shape)}  "
                    f"CONTACT_cells={n_contact}  (== 2·N_contacts if symmetric)\n"
                )
            else:
                sys.stderr.write("  contact_conditioning  MISSING\n")

            # TFG potential side.
            for key in ("contact_pair_index", "contact_union_index",
                        "contact_negation_mask", "contact_thresholds"):
                v = features.get(key)
                if v is None:
                    sys.stderr.write(f"  {key}  MISSING\n")
                    continue
                arr = v.numpy() if isinstance(v, _torch.Tensor) else _np.asarray(v)
                line = f"  {key:24s}  shape={tuple(arr.shape)}"
                if key == "contact_thresholds" and arr.size > 0:
                    line += f"  values={arr.flatten()[:8].tolist()}"
                sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[bt2 feats-log] dump failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
        _logged[0] = True
        return features

    _inv2.PredictionDataset.__getitem__ = _wrap
    logger.info("[bt2_jax] runtime feats logger installed (one-shot)")


# ── AF3 alignment override (same as torch path) ─────────────────────────

def install_af3_alignment_override() -> None:
    """Monkey-patch boltz template alignment to consult AF3 sidecar JSONs."""
    try:
        from boltz.data.parse import schema as _bschema
        from boltz.data.types import TemplateInfo as _TI
    except Exception:
        return

    aligns: dict = {}
    yaml_arg = None
    for i, a in enumerate(sys.argv):
        if a == "predict" and i + 1 < len(sys.argv):
            yaml_arg = sys.argv[i + 1]
            break
    if yaml_arg:
        ya = Path(yaml_arg)
        search_dirs = [ya.parent if ya.is_file() else ya]
        for sd in search_dirs:
            for js in sd.glob("*.alignments.json"):
                try:
                    aligns.update(json.loads(js.read_text()))
                except Exception:
                    pass
    if not aligns:
        return

    def _override(template_id, chain_ids, sequences,
                  template_chain_ids, template_sequences,
                  force=False, threshold=None):
        rec = aligns.get(template_id)
        if rec is None:
            return None
        out = []
        for cid in chain_ids:
            for tcid in template_chain_ids:
                out.append(_TI(
                    name=template_id,
                    query_chain=cid,
                    query_st=int(rec.get("query_st", 0)),
                    query_en=int(rec.get("query_en", 0)),
                    template_chain=tcid,
                    template_st=int(rec.get("template_st", 0)),
                    template_en=int(rec.get("template_en", 0)),
                    force=force,
                    threshold=threshold,
                ))
        return out

    _orig_match = _bschema.get_template_records_from_matching
    _orig_search = _bschema.get_template_records_from_search

    def _wrap(orig):
        def w(template_id, chain_ids, sequences,
              template_chain_ids, template_sequences,
              force=False, threshold=None):
            r = _override(template_id, chain_ids, sequences,
                          template_chain_ids, template_sequences,
                          force=force, threshold=threshold)
            if r is not None:
                return r
            return orig(template_id, chain_ids, sequences,
                        template_chain_ids, template_sequences,
                        force=force, threshold=threshold)
        return w

    _bschema.get_template_records_from_matching = _wrap(_orig_match)
    _bschema.get_template_records_from_search = _wrap(_orig_search)
    logger.info(f"[bt2_jax] AF3 alignment override active for {len(aligns)} templates")


# ── Unified writer (BoltzWriter monkey-patch) ───────────────────────────

def install_unified_writer() -> None:
    """Replace BoltzWriter.write_on_batch_end with seed_<S>/sample_<i>/ layout."""
    import torch as _torch
    from boltz.data.write import writer as _bw_mod
    from boltz.data.write.mmcif import to_mmcif as _to_mmcif
    from boltz.data.write.pdb import to_pdb as _to_pdb
    from boltz.data.types import Coords as _Coords, Interface as _Interface

    # Patch predict_step to (a) attach a runtime_s timestamp, (b) forward
    # pdistogram for the contact_probs derivation.
    try:
        from boltz.model.models.boltz2 import Boltz2 as _Boltz2T
        _orig_predict_step = _Boltz2T.predict_step
        _peak_logged = [False]   # one-shot peak dump on first predict

        def _timed_predict_step(self, *a, **kw):
            if (
                isinstance(getattr(self, "predict_args", None), dict)
                and "pdistogram" not in self.predict_args.get("keys_dict_out", [])
            ):
                self.predict_args.setdefault("keys_dict_out", [])
                if "pdistogram" not in self.predict_args["keys_dict_out"]:
                    self.predict_args["keys_dict_out"] = list(
                        self.predict_args["keys_dict_out"]
                    ) + ["pdistogram"]
            # Reset peak counters at entry so we measure THIS forward's peak.
            try:
                if not _peak_logged[0]:
                    _torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            _t0 = time.perf_counter()
            out = _orig_predict_step(self, *a, **kw)
            if isinstance(out, dict):
                out["runtime_s"] = float(time.perf_counter() - _t0)
            # One-shot peak dump: top-10 LIVE CUDA tensors right after forward,
            # plus peak_allocated bytes (= max torch transient).
            if not _peak_logged[0]:
                _peak_logged[0] = True
                try:
                    import gc
                    peak = _torch.cuda.max_memory_allocated() / 1e9
                    peak_r = _torch.cuda.max_memory_reserved() / 1e9
                    sys.stderr.write(
                        f"[bt2-mem post-forward] torch.cuda peak_alloc={peak:.2f}GB  peak_reserved={peak_r:.2f}GB\n"
                    )
                    sizes = []
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, _torch.Tensor) and obj.is_cuda:
                                sizes.append((
                                    obj.numel() * obj.element_size(),
                                    tuple(obj.shape),
                                    str(obj.dtype).replace("torch.", ""),
                                ))
                        except Exception:
                            continue
                    sizes.sort(reverse=True)
                    sys.stderr.write(
                        f"[bt2-mem post-forward] live cuda tensors={len(sizes)} (top 15):\n"
                    )
                    for nbytes, shape, dt in sizes[:15]:
                        sys.stderr.write(
                            f"  {nbytes/1e9:7.3f}GB  shape={shape}  dtype={dt}\n"
                        )
                    sys.stderr.flush()
                except Exception as e:
                    sys.stderr.write(f"[bt2-mem post-forward] dump failed: {e}\n")
                    sys.stderr.flush()
            return out

        _Boltz2T.predict_step = _timed_predict_step
    except Exception as e:
        logger.warning(f"[bt2_jax unified] predict_step wrap failed: {e}")

    def _bt_seed_now() -> int:
        return int(os.environ.get("BT2_SEED", "1"))

    def _on_batch_end(self, trainer, pl_module, prediction, batch_indices,
                      batch, batch_idx, dataloader_idx):
        try:
            return _write_inner(self, prediction, batch)
        except Exception:
            import traceback
            sys.stderr.write("=== [bt2_jax unified writer] EXCEPTION ===\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    def _write_inner(self, prediction, batch):
        if prediction.get("exception"):
            self.failed += 1
            return
        records = batch["record"]
        coords = prediction["coords"].unsqueeze(0)
        pad_masks = prediction["masks"]

        if "confidence_score" in prediction:
            argsort = _torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        seed = _bt_seed_now()

        for record, coord, pad_mask in zip(records, coords, pad_masks):
            seed_dir = self.output_dir / record.id / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Distogram dump (sample-independent, seed-level).
            pdist = prediction.get("pdistogram")
            if pdist is not None:
                try:
                    pd = pdist
                    if hasattr(pd, "dim"):
                        if pd.dim() == 5:
                            pd = pd[0, ..., 0, :]
                        elif pd.dim() == 4:
                            pd = pd[0]
                    pd_np = (pd.detach().cpu().float().numpy()
                             if isinstance(pd, _torch.Tensor) else np.asarray(pd))
                    np.savez_compressed(seed_dir / "distogram_logits.npz", distogram=pd_np)
                except Exception as e:
                    logger.warning(f"[bt2_jax distogram] {type(e).__name__}: {e}")

            # Load source structure for CIF reconstruction.
            data_path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                from boltz.data.types import StructureV2
                struct = StructureV2.load(data_path)
            else:
                from boltz.data.types import Structure
                struct = Structure.load(data_path)
            struct = struct.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                model_coord = coord[model_idx]
                coord_unpad = model_coord[pad_mask.bool()].cpu().numpy()
                atoms = struct.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    coord_unpad = np.array([(x,) for x in coord_unpad], dtype=_Coords)
                residues = struct.residues
                residues["is_present"] = True
                interfaces = np.array([], dtype=_Interface)
                if self.boltz2:
                    new_struct = replace(struct, atoms=atoms, residues=residues,
                                         interfaces=interfaces, coords=coord_unpad)
                else:
                    new_struct = replace(struct, atoms=atoms, residues=residues,
                                         interfaces=interfaces)

                rank = idx_to_rank[model_idx]
                sample_dir = seed_dir / f"sample_{rank}"
                arrays_dir = sample_dir / "arrays"
                arrays_dir.mkdir(parents=True, exist_ok=True)

                plddts = prediction.get("plddt")
                plddt_atom = plddts[model_idx] if plddts is not None else None

                if self.output_format == "pdb":
                    (sample_dir / "model.pdb").write_text(
                        _to_pdb(new_struct, plddts=plddt_atom, boltz2=self.boltz2))
                else:
                    cif_path = sample_dir / "model.cif"
                    cif_path.write_text(
                        _to_mmcif(new_struct, plddts=plddt_atom, boltz2=self.boltz2))

                # plddt arrays — boltz's plddt is per-token, range 0..1.
                # Scale ×100 for cross-model parity.
                if plddt_atom is not None:
                    p100 = plddt_atom * 100
                    _save_npy(arrays_dir / "plddt_atom.npy", p100)
                    _save_npy(arrays_dir / "plddt_token.npy", p100)

                # pae / pde — raw expected-value 2D matrices from prediction.
                if "pae" in prediction:
                    _save_npy(arrays_dir / "pae.npy", prediction["pae"][model_idx])
                if "pde" in prediction:
                    _save_npy(arrays_dir / "pde.npy", prediction["pde"][model_idx])

                # contact_probs from pdistogram (AF3/OF3 bin_top<=8Å rule).
                contact_probs_np = None
                if pdist is not None:
                    try:
                        pd_full = pdist
                        if hasattr(pd_full, "dim"):
                            if pd_full.dim() == 5:
                                pd_full = pd_full[0, ..., 0, :]
                            elif pd_full.dim() == 4:
                                pd_full = pd_full[0]
                        pdn = (pd_full.detach().cpu().float().numpy()
                               if isinstance(pd_full, _torch.Tensor) else np.asarray(pd_full))
                        pdn_shift = pdn - pdn.max(axis=-1, keepdims=True)
                        pp = np.exp(pdn_shift)
                        pp /= pp.sum(axis=-1, keepdims=True)
                        K = pp.shape[-1]
                        boundaries = np.linspace(2.0, 22.0, K - 1, dtype=np.float32)
                        step = (22.0 - 2.0) / (K - 2)
                        bin_tops = np.append(boundaries, boundaries[-1] + step)
                        contact_probs_np = pp[..., bin_tops <= 8.0].sum(axis=-1).astype(np.float32)
                        _save_npy(arrays_dir / "contact_probs.npy", contact_probs_np)
                    except Exception as e:
                        logger.warning(f"[bt2_jax contact_probs] {type(e).__name__}: {e}")

                # Summary
                pde_per_sample = prediction["pde"][model_idx] if "pde" in prediction else None
                gpde_unified = _compute_gpde(pde_per_sample, contact_probs_np)
                try:
                    n_atoms_real = int(len(new_struct.atoms))
                except Exception:
                    n_atoms_real = None
                n_tokens_real = int(plddt_atom.shape[-1]) if plddt_atom is not None else None
                _plddt_scalar = (_f(prediction["complex_plddt"][model_idx])
                                 if "complex_plddt" in prediction else None)
                summary = {
                    "model_name": "bt2_jax",
                    "seed": seed,
                    "sample_idx": int(rank),
                    "n_tokens": n_tokens_real,
                    "n_atoms": n_atoms_real,
                    "plddt": _plddt_scalar * 100 if _plddt_scalar is not None else None,
                    "ranking_score": (_f(prediction["confidence_score"][model_idx])
                                      if "confidence_score" in prediction else None),
                    "ptm": _f(prediction["ptm"][model_idx]) if "ptm" in prediction else None,
                    "iptm": _f(prediction["iptm"][model_idx]) if "iptm" in prediction else None,
                    "gpde": gpde_unified,
                    "has_clash": None,
                }
                # chain_ptm / chain_pair_iptm / chain_iptm (letter-keyed,
                # symmetrized; matches torch unified writer).
                if "pair_chains_iptm" in prediction:
                    raw_cpi = prediction["pair_chains_iptm"]
                    raw_keys = sorted(raw_cpi.keys(), key=lambda x: int(x))
                    chain_ptm = {
                        _chain_letter(int(k)): _f(raw_cpi[k][k][model_idx])
                        for k in raw_keys
                    }
                    cpi = {}
                    for i1 in raw_keys:
                        L1 = _chain_letter(int(i1))
                        cpi[L1] = {}
                        for i2 in raw_keys:
                            L2 = _chain_letter(int(i2))
                            if i1 == i2:
                                cpi[L1][L2] = _f(raw_cpi[i1][i2][model_idx])
                            else:
                                a = _f(raw_cpi[i1][i2][model_idx])
                                b = _f(raw_cpi[i2][i1][model_idx])
                                cpi[L1][L2] = (
                                    0.5 * (a + b) if (a is not None and b is not None)
                                    else (a if a is not None else b)
                                )
                    summary["chain_ptm"] = chain_ptm
                    summary["chain_pair_iptm"] = cpi
                    if len(raw_keys) > 1:
                        chain_iptm = {}
                        for L in chain_ptm:
                            offs = [cpi[L][L2] for L2 in cpi[L]
                                    if L2 != L and cpi[L][L2] is not None]
                            chain_iptm[L] = float(np.mean(offs)) if offs else None
                        summary["chain_iptm"] = chain_iptm
                    else:
                        summary["chain_iptm"] = None
                else:
                    summary["chain_ptm"] = None
                    summary["chain_pair_iptm"] = None
                    summary["chain_iptm"] = None

                _save_json(sample_dir / "summary.json", _round3(summary))
                _save_json(sample_dir / "meta.json", {
                    "writer": "BoltzWriter (bt2_jax unified)",
                    "model": "bt2_jax",
                    "job_id": record.id,
                    "seed": seed,
                    "sample_idx": int(rank),
                    "native_sample_idx": int(model_idx),
                })

            # timing.json LAST — watcher uses its presence as the seed-done signal.
            if "runtime_s" in prediction:
                (seed_dir / "timing.json").write_text(
                    json.dumps({"runtime_s": float(prediction["runtime_s"])}, indent=4)
                )

    _bw_mod.BoltzWriter.write_on_batch_end = _on_batch_end
    logger.info("[bt2_jax] BoltzWriter monkey-patched → seed_<S>/sample_<i>/ layout")


# ── Optional: surface the real OOM instead of boltz's silent skip ─────────

def install_debug_oom() -> None:
    """When BOLTZ_DEBUG_OOM=1, REIMPLEMENT Boltz2.predict_step so it calls the
    forward directly and re-raises, bypassing boltz's own `except RuntimeError:
    if "out of memory" -> return {"exception": True}` swallow (boltz2.py:~1128).
    Without this, the JAX path silently skips the batch and the existing
    post-forward [bt2-mem] dump never runs (the swallow returns before the
    failing allocation can be inspected). JAX-path analogue of the patch in
    scripts/boltz_inference.py, which the JAX entry deliberately never imports.

    On failure it dumps, to BOTH stdout (captured in inference.log) and stderr,
    the exception, torch.cuda peak/summary and the largest live CUDA tensors,
    then re-raises so the run fails loudly. No-op unless BOLTZ_DEBUG_OOM=1."""
    if os.environ.get("BOLTZ_DEBUG_OOM", "0") != "1":
        return
    import traceback as _tb
    try:
        import torch as _torch
        from boltz.model.models.boltz2 import Boltz2 as _Boltz2D
    except Exception as e:  # pragma: no cover
        logger.warning(f"[bt2_jax debug-oom] install failed: {type(e).__name__}: {e}")
        return

    def _emit(msg: str) -> None:
        # Duplicate to both streams: inference.log capture differs per harness.
        sys.stdout.write(msg); sys.stdout.flush()
        sys.stderr.write(msg); sys.stderr.flush()

    def _dump_cuda(tag: str) -> None:
        try:
            import gc
            peak = _torch.cuda.max_memory_allocated() / 1e9
            peak_r = _torch.cuda.max_memory_reserved() / 1e9
            _emit(f"[BOLTZ_DEBUG_OOM {tag}] torch.cuda peak_alloc={peak:.2f}GB "
                  f"peak_reserved={peak_r:.2f}GB\n")
            sizes = []
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, _torch.Tensor) and obj.is_cuda:
                        sizes.append((obj.numel() * obj.element_size(),
                                      tuple(obj.shape),
                                      str(obj.dtype).replace("torch.", "")))
                except Exception:
                    continue
            sizes.sort(reverse=True)
            _emit(f"[BOLTZ_DEBUG_OOM {tag}] live cuda tensors={len(sizes)} (top 15):\n")
            for nbytes, shape, dt in sizes[:15]:
                _emit(f"  {nbytes/1e9:7.3f}GB  shape={shape}  dtype={dt}\n")
            _emit(_torch.cuda.memory_summary() + "\n")
        except Exception as e:
            _emit(f"[BOLTZ_DEBUG_OOM {tag}] dump failed: {type(e).__name__}: {e}\n")

    def _debug_predict_step(self, batch, *a, **kw):
        pa = self.predict_args
        try:
            out = self(
                batch,
                recycling_steps=pa["recycling_steps"],
                num_sampling_steps=pa["sampling_steps"],
                diffusion_samples=pa["diffusion_samples"],
                max_parallel_samples=pa["max_parallel_samples"],
                run_confidence_sequentially=True,
            )
        except BaseException:  # torch OOM AND jax XlaRuntimeError(RESOURCE_EXHAUSTED)
            _emit("=== [BOLTZ_DEBUG_OOM] forward() raised ===\n")
            _tb.print_exc(file=sys.stdout); sys.stdout.flush()
            _tb.print_exc(file=sys.stderr); sys.stderr.flush()
            _dump_cuda("on-raise")
            raise
        return {"exception": False, "coords": out.get("sample_atom_coords")}

    _Boltz2D.predict_step = _debug_predict_step
    logger.info("[bt2_jax] BOLTZ_DEBUG_OOM=1: predict_step reimplemented "
                "(no OOM swallow); real traceback + CUDA dump will surface")


# ── Template OOM fix: chunk TemplateV2Module over the template axis ───────

def install_template_chunking() -> None:
    """Replace TemplateV2Module.forward with a template-by-template loop.

    Boltz builds the per-template pair stack v=[B,T,N,N,td] in one shot
    (trunkv2.py: `v = z_proj(z_norm(z[:,None])) + a_tij; v = v + pairformer(v)`),
    then mean-aggregates over T. At T~29, N~1836 that is two ~25 GB float32
    tensors (50 GB) — the dominant PyTorch allocation that starves the JAX
    trunk and OOMs at the dlpack bridge (see [[project_bt2jax_largeN_oom]]).

    The aggregation `u = sum_t(v_t * mask_t) / num_templates` is LINEAR, so
    computing each template's v_t and accumulating the weighted sum is exactly
    equivalent (the JAX pairformer wrapper already loops per batch element, so
    a per-template call returns the same result as the batched one). Peak drops
    from O(T·N²) to O(N²) — template stage ~50 GB → ~3 GB.

    Numerically faithful: same ops, same float32 accumulation, only the T loop
    is reordered (float add is non-associative but the delta is ~1e-6, far
    below model noise). No-op via BIO_TRT_BT2_TEMPLATE_CHUNK=0."""
    if os.environ.get("BIO_TRT_BT2_TEMPLATE_CHUNK", "1") != "1":
        return
    try:
        import torch as _torch
        from torch.nn.functional import one_hot as _one_hot
        from boltz.model.modules.trunkv2 import TemplateV2Module as _Tmpl
    except Exception as e:  # pragma: no cover
        logger.warning(f"[bt2_jax] template-chunk install failed: {type(e).__name__}: {e}")
        return

    def _chunked_forward(self, z, feats, pair_mask, use_kernels: bool = False):
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = feats["template_mask"].any(dim=2).float().sum(dim=1).clamp(min=1)

        B, T = res_type.shape[:2]  # noqa: N806
        # z-side projection is shared across templates → compute once.
        z_part = self.z_proj(self.z_norm(z[:, None]))  # [B,1,N,N,td]

        u_accum = None
        boundaries = _torch.linspace(
            self.min_dist, self.max_dist, self.num_bins - 1, device=z.device
        )
        for t in range(T):
            res_type_t = res_type[:, t:t + 1]
            frame_rot_t = frame_rot[:, t:t + 1]
            frame_t_t = frame_t[:, t:t + 1]
            frame_mask_t = frame_mask[:, t:t + 1]
            cb_coords_t = cb_coords[:, t:t + 1]
            ca_coords_t = ca_coords[:, t:t + 1]
            cb_mask_t = cb_mask[:, t:t + 1]
            vis_t = visibility_ids[:, t:t + 1]

            b_cb_mask = (cb_mask_t[:, :, :, None] * cb_mask_t[:, :, None, :])[..., None]
            b_frame_mask = (frame_mask_t[:, :, :, None] * frame_mask_t[:, :, None, :])[..., None]
            tmlp_pair_mask = (vis_t[:, :, :, None] == vis_t[:, :, None, :]).float()

            with _torch.autocast(device_type="cuda", enabled=False):
                cb_dists = _torch.cdist(cb_coords_t, cb_coords_t)
                distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
                distogram = _one_hot(distogram, num_classes=self.num_bins)

                fr = frame_rot_t.unsqueeze(2).transpose(-1, -2)
                ft = frame_t_t.unsqueeze(2).unsqueeze(-1)
                ca = ca_coords_t.unsqueeze(3).unsqueeze(-1)
                vector = _torch.matmul(fr, (ca - ft))
                norm = _torch.norm(vector, dim=-1, keepdim=True)
                unit_vector = _torch.where(norm > 0, vector / norm, _torch.zeros_like(vector))
                unit_vector = unit_vector.squeeze(-1)

                a_tij = _torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
                a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)
                res_type_i = res_type_t[:, :, :, None].expand(-1, -1, -1, res_type_t.size(2), -1)
                res_type_j = res_type_t[:, :, None, :].expand(-1, -1, res_type_t.size(2), -1, -1)
                a_tij = _torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
                a_tij = self.a_proj(a_tij)

            v_t = z_part + a_tij                       # [B,1,N,N,td]
            v_t = v_t.view(B, *v_t.shape[2:])          # [B,N,N,td]  (T=1 squeezed)
            v_t = v_t + self.pairformer(v_t, pair_mask, use_kernels=use_kernels)
            v_t = self.v_norm(v_t)

            w_t = template_mask[:, t][:, None, None, None]  # [B,1,1,1]
            contrib = v_t * w_t
            u_accum = contrib if u_accum is None else u_accum + contrib

        if u_accum is None:  # T==0: match original's empty-sum → zeros
            u_accum = z_part[:, 0] * 0.0
        u = u_accum / num_templates[:, None, None, None].to(u_accum)
        u = self.u_proj(self.relu(u))
        return u

    _Tmpl.forward = _chunked_forward
    logger.info("[bt2_jax] TemplateV2Module.forward → per-template chunked "
                "(peak O(T·N²)→O(N²); set BIO_TRT_BT2_TEMPLATE_CHUNK=0 to disable)")


# ── torch.compile of Boltz-2 submodules (BT2_TORCH_COMPILE) ───────────────

def install_torch_compile() -> None:
    """Wrap chosen Boltz-2 submodules in torch.compile right after load.

    BT2_TORCH_COMPILE = comma list of: score_model (the diffusion step network,
    run ~200× per sample), pairformer, msa, template, confidence.
    BT2_TORCH_COMPILE_MODE = default | reduce-overhead (CUDA graphs) | max-autotune.

    Boltz itself exposes compile_* constructor flags but its CLI never sets them;
    wrapping after load keeps the checkpoint loading path untouched and lets the
    mode be chosen. Shapes are static across sampling steps (dynamic=False).
    Inductor's on-disk cache (TORCHINDUCTOR_CACHE_DIR, set by lantern-engine to the
    site cache) makes the compile a one-off per shape class."""
    targets = [t.strip() for t in os.environ.get("BT2_TORCH_COMPILE", "").split(",") if t.strip()]
    if not targets:
        return
    mode = os.environ.get("BT2_TORCH_COMPILE_MODE", "default")
    try:
        import torch as _torch
        from boltz.model.models.boltz2 import Boltz2 as _Boltz2C
    except Exception as e:  # pragma: no cover
        logger.warning(f"[bt2 compile] install failed: {type(e).__name__}: {e}")
        return

    def _wrap(model):
        kw = {"dynamic": False, "fullgraph": False}
        if mode != "default":
            kw["mode"] = mode
        done = []
        for t in targets:
            if t == "score_model":
                sm = model.structure_module
                sm.score_model = _torch.compile(sm.score_model, **kw)
            elif t == "pairformer":
                model.pairformer_module = _torch.compile(model.pairformer_module, **kw)
            elif t == "msa":
                model.msa_module = _torch.compile(model.msa_module, **kw)
            elif t == "template" and getattr(model, "template_module", None) is not None:
                model.template_module = _torch.compile(model.template_module, **kw)
            elif t == "confidence" and getattr(model, "confidence_module", None) is not None:
                model.confidence_module = _torch.compile(model.confidence_module, **kw)
            else:
                logger.warning(f"[bt2 compile] unknown/absent target {t!r}; skipped")
                continue
            done.append(t)
        logger.info("[bt2 compile] torch.compile(mode=%s) on: %s", mode, ", ".join(done) or "-")
        print(f"[bt2 compile] torch.compile(mode={mode}) on: {', '.join(done) or '-'}", flush=True)

    _orig_load = _Boltz2C.load_from_checkpoint

    def _patched_load(cls, *a, **kw):
        model = _orig_load(*a, **kw)
        _wrap(model)
        return model

    _Boltz2C.load_from_checkpoint = classmethod(_patched_load)


# ── Stage profiler (LANTERN_ENGINE_PROFILE=1) ────────────────────────────────────

def install_stage_profiler() -> None:
    """Per-stage wall time and peak memory of one Boltz-2 forward, any backend.

    Enabled by LANTERN_ENGINE_PROFILE=1. On the first predict_step the model's stage
    modules (input_embedder, template_module, msa_module, pairformer_module,
    confidence_module) get synchronising forward hooks and
    structure_module.sample (the diffusion loop) is wrapped. Stage timings
    accumulate per predict_step and are written as JSON to
    $LANTERN_ENGINE_PROFILE_OUT (default ./profile.json) and printed as a table.

    Works unchanged for the TRT and JAX hot-swaps because both replace the
    stage modules with nn.Module wrappers whose forward returns torch
    tensors — the synchronise after the call therefore covers engine / XLA
    execution too. Hooks cost two cuda synchronises per stage call (~µs)."""
    if os.environ.get("LANTERN_ENGINE_PROFILE", "0") != "1":
        return
    try:
        import json as _json
        import torch as _torch
        from boltz.model.models.boltz2 import Boltz2 as _Boltz2P
    except Exception as e:  # pragma: no cover
        logger.warning(f"[bt2 profile] install failed: {type(e).__name__}: {e}")
        return

    out_path = os.environ.get("LANTERN_ENGINE_PROFILE_OUT") or "profile.json"
    STAGES = ("input_embedder", "template_module", "msa_module", "pairformer_module",
              "confidence_module")
    state = {"installed": False, "stages": {}, "steps": [], "t_step0": None}

    def _sync():
        try:
            _torch.cuda.synchronize()
        except Exception:
            pass

    def _acc(name, dt):
        st = state["stages"].setdefault(name, {"calls": 0, "time_s": 0.0, "peak_alloc_gb": 0.0})
        st["calls"] += 1
        st["time_s"] += dt
        try:
            st["peak_alloc_gb"] = max(st["peak_alloc_gb"], _torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass

    def _pre(name):
        def hook(mod, args):
            _sync()
            mod.__dict__["_fmx_t0"] = time.perf_counter()
        return hook

    def _post(name):
        def hook(mod, args, output):
            _sync()
            t0 = mod.__dict__.pop("_fmx_t0", None)
            if t0 is not None:
                _acc(name, time.perf_counter() - t0)
        return hook

    def _install_on(model):
        for name in STAGES:
            mod = getattr(model, name, None)
            if mod is None or not isinstance(mod, _torch.nn.Module):
                continue
            mod.register_forward_pre_hook(_pre(name))
            mod.register_forward_hook(_post(name))
        sm = getattr(model, "structure_module", None)
        if sm is not None and hasattr(sm, "sample"):
            _orig_sample = sm.sample

            def _timed_sample(*a, **kw):
                _sync()
                t0 = time.perf_counter()
                try:
                    return _orig_sample(*a, **kw)
                finally:
                    _sync()
                    _acc("diffusion_sample", time.perf_counter() - t0)
            sm.sample = _timed_sample
        state["installed"] = True

    def _dump():
        stages = state["stages"]
        total = sum(s["time_s"] for s in stages.values())
        step_total = sum(state["steps"]) if state["steps"] else None
        rec = {
            "stages": stages,
            "stages_total_s": total,
            "predict_steps": state["steps"],
            "unattributed_s": (step_total - total) if step_total is not None else None,
            "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
        }
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(_json.dumps(rec, indent=2) + "\n")
        except Exception as e:
            logger.warning(f"[bt2 profile] could not write {out_path}: {e}")
        lines = ["[bt2 profile] stage                 calls    time_s   share  peak_alloc_gb"]
        for name, s in sorted(stages.items(), key=lambda kv: -kv[1]["time_s"]):
            share = (s["time_s"] / total * 100) if total else 0.0
            lines.append(f"[bt2 profile] {name:<20s} {s['calls']:5d} {s['time_s']:9.2f} {share:6.1f}%  {s['peak_alloc_gb']:7.2f}")
        if step_total is not None:
            lines.append(f"[bt2 profile] predict_step total {step_total:.2f}s; unattributed {step_total - total:.2f}s")
        msg = "\n".join(lines) + "\n"
        sys.stdout.write(msg); sys.stdout.flush()
        sys.stderr.write(msg); sys.stderr.flush()

    _orig_ps = _Boltz2P.predict_step

    def _profiled_predict_step(self, *a, **kw):
        if not state["installed"]:
            _install_on(self)
        _sync()
        t0 = time.perf_counter()
        try:
            return _orig_ps(self, *a, **kw)
        finally:
            _sync()
            state["steps"].append(time.perf_counter() - t0)
            _dump()

    _Boltz2P.predict_step = _profiled_predict_step
    logger.info("[bt2 profile] LANTERN_ENGINE_PROFILE=1: stage hooks armed; output → %s", out_path)


# ── Multi-seed loop ─────────────────────────────────────────────────────

def install_multiseed_predict() -> None:
    """Loop BT2_SEEDS inside one model load. Each pass sets BT2_SEED env."""
    seeds_env = os.environ.get("BT2_SEEDS", "").strip()
    if not seeds_env:
        return
    try:
        seeds = [int(s) for s in seeds_env.split(",") if s.strip()]
    except ValueError:
        logger.warning(f"[bt2_jax multi-seed] BT2_SEEDS={seeds_env!r} not all-int; ignoring")
        return
    if len(seeds) <= 1:
        if len(seeds) == 1:
            os.environ["BT2_SEED"] = str(seeds[0])
        return

    import torch as _torch
    try:
        from pytorch_lightning import seed_everything as _se
    except Exception:
        _se = None
    from pytorch_lightning import Trainer as _Trainer

    _orig_predict = _Trainer.predict

    def _dump_mem(stage: str, trainer=None):
        """Print a fine-grained memory breakdown to inference.log.

        Sections:
          - torch.cuda allocator state (allocated, reserved, active)
          - JAX live arrays count & total bytes
          - top-10 torch tensor blocks by size from gc-traced refs
          - if trainer given: model.parameters + model.buffers sum
        """
        import gc
        try:
            stats = _torch.cuda.memory_stats()
            allocated_b = stats.get("allocated_bytes.all.current", 0)
            reserved_b = stats.get("reserved_bytes.all.current", 0)
            active_b = stats.get("active_bytes.all.current", 0)
            sys.stderr.write(
                f"[bt2-mem {stage}] torch.cuda  alloc={allocated_b/1e9:.2f}GB  "
                f"reserved={reserved_b/1e9:.2f}GB  active={active_b/1e9:.2f}GB\n"
            )
        except Exception as e:
            sys.stderr.write(f"[bt2-mem {stage}] torch.cuda stats failed: {e}\n")

        try:
            import jax  # type: ignore
            arrs = jax.live_arrays()
            n = len(arrs)
            total_b = sum(getattr(a, "nbytes", 0) for a in arrs)
            sys.stderr.write(
                f"[bt2-mem {stage}] jax  live_arrays={n}  total={total_b/1e9:.2f}GB\n"
            )
            # Device-level stats — captures the VMM pool reservation (which is
            # invisible to live_arrays). Keys vary by allocator; pool_bytes /
            # bytes_in_use / peak_bytes_in_use are typical.
            try:
                dev = jax.devices()[0]
                ms = dev.memory_stats()
                ms_str = ", ".join(
                    f"{k}={v/1e9:.2f}GB" for k, v in ms.items()
                    if isinstance(v, (int, float)) and "bytes" in k.lower()
                )
                if ms_str:
                    sys.stderr.write(f"[bt2-mem {stage}] jax dev0  {ms_str}\n")
            except Exception as e2:
                sys.stderr.write(f"[bt2-mem {stage}] jax dev memory_stats: {e2}\n")
            # Also probe nvidia-smi for total used-on-GPU (== process RSS on GPU).
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    sys.stderr.write(f"[bt2-mem {stage}] nvidia-smi  used,free(MiB)= {r.stdout.strip()}\n")
            except Exception:
                pass
        except Exception as e:
            sys.stderr.write(f"[bt2-mem {stage}] jax stats failed: {e}\n")

        if trainer is not None:
            try:
                m = trainer.lightning_module
                p = sum(p.numel() * p.element_size() for p in m.parameters())
                b = sum(bf.numel() * bf.element_size() for bf in m.buffers())
                sys.stderr.write(
                    f"[bt2-mem {stage}] model.params={p/1e9:.2f}GB  "
                    f"buffers={b/1e9:.2f}GB\n"
                )
            except Exception as e:
                sys.stderr.write(f"[bt2-mem {stage}] model size failed: {e}\n")

        # Top-K torch.Tensor blocks by size (gc-traced — live CUDA tensors).
        try:
            sizes: list[tuple[int, tuple, str]] = []
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, _torch.Tensor) and obj.is_cuda:
                        sizes.append((
                            obj.numel() * obj.element_size(),
                            tuple(obj.shape),
                            str(obj.dtype).replace("torch.", ""),
                        ))
                except Exception:
                    continue
            sizes.sort(reverse=True)
            sys.stderr.write(
                f"[bt2-mem {stage}] top tensors ({len(sizes)} total):\n"
            )
            for nbytes, shape, dt in sizes[:10]:
                sys.stderr.write(
                    f"  {nbytes/1e9:6.2f}GB  shape={shape}  dtype={dt}\n"
                )
        except Exception as e:
            sys.stderr.write(f"[bt2-mem {stage}] tensor walk failed: {e}\n")
        sys.stderr.flush()

    def _free_between_seeds(trainer):
        """Aggressively drop refs held across seeds: Lightning prediction
        accumulators (predict_loop._predictions/_results), BasePredictionWriter
        per-batch buffers, then GC + torch+jax cache flush.

        Without this, 89+ GiB of activations stay pinned in PyTorch's caching
        allocator after seed_N and seed_N+1 OOMs on the template-module
        forward (Boltz2 trunk).
        """
        import gc
        # 1. Lightning's predict_loop holds returned predictions internally.
        try:
            ploop = getattr(trainer, "predict_loop", None)
            for attr in ("_predictions", "_results",
                         "epoch_batch_indices", "predictions"):
                if ploop is not None and hasattr(ploop, attr):
                    try:
                        v = getattr(ploop, attr)
                        if isinstance(v, list):
                            v.clear()
                        elif isinstance(v, dict):
                            v.clear()
                        else:
                            setattr(ploop, attr, None)
                    except Exception:
                        pass
        except Exception:
            pass
        # 2. Callbacks (e.g. BoltzWriter) may keep per-batch accumulators.
        try:
            for cb in getattr(trainer, "callbacks", []) or []:
                for attr in ("predictions", "_results", "outputs"):
                    if hasattr(cb, attr):
                        try:
                            v = getattr(cb, attr)
                            if isinstance(v, list):
                                v.clear()
                            elif isinstance(v, dict):
                                v.clear()
                            else:
                                setattr(cb, attr, None)
                        except Exception:
                            pass
        except Exception:
            pass
        # 3. GC twice (first pass breaks cycles, second pass collects freed).
        gc.collect(); gc.collect()
        try:
            _torch.cuda.empty_cache()
            _torch.cuda.ipc_collect()
        except Exception:
            pass
        # 4. JAX side: clear compilation/dispatch caches. With VMM allocator,
        # this returns large virtual-mapped buffers to the pool so torch can
        # reclaim physical pages.
        try:
            import jax  # type: ignore
            jax.clear_caches()
        except Exception:
            pass

    def _multiseed_predict(self, *args, **kwargs):
        result = None
        for i, s in enumerate(seeds):
            print(f"[bt2_jax-multiseed] === seed={s} ===", flush=True)
            os.environ["BT2_SEED"] = str(s)
            if _se is not None:
                _se(int(s), workers=True)
            else:
                _torch.manual_seed(int(s))
            if i == 0:
                _dump_mem(f"pre-seed{s}", self)
            result = _orig_predict(self, *args, **kwargs)
            _dump_mem(f"post-seed{s}-pre-free", self)
            del result
            result = None
            _free_between_seeds(self)
            _dump_mem(f"post-seed{s}-post-free", self)
        return result

    _Trainer.predict = _multiseed_predict
    logger.info(f"[bt2_jax multi-seed] active for seeds={seeds}; one model load")
