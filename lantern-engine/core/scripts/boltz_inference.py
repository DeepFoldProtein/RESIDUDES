#!/usr/bin/env python3
"""Boltz-2 inference wrapper with TRT engine injection.

Thin shim around `boltz predict` that monkey-patches the Boltz2 model
loader so the structure pairformer, token transformer, and optional
confidence pairformer are replaced by TRT engines.

Usage:
    python boltz/inference.py <boltz-predict-args>... \\
        --engines_dir /path/to/engines/boltz2

Any argv not recognized here is forwarded to boltz's Click entrypoint.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Scrub REPO_ROOT from sys.path so its `boltz/` dir doesn't shadow the
# installed `boltz` package from site-packages.
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _REPO_ROOT]
sys.path.insert(0, str(_REPO_ROOT / "thirdparty"))

from tensorrt_llm_lite import logger

# ── Pop our own flags out of argv before boltz sees them ──────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--engines_dir", type=str, default=None)
_parser.add_argument("--no_trt", action="store_true")
_trt_args, _remaining = _parser.parse_known_args()

# Inject `--recycling_steps 10` if not explicitly set: boltz's CLI default
# is 3, but we run bt2 alongside af3 (10) / pt2 (10) / of3 (10) — keep the
# refinement depth aligned across the four models. Override via
# BT2_RECYCLING_STEPS env var if needed.
def _resolved_recycling_steps(remaining):
    for i, a in enumerate(remaining):
        if a in ("--recycling_steps", "--recycling-steps"):
            if i + 1 < len(remaining):
                return remaining[i + 1], "explicit"
        if a.startswith(("--recycling_steps=", "--recycling-steps=")):
            return a.split("=", 1)[1], "explicit"
    return os.environ.get("BT2_RECYCLING_STEPS", "10"), "default"

_bt2_recycles, _bt2_recycles_src = _resolved_recycling_steps(_remaining)
if _bt2_recycles_src == "default":
    _remaining += ["--recycling_steps", str(_bt2_recycles)]
print(f"[bt2-input] recycling_steps = {_bt2_recycles} ({_bt2_recycles_src})", flush=True)

sys.argv = [sys.argv[0]] + _remaining


# ── AF3 input.json → bt2 yaml passthrough ─────────────────────────────────
# If the `predict` arg is an AF3-dialect .json (or a directory of such jsons),
# convert each to a boltz yaml in a temp dir and rewrite argv. Lets the same
# AF3 input.json drive bt2 too.
def _maybe_convert_af3_predict_arg(argv):
    import json as _json
    import tempfile
    import importlib.util
    for i, a in enumerate(argv):
        if a == "predict" and i + 1 < len(argv):
            arg = argv[i + 1]
            if not os.path.exists(arg):
                return
            spec = importlib.util.spec_from_file_location(
                "biotrt_af3_adapter",
                str(_REPO_ROOT / "scripts" / "af3_input_adapter.py"),
            )
            adapter = importlib.util.module_from_spec(spec)
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
                # Stash the MSA variant (`af3_msa`, `tcrm_msa`, ...) so
                # _log_input_summary can pass it into inputs_runtime.json
                # later — boltz's converted YAML loses the original
                # `inputs/<sto>/<variant>/` path components.
                try:
                    import re as _re
                    _vpat = _re.compile(r"/inputs/[^/]+/([^/]+)/")
                    _seen = set()
                    def _scan(o):
                        if isinstance(o, dict):
                            for v in o.values(): _scan(v)
                        elif isinstance(o, list):
                            for v in o: _scan(v)
                        elif isinstance(o, str):
                            m = _vpat.search(o)
                            if m: _seen.add(m.group(1))
                    _scan(data)
                    if len(_seen) == 1:
                        (out_dir / f"{name}.variant").write_text(next(iter(_seen)))
                except Exception:
                    pass
                # Stage AF3-derived alignment sidecar so the monkey-patch
                # below can override boltz's PairwiseAligner with the same
                # query/templateIndices that af3/pt2/of3 use. Key = the
                # cif filename stem (matches boltz's `template_id`).
                aligns: dict = {}
                try:
                    import sys as _sys
                    _scripts = str(_REPO_ROOT / "scripts")
                    if _scripts not in _sys.path:
                        _sys.path.insert(0, _scripts)
                    from inputs_meta import _count_entity_poly_seq  # type: ignore
                except Exception:
                    _count_entity_poly_seq = None  # type: ignore
                for sb in data.get("sequences") or []:
                    prot = sb.get("protein") or sb.get("proteinChain")
                    if not prot:
                        continue
                    for t in (prot.get("templates") or []):
                        cif_path = t.get("mmcifPath")
                        if not cif_path:
                            continue
                        qi = t.get("queryIndices") or []
                        ti = t.get("templateIndices") or []
                        length = None
                        if _count_entity_poly_seq is not None:
                            try:
                                length = _count_entity_poly_seq(Path(cif_path).read_text())
                            except Exception:
                                pass
                        key = Path(cif_path).stem   # boltz template_id
                        aligns[key] = {
                            "query_st": min(qi) if qi else 0,
                            "query_en": (max(qi) + 1) if qi else 0,
                            "template_st": min(ti) if ti else 0,
                            "template_en": (max(ti) + 1) if ti else 0,
                            "length": length,
                        }
                if aligns:
                    (out_dir / f"{name}.alignments.json").write_text(
                        _json.dumps(aligns)
                    )
                return out

            if os.path.isdir(arg):
                jsons = sorted(p for p in Path(arg).iterdir() if _is_af3(str(p)))
                if not jsons:
                    return
                tmp_dir = Path(tempfile.mkdtemp(prefix="biotrt_bt2_af3_"))
                for jp in jsons:
                    _convert_one(str(jp), tmp_dir)
                logger.info(f"AF3 dialect dir → bt2 yamls: {tmp_dir} ({len(jsons)} files)")
                argv[i + 1] = str(tmp_dir)
            elif _is_af3(arg):
                tmp_dir = Path(tempfile.mkdtemp(prefix="biotrt_bt2_af3_"))
                out = _convert_one(arg, tmp_dir)
                logger.info(f"AF3 dialect → bt2 yaml: {out}")
                argv[i + 1] = str(out)
            return


_maybe_convert_af3_predict_arg(sys.argv)


# When running batch (predict <DIR>) with our AF3 → yaml conversion, the
# temp dir also contains alignment-override JSONs and MSA CSV sidecars.
# boltz.main.check_inputs rejects non-yaml/fasta files, so monkey-patch it
# to only enumerate yaml/fasta entries when given a dir.
def _install_check_inputs_filter():
    try:
        import boltz.main as _bmain
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


# Surface bt2's post-pairing row count (paired vs unpaired vs total) so logs
# are comparable with af3 / pt2 / of3 patches. Wraps boltz's
# featurizerv2.construct_paired_msa.
def _install_pairing_log():
    try:
        from boltz.data.feature import featurizerv2 as _fv2
    except Exception:
        return
    _orig = _fv2.construct_paired_msa
    def _wrapped(*args, **kwargs):
        out = _orig(*args, **kwargs)
        try:
            # Boltz returns (msa, deletion, paired_mask) all shaped
            # [n_tokens, n_pairs] (token-major; each (token, pair_idx) cell
            # is 1 iff that token's chain has a paired hit at MSA depth
            # pair_idx). To get per-chain stats we map tokens → chain via
            # the input `data.tokens["asym_id"]`.
            import numpy as _np
            _msa, _del, _paired_mask = out[0], out[1], out[2]
            n_tokens = int(_msa.shape[0])
            n_pairs = int(_msa.shape[1])
            # Pull `data` from args/kwargs (1st positional in boltz signature).
            _data = kwargs.get("data") if kwargs.get("data") is not None else (args[0] if args else None)
            asym_ids = None
            if _data is not None and hasattr(_data, "tokens"):
                try:
                    asym_ids = _np.asarray(_data.tokens["asym_id"])
                except Exception:
                    asym_ids = None
            # Per-chain paired depth: rows where this chain has ≥1 non-gap
            # hit. All tokens in a chain share the same paired bit per row,
            # so sum along pair_idx for one representative token per chain.
            mask_np = _paired_mask.detach().cpu().numpy() if hasattr(_paired_mask, "detach") else _np.asarray(_paired_mask)
            per_chain_depth: list[tuple[int, int]] = []
            if asym_ids is not None and asym_ids.shape[0] == n_tokens:
                _uniq, _first_idx = _np.unique(asym_ids, return_index=True)
                for cid, idx in zip(_uniq.tolist(), _first_idx.tolist()):
                    per_chain_depth.append((int(cid), int(mask_np[idx].sum())))
            paired_rows_any = int((mask_np.any(axis=0)).sum())  # pair_idx with ≥1 chain hit
            print(
                f"[bt2 pairing] post-construct_paired_msa: "
                f"paired_msa_depth={n_pairs} (rows_with_any_chain_hit={paired_rows_any}); "
                f"n_tokens={n_tokens}; per_chain_paired_depth={per_chain_depth}",
                flush=True,
            )
            global _BT2_PAIRING_RUNTIME
            _BT2_PAIRING_RUNTIME = {
                "paired_msa_depth": n_pairs,
                "rows_with_any_chain_hit": paired_rows_any,
                "n_tokens": n_tokens,
                "per_chain_paired_depth": [
                    {"chain": cid, "depth": d} for cid, d in per_chain_depth
                ],
            }
            # DataLoader workers may fork before main — persist to env-pointed
            # JSON cache so the writer (in main) can read it back.
            import os as _os, json as _json
            _p = _os.environ.get("BT2_PAIRING_JSON")
            if _p:
                try:
                    with open(_p, "w") as _f:
                        _json.dump(_BT2_PAIRING_RUNTIME, _f)
                except Exception:
                    pass
        except Exception as _e:
            print(f"[bt2 pairing log] failed: {type(_e).__name__}: {_e}", flush=True)
        return out
    _fv2.construct_paired_msa = _wrapped


_install_pairing_log()


# ── precomputed-mode CSV dedup bypass ─────────────────────────────────────
# boltz.data.parse.csv.parse_csv dedups by `line.replace("-","").upper()`.
# That breaks our row-aligned (precomputed) pipeline: when paired_<X>.a3m
# rows are gap-padded copies of the same underlying sequence (e.g. chain
# E with no homologs has 1956 near-identical gap rows), boltz collapses
# them to 1 entry and the carefully-injected row-index taxonomy tags
# stop matching across chains.
#
# Toggle: BT2_DISABLE_CSV_DEDUP=1 (set by the precomputed branch of the
# af3-dialect adapter via env). Online mode keeps native boltz dedup so
# we don't disturb the training-time data distribution.
def _install_csv_dedup_bypass():
    if os.environ.get("BT2_DISABLE_CSV_DEDUP", "") not in ("1", "true", "yes"):
        return
    try:
        from boltz.data.parse import csv as _csvmod
        from boltz.data import const as _const
        from boltz.data.types import MSA as _MSA, MSADeletion as _MSADel, \
            MSAResidue as _MSARes, MSASequence as _MSASeq
        import pandas as _pd
        import numpy as _np
    except Exception:
        return

    def _parse_csv_no_dedup(path, max_seqs=None):
        """Drop-in replacement for boltz.data.parse.csv.parse_csv that
        DOES NOT dedup by gap-stripped sequence content. Keeps row
        alignment intact for our precomputed-mode adapter (rows N
        across chains share the same row-index taxonomy id and must
        end up at the same `seq_idx` in the resulting MSA)."""
        data = _pd.read_csv(path)
        if tuple(sorted(data.columns)) != ("key", "sequence"):
            raise ValueError("Invalid CSV format, expected columns: ['sequence', 'key']")
        sequences, deletions, residues = [], [], []
        seq_idx = 0
        for line, key in zip(data["sequence"], data["key"]):
            line = str(line).strip()
            if not line:
                continue
            taxonomy_id = -1
            if (str(key) != "nan") and (key is not None) and (key != ""):
                taxonomy_id = key
            # NOTE: dedup check intentionally omitted (BT2_DISABLE_CSV_DEDUP=1).
            residue, deletion = [], []
            count = 0; res_idx = 0
            for c in line:
                if c != "-" and c.islower():
                    count += 1; continue
                token = _const.token_ids[_const.prot_letter_to_token[c]]
                residue.append(token)
                if count > 0:
                    deletion.append((res_idx, count)); count = 0
                res_idx += 1
            rs, re_ = len(residues), len(residues) + len(residue)
            ds, de_ = len(deletions), len(deletions) + len(deletion)
            sequences.append((seq_idx, taxonomy_id, rs, re_, ds, de_))
            residues.extend(residue); deletions.extend(deletion)
            seq_idx += 1
            if (max_seqs is not None) and (seq_idx >= max_seqs):
                break
        return _MSA(
            residues=_np.array(residues, dtype=_MSARes),
            deletions=_np.array(deletions, dtype=_MSADel),
            sequences=_np.array(sequences, dtype=_MSASeq),
        )

    _csvmod.parse_csv = _parse_csv_no_dedup
    # boltz/main.py does `from boltz.data.parse.csv import parse_csv` at
    # import time, capturing the original function in its module namespace.
    # That bound name escapes the module-attr swap above, so patch it too —
    # otherwise main.py:622 keeps calling the dedup-ing version.
    _patched_modules = ["boltz.data.parse.csv"]
    try:
        import boltz.main as _bmain
        if hasattr(_bmain, "parse_csv"):
            _bmain.parse_csv = _parse_csv_no_dedup
            _patched_modules.append("boltz.main")
    except Exception:
        pass
    print("[bt2 csv-dedup-bypass] BT2_DISABLE_CSV_DEDUP=1 — boltz.parse_csv "
          f"monkey-patched in {_patched_modules} (row-aligned precomputed mode)",
          flush=True)


_install_csv_dedup_bypass()


# Last post-pairing stats from the construct_paired_msa monkey-patch above,
# surfaced into inputs_runtime.json + summary.json by the writer below.
# Workers may fork → main copy stays None; fallback path uses env-pointed
# tempfile (BT2_PAIRING_JSON, set below).
_BT2_PAIRING_RUNTIME: dict | None = None
_os_for_pair = __import__("os")
_tf_for_pair = __import__("tempfile")
if "BT2_PAIRING_JSON" not in _os_for_pair.environ:
    _os_for_pair.environ["BT2_PAIRING_JSON"] = _tf_for_pair.mkstemp(prefix="bt2_pairing_", suffix=".json")[1]


def _get_bt2_pairing_runtime():
    return None if _BT2_PAIRING_RUNTIME is None else dict(_BT2_PAIRING_RUNTIME)


def _reset_bt2_pairing_runtime():
    global _BT2_PAIRING_RUNTIME
    _BT2_PAIRING_RUNTIME = None


def _inject_trt(model, engines_dir: str) -> None:
    # Import our local Bio-TRT compat module directly (site-packages `boltz`
    # has no compat submodule).
    import importlib.util
    _compat_path = _REPO_ROOT / "boltz" / "compat.py"
    _spec = importlib.util.spec_from_file_location("_biotrt_boltz_compat", str(_compat_path))
    compat = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(compat)
    from tensorrt_bionemo.models.helper import AcceleratedConfig

    engines_dir = Path(engines_dir)
    # load_weights appends "/trt/" internally, so pass parent dir.
    pf_dir = engines_dir / "pairformer_engine"
    tt_dir = engines_dir / "token_transformer_engine"
    conf_dir = engines_dir / "confidence_pairformer_engine"

    if not (pf_dir / "trt").exists():
        raise FileNotFoundError(f"Pairformer engine not found: {pf_dir / 'trt'}")

    configs = {"structure_pairformer": AcceleratedConfig(
        checkpoint=str(pf_dir), backend="trt")}
    # Diffusion token transformer (24 blocks, run at every one of the ~200 sampling
    # steps — about half of a Boltz-2 forward below ~1000 tokens). The compat
    # adapter maps Boltz's flattened (b·m, N, d) call onto the v2 engine, which takes
    # the precomputed per-block pair bias as `z`. BT2_TRT_TOKEN_TRANSFORMER=0 keeps
    # it in PyTorch (the pre-2026-11 behaviour).
    if (tt_dir / "trt").exists() and os.environ.get("BT2_TRT_TOKEN_TRANSFORMER", "1") == "1":
        configs["token_transformer"] = AcceleratedConfig(
            checkpoint=str(tt_dir), backend="trt")
    if (conf_dir / "trt").exists():
        configs["confidence_pairformer"] = AcceleratedConfig(
            checkpoint=str(conf_dir), backend="trt")

    from tensorrt_bionemo.runtime.allocator import OnDemandContextMemoryManager
    allocator = OnDemandContextMemoryManager()
    wrapped = compat.BoltzTRT(model)
    wrapped.optimize(configs, context_memory_allocator=allocator)
    logger.info(f"TRT injected: {list(configs.keys())}")


# ── Monkey-patch Boltz2.load_from_checkpoint to inject TRT after load ────
if _trt_args.engines_dir and not _trt_args.no_trt:
    from boltz.model.models.boltz2 import Boltz2

    # Capture the already-bound classmethod (cls pre-bound).
    _orig = Boltz2.load_from_checkpoint

    def _patched(cls, *args, **kwargs):
        # _orig is bound-to-class, so drop the cls we receive.
        model = _orig(*args, **kwargs)
        logger.info(f"Injecting TRT engines from {_trt_args.engines_dir}")
        _inject_trt(model, _trt_args.engines_dir)
        return model

    Boltz2.load_from_checkpoint = classmethod(_patched)


# ── Monkey-patch TemplateModule.forward for selective multichain mask ────
# Upstream boltz's TemplateModule.forward (boltz/model/modules/trunkv2.py)
# applies asym_mask = (asym_id == asym_id) — strict block-diagonal — to the
# per-template pair features, zeroing out every inter-chain entry. This
# means a multi-chain template (yaml `chain_id: [C, D, E]` with matching
# `template_id: [c, d, e]`) carries true inter-chain geometry through the
# data pipeline (process_template_features unifies all chain-pair records
# of one template name into a single row that covers tokens across all
# chains the template touches), but the embedder zeroes the inter-chain
# pair features just before mixing into z. So multi-chain templates are
# effectively reduced to per-chain templates.
#
# Patched mask, per template t: asym_mask[t] = block_diag ∨ pair_covered[t]
# where pair_covered[t, i, j] = (cb_mask[t, i] ∧ cb_mask[t, j]). Per-chain
# template rows (cb_mask non-zero on a single chain) yield pair_covered ⊆
# block_diag, so the OR collapses to block_diag — bit-for-bit equivalent
# to upstream for those rows. Multi-chain template rows (cb_mask non-zero
# across multiple chains) get pair_covered with inter-chain entries set,
# which opens those off-diagonal blocks in asym_mask. Default ON; opt out
# with BOLTZ_MULTIMER_TEMPLATE=0.
if os.environ.get("BOLTZ_MULTIMER_TEMPLATE", "1") == "1":
    print(
        f"[bt2 multimer template] patch entry, "
        f"BT2_TEMPLATE_DUMP_PATH={os.environ.get('BT2_TEMPLATE_DUMP_PATH', '')!r}",
        flush=True,
    )
    import boltz.model.modules.trunkv2 as _bt_trunk
    import torch as _torch_mt
    import threading as _threading_mt
    from torch.nn.functional import one_hot as _one_hot_mt

    _orig_template_fwd = _bt_trunk.TemplateModule.forward

    # Optional: BT2_TEMPLATE_DUMP_PATH=<file.npz> dumps the aggregated
    # ||template_act||_2 (N, N) plus the per-template asym_mask collapsed to
    # a single (N, N) "any cross-chain coverage" mask, and asym_id, on the
    # FIRST forward of each process. Schema mirrors the AF3 dump produced
    # by AF3_TEMPLATE_DUMP_PATH so gen_inputs/viz_template_mask.py
    # --runtime <npz> renders bt2 the same way.
    _bt_dump_path = os.environ.get("BT2_TEMPLATE_DUMP_PATH", "") or None
    _bt_dump_lock = _threading_mt.Lock()
    _bt_dump_done = False

    _fwd_call_count = [0]
    _bt_tpl_w_logged = [False]

    def _patched_template_forward(self, z, feats, pair_mask, use_kernels=False):
        _fwd_call_count[0] += 1
        if _fwd_call_count[0] <= 2:
            print(
                f"[bt2 multimer template] patched forward called "
                f"(call #{_fwd_call_count[0]}) z.shape={tuple(z.shape)} "
                f"dump_path={_bt_dump_path!r}",
                flush=True,
            )
        asym_id = feats["asym_id"]
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1).clamp(min=1)

        b_cb_mask = (cb_mask[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
        b_frame_mask = (frame_mask[:, :, :, None] * frame_mask[:, :, None, :])[..., None]

        B, T = res_type.shape[:2]
        # Selective multichain mask, per template.
        block_diag = (asym_id[:, :, None] == asym_id[:, None, :])  # [B, N, N] bool
        tok_covered = cb_mask.bool()  # [B, T, N]
        pair_covered = tok_covered[..., :, None] & tok_covered[..., None, :]  # [B, T, N, N]
        asym_mask = (
            block_diag.unsqueeze(1).expand_as(pair_covered) | pair_covered
        ).to(b_cb_mask.dtype)

        with _torch_mt.autocast(device_type="cuda", enabled=False):
            cb_dists = _torch_mt.cdist(cb_coords, cb_coords)
            boundaries = _torch_mt.linspace(self.min_dist, self.max_dist,
                                             self.num_bins - 1).to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = _one_hot_mt(distogram, num_classes=self.num_bins)

            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = _torch_mt.matmul(frame_rot, (ca_coords - frame_t))
            norm = _torch_mt.norm(vector, dim=-1, keepdim=True)
            unit_vector = _torch_mt.where(
                norm > 0, vector / norm, _torch_mt.zeros_like(vector)
            ).squeeze(-1)

            a_tij = _torch_mt.cat(
                [distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1
            )
            a_tij = a_tij * asym_mask.unsqueeze(-1)

            res_type_i = res_type[:, :, :, None].expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type[:, :, None, :].expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = _torch_mt.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1).reshape(
            B * T, *pair_mask.shape[-2:]
        )
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, *v.shape[2:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, T, *v.shape[1:])

        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # First-call dump (env-gated, safe under multi-seed Trainer.predict).
        if _bt_dump_path:
            global _bt_dump_done
            with _bt_dump_lock:
                if not _bt_dump_done:
                    import numpy as _np_mt
                    # Aggregated template_act channel-norm: ||u[i,j,:]||_2.
                    # u is [B, N, N, C] post-aggregate; take batch 0.
                    act_norm = u[0].detach().to(_torch_mt.float32) \
                        .norm(dim=-1).cpu().numpy()
                    # Collapse per-template asym_mask to "any cross-chain
                    # coverage" — matches the (N, N) shape of the AF3 dump.
                    mask_any = (
                        asym_mask[0].any(dim=0)  # [N, N]
                        .to(_torch_mt.int8).cpu().numpy()
                    )
                    asym_np = asym_id[0].to(_torch_mt.int32).cpu().numpy()
                    out_dir = os.path.dirname(_bt_dump_path) or "."
                    os.makedirs(out_dir, exist_ok=True)
                    _np_mt.savez(
                        _bt_dump_path,
                        template_act_norm=act_norm.astype(_np_mt.float32),
                        multichain_mask=mask_any.astype(_np_mt.int8),
                        asym_id=asym_np.astype(_np_mt.int32),
                    )
                    _bt_dump_done = True
                    logger.info(
                        f"bt2: dumped template runtime → {_bt_dump_path}"
                    )

        u = self.u_proj(self.relu(u))
        # BT2_TPL_WEIGHT: scale the template module contribution to z before
        # the caller adds it. >1 lets a docking-quality multimer template
        # dominate over MSA signal; <1 softens it. Default 1.0 = no change.
        # Mirrors protenix's PT2_TPL_WEIGHT.
        _bt2_tpl_w = float(os.environ.get("BT2_TPL_WEIGHT", "1.0"))
        if _bt2_tpl_w != 1.0:
            u = u * _bt2_tpl_w
            if not _bt_tpl_w_logged[0]:
                logger.info(
                    f"bt2: BT2_TPL_WEIGHT={_bt2_tpl_w:.3f} applied to "
                    f"template embedder output (one-time)"
                )
                _bt_tpl_w_logged[0] = True
        return u

    _bt_trunk.TemplateModule.forward = _patched_template_forward
    print(
        "[bt2 multimer template] TemplateModule.forward patched — selective "
        "multichain_mask (BOLTZ_MULTIMER_TEMPLATE=1) "
        f"dump_path={_bt_dump_path!r}",
        flush=True,
    )

    # ── TemplateV2Module: model already supports multi-chain via
    # visibility_ids (== pdb_id on every chain a multi-chain template
    # touches; -1-asym_id elsewhere → tmlp_pair_mask is True both intra-
    # chain on uncovered chains AND across the entire union of chains a
    # given multi-chain template covers). No mask logic patch needed —
    # only add the dump hook so we can visualize the runtime template
    # signal the same way as the v1 path.
    _orig_v2_fwd = _bt_trunk.TemplateV2Module.forward
    _v2_dump_done = [False]

    def _patched_v2_forward(self, z, feats, pair_mask, use_kernels=False):
        u = _orig_v2_fwd(self, z, feats, pair_mask, use_kernels=use_kernels)
        if _bt_dump_path and not _v2_dump_done[0]:
            try:
                import numpy as _np_v2
                visibility = feats["visibility_ids"]  # [B, T, N]
                # multichain_mask = ∨_t (visibility[t,i] == visibility[t,j])
                vis_b0 = visibility[0]                # [T, N]
                pair_per_t = (
                    vis_b0[:, :, None] == vis_b0[:, None, :]
                )                                     # [T, N, N] bool
                mask_any = pair_per_t.any(dim=0).to(_torch_mt.int8) \
                    .cpu().numpy()
                # Aggregated channel-norm of u (already returned).
                act_norm = u[0].detach().to(_torch_mt.float32) \
                    .norm(dim=-1).cpu().numpy()
                # asym_id is in the parent batch's token features. v2's
                # feats dict mirrors that key.
                asym = feats.get("asym_id")
                if asym is None:
                    # Fallback: derive from negative visibility on
                    # uncovered chains (id = -1 - asym_id) — pick the
                    # first template that has any negatives.
                    asym = _torch_mt.zeros(vis_b0.shape[-1],
                                           dtype=_torch_mt.int32,
                                           device=vis_b0.device)
                asym_np = asym[0].to(_torch_mt.int32).cpu().numpy() \
                    if asym.dim() > 1 else asym.to(_torch_mt.int32) \
                                                 .cpu().numpy()
                out_dir = os.path.dirname(_bt_dump_path) or "."
                os.makedirs(out_dir, exist_ok=True)
                _np_v2.savez(
                    _bt_dump_path,
                    template_act_norm=act_norm.astype(_np_v2.float32),
                    multichain_mask=mask_any.astype(_np_v2.int8),
                    asym_id=asym_np.astype(_np_v2.int32),
                )
                _v2_dump_done[0] = True
                print(
                    f"[bt2 multimer template] V2 dump → {_bt_dump_path}",
                    flush=True,
                )
            except Exception as _e:  # pylint: disable=broad-except
                print(
                    f"[bt2 multimer template] V2 dump failed: {_e!r}",
                    flush=True,
                )
        return u

    _bt_trunk.TemplateV2Module.forward = _patched_v2_forward
    print(
        "[bt2 multimer template] TemplateV2Module.forward patched (dump-only; "
        "model already multimer-aware via visibility_ids)",
        flush=True,
    )


# ── Optional: defeat Boltz's silent OOM catch to surface the real traceback ─
# Enable with BOLTZ_DEBUG_OOM=1.  Boltz2.predict_step wraps the forward pass
# in a try/except RuntimeError that prints "ran out of memory, skipping batch"
# and returns {"exception": True} — useful for long multi-target runs, but
# hides *where* memory ran out.  Re-raising also makes any non-OOM
# RuntimeError easier to diagnose.
if os.environ.get("BOLTZ_DEBUG_OOM", "0") == "1":
    from boltz.model.models.boltz2 import Boltz2 as _Boltz2Debug
    import torch as _torch
    import traceback as _tb

    # Replace the body of Boltz2.predict_step with one that skips the inner
    # try/except RuntimeError so the real traceback (and the allocation that
    # pushed us over the VRAM budget) reach stderr.
    def _debug_predict_step(self, batch, *a, **kw):
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                max_parallel_samples=self.predict_args["max_parallel_samples"],
                run_confidence_sequentially=True,
            )
        except RuntimeError:
            sys.stderr.write("=== [BOLTZ_DEBUG_OOM] forward() raised ===\n")
            _tb.print_exc(file=sys.stderr)
            try:
                sys.stderr.write(_torch.cuda.memory_summary() + "\n")
            except Exception:
                pass
            raise
        # Fast path for debug runs: skip the per-metric unpacking and just
        # return a minimal successful dict.  We care about WHETHER forward
        # completes, not the prediction quality, when hunting OOMs.
        return {"exception": False, "coords": out.get("sample_atom_coords")}

    _Boltz2Debug.predict_step = _debug_predict_step
    logger.info("BOLTZ_DEBUG_OOM=1: OOM catcher disabled, real traceback will surface")


# ── [bt2-input] Log the MSA + template sources Boltz is about to see ──────
# Prefix [bt2-input] mirrors AF3's [af3-input], OF3's [of3-input], and pt2's
# [pt2-input] so downstream log scrapers can use a single grep pattern.
def _log_input_summary(argv):
    # Single `global` block at function entry — Python disallows declaring
    # a name `global` AFTER it has been read in the same function body.
    global _BT2_RUNTIME_MSA, _BT2_OUTPUT_DIR, _BT2_RUNTIME_MSA_BY_TARGET
    import re
    input_path = None
    out_dir = None
    for i, a in enumerate(argv):
        if a == "predict" and i + 1 < len(argv):
            input_path = argv[i + 1]
        if a == "--out_dir" and i + 1 < len(argv):
            out_dir = argv[i + 1]
    if not input_path:
        return
    # Batch mode: input_path is a dir of yamls — recurse per-target so
    # inputs_runtime.json gets dumped under each <out>/<target>/.
    if os.path.isdir(input_path):
        _BT2_RUNTIME_MSA_BY_TARGET = {}
        for y in sorted(Path(input_path).glob("*.yaml")):
            sub = list(argv)
            for i, a in enumerate(sub):
                if a == "predict" and i + 1 < len(sub):
                    sub[i + 1] = str(y); break
            _log_input_summary(sub)
            tgt = y.stem
            if _BT2_RUNTIME_MSA:
                _BT2_RUNTIME_MSA_BY_TARGET[tgt] = _BT2_RUNTIME_MSA
        # Restore aggregate state so legacy single-target path still works.
        return
    if not os.path.isfile(input_path):
        return
    try:
        text = Path(input_path).read_text()
    except Exception as e:
        logger.warning(f"failed reading {input_path}: {e}")
        return
    # Use plain print() rather than `logger` — tensorrt_llm_lite's logger
    # defaults to WARNING in this venv, so logger.info(...) is silent.
    def _p(msg: str) -> None:
        print(msg, flush=True)

    _p(f"[bt2-input] schema: {input_path}")
    runtime_msa: dict = {}
    runtime_tpls: dict = {}
    # Split the YAML into per-chain `- protein:` blocks. We rely on the boltz
    # YAML having a fixed 2-space indent for top-level list items under
    # `sequences:` (`  - protein:` at column 2). The earlier regex
    # `(?:\s+[^\n]*\n)+` was greedy and swallowed every chain into one match.
    blocks = re.split(r"^  - protein:\s*\n", text, flags=re.MULTILINE)
    for body in blocks[1:]:  # blocks[0] is everything before the first protein
        # Stop at the next top-level list item (next chain or templates: list).
        body = re.split(r"^  - \w", body, maxsplit=1, flags=re.MULTILINE)[0]
        body = re.split(r"^[a-z_]+:", body, maxsplit=1, flags=re.MULTILINE)[0]

        m_id  = re.search(r"id:\s*\[([^\]]+)\]", body)
        m_seq = re.search(r'sequence:\s*"((?:[^"\\]|\\.)*)"', body)
        m_msa = re.search(r"msa:\s*(\S+)", body)
        if not (m_id and m_seq):
            continue
        ids = [s.strip() for s in m_id.group(1).split(",") if s.strip()]
        seq = m_seq.group(1)
        msa = m_msa.group(1).strip() if m_msa else None
        # Adapter emits a CSV (sequence,key) where rows with non-empty key
        # are the paired-pool (uniprot, OX-tagged) entries. Split the runtime
        # view accordingly. For legacy a3m inputs (no csv), everything is
        # treated as unpaired since boltz can't extract tax_ids from a3m.
        unpaired_depth = 0
        paired_depth = 0
        msa_size = 0
        if msa and os.path.isfile(msa):
            msa_size = os.path.getsize(msa)
            if msa.endswith(".csv"):
                with open(msa) as f:
                    next(f, None)   # header row
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        _, _, key = line.rpartition(",")
                        if key.strip():
                            paired_depth += 1
                        else:
                            unpaired_depth += 1
            else:
                with open(msa) as f:
                    unpaired_depth = sum(1 for ln in f if ln.startswith(">"))
        _p(f"[bt2-input] chain id={ids}  (protein, len={len(seq)})")
        _p(f"  msa: src={msa or '-'}  unpaired={unpaired_depth}  paired={paired_depth}  size={msa_size}B")
        for cid in ids:
            runtime_msa[cid] = {
                "unpaired": {"src": msa, "depth": unpaired_depth} if (msa and unpaired_depth) else None,
                "paired":   {"src": msa, "depth": paired_depth} if (msa and paired_depth) else None,
            }
            runtime_tpls.setdefault(cid, [])
    # Templates: just log here (no JSON dump). The actual featurizer-time
    # alignment that boltz uses comes from `Record.templates` (filled by
    # boltz.data.parse.schema.get_template_records_from_matching with
    # Bio.Align.PairwiseAligner) — that's read in BoltzWriter, where
    # `record` is in scope, and dumped into inputs_runtime.json from there.
    tpl_pat = re.compile(
        r"-\s*cif:\s*(\S+)\s*\n\s*chain_id:\s*\[([^\]]+)\]", re.MULTILINE,
    )
    tpls = [(m.group(1).strip(), m.group(2)) for m in tpl_pat.finditer(text)]
    if tpls:
        _p(f"[bt2-input] templates: {len(tpls)} entries")
        for i, (cif, cid_str) in enumerate(tpls):
            sz = os.path.getsize(cif) if os.path.isfile(cif) else 0
            _p(f"  [{i:02d}] cif={cif}  chain_id=[{cid_str}]  size={sz}B")

    # Stash MSA view + output path on the module so _install_unified_boltz_writer
    # can merge with template alignments at writer time. (`global` already
    # declared at function entry above.)
    _BT2_RUNTIME_MSA = runtime_msa
    _BT2_OUTPUT_DIR = out_dir


# Populated by _log_input_summary at startup, consumed by the unified
# BoltzWriter monkey-patch when `record.templates` (with the pairwise
# alignments boltz actually used) is available.
_BT2_RUNTIME_MSA: dict = {}
_BT2_RUNTIME_MSA_BY_TARGET: dict = {}   # batch mode: target_name → runtime_msa
_BT2_OUTPUT_DIR = None

# Captures the confidence head's pre-softmax logits so the unified writer
# can save them via logit_codec. Boltz's `predict()` only copies the
# expected-value scalars (pae/pde/plddt) into pred_dict and drops the
# `*_logits` siblings. We hook ConfidenceModule.forward to stash them in a
# module-level dict; shape is [multiplicity, ...] matching the model's
# per-sample output ordering.
_BT2_LATEST_HEAD_LOGITS: dict = {}

try:
    from boltz.model.modules.confidencev2 import ConfidenceModule as _ConfV2
    _orig_v2_forward = _ConfV2.forward

    def _conf_v2_forward_capture(self, *args, **kwargs):
        out = _orig_v2_forward(self, *args, **kwargs)
        try:
            for k in ("pae_logits", "pde_logits", "plddt_logits"):
                v = out.get(k) if isinstance(out, dict) else None
                if v is not None:
                    _BT2_LATEST_HEAD_LOGITS[k] = v.detach()
        except Exception:
            pass
        return out

    _ConfV2.forward = _conf_v2_forward_capture
except Exception as _e:
    logger.warning(f"[bt2 head-logits hook] could not patch ConfidenceModule: {_e}")

_log_input_summary(sys.argv)


# Surface yaml path (single file OR directory of yamls in bulk mode) so the
# monkey-patched BoltzWriter can attach MSA / template provenance to
# summary.json. lookup_inputs_meta() in scripts/inputs_meta.py handles both.
def _set_input_path_env(argv):
    for i, a in enumerate(argv):
        if a == "predict" and i + 1 < len(argv):
            yaml_arg = argv[i + 1]
            if os.path.exists(yaml_arg):
                os.environ["BIO_TRT_INPUT_PATH"] = str(Path(yaml_arg).resolve())
            return


_set_input_path_env(sys.argv)


# ── Monkey-patch BoltzWriter to emit unified cross-model output layout ────
# Replaces native:
#   <output_dir>/<record.id>/<record.id>_model_<rank>.cif
#   <output_dir>/<record.id>/confidence_<record.id>_model_<rank>.json
#   <output_dir>/<record.id>/{plddt,pae,pde}_<record.id>_model_<rank>.npz
# with unified:
#   <output_dir>/<record.id>/seed_<seed>/sample_<rank>/
#       model.cif
#       summary.json (cross-model schema)
#       meta.json
#       arrays/{plddt_atom.npy, pae.npy, pde.npy, contact_probs.npy*}
#
# (* contact_probs not natively in boltz output; left None unless model exposes it.)
#
# bt2 has no notion of "seed" beyond the run-level seed; we encode it under
# seed_1 by default and let the caller's manifest seed-tag inform the path
# construction (matches af3/of3/pt2 sample_<idx> 0-indexed convention).
def _load_cif_normalize():
    """Import cif_normalize.normalize_model_cif from this script's directory."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from cif_normalize import normalize_model_cif  # type: ignore
        return normalize_model_cif
    except Exception:
        return None


_norm_cif = _load_cif_normalize()


def _install_unified_boltz_writer():
    import json as _json
    import string as _string
    import time as _time
    from dataclasses import asdict as _asdict, replace as _replace
    from pathlib import Path as _Path

    import numpy as _np
    import torch as _torch
    from boltz.data.write import writer as _bw_mod
    from boltz.data.write.mmcif import to_mmcif as _to_mmcif
    from boltz.data.write.pdb import to_pdb as _to_pdb
    from boltz.data.types import Coords as _Coords, Interface as _Interface

    # Per-call seed: read from env at write-time (not at install-time) so the
    # multi-seed wrapper around Trainer.predict can flip BT2_SEED between
    # seeds and have each pass land in seed_<N>/.
    def _bt_seed_now() -> int:
        return int(os.environ.get("BT2_SEED", "1"))

    # Mapping from boltz's numeric chain ids ("0","1",...) to AF3-style
    # letter keys ("A","B",...) for cross-model parity. Extends to AA, AB...
    # past 26 chains.
    def _chain_letter(idx: int) -> str:
        if idx < 26:
            return _string.ascii_uppercase[idx]
        return _string.ascii_uppercase[(idx // 26) - 1] + _string.ascii_uppercase[idx % 26]

    # Wrap Boltz2.predict_step so each prediction carries its own runtime
    # AND the raw `pdistogram` (Boltz's stock predict_step strips it from
    # pred_dict — we add it via predict_args["keys_dict_out"]).
    from boltz.model.models.boltz2 import Boltz2 as _Boltz2T
    _orig_predict_step = _Boltz2T.predict_step

    def _timed_predict_step(self, *a, **kw):
        # Ensure pdistogram is forwarded into pred_dict.
        if (
            isinstance(getattr(self, "predict_args", None), dict)
            and "pdistogram" not in self.predict_args.get("keys_dict_out", [])
        ):
            self.predict_args.setdefault("keys_dict_out", [])
            if "pdistogram" not in self.predict_args["keys_dict_out"]:
                self.predict_args["keys_dict_out"] = list(
                    self.predict_args["keys_dict_out"]
                ) + ["pdistogram"]
        _t0 = _time.perf_counter()
        out = _orig_predict_step(self, *a, **kw)
        if isinstance(out, dict):
            out["runtime_s"] = float(_time.perf_counter() - _t0)
        return out

    _Boltz2T.predict_step = _timed_predict_step

    def _save_npy(p: _Path, arr) -> None:
        if arr is None:
            return
        if isinstance(arr, _torch.Tensor):
            arr = arr.detach().cpu().float().numpy()
        if not isinstance(arr, _np.ndarray):
            arr = _np.asarray(arr)
        if arr.dtype.kind == "f" and arr.dtype != _np.float32:
            arr = arr.astype(_np.float32, copy=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        _np.save(p, arr)

    def _save_json(p: _Path, d: dict) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(d, indent=2, default=lambda x: float(x) if hasattr(x, "item") else None))

    def _f(x):
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return None

    def _compute_gpde(pde, contact_probs):
        """AF3 SI 5.7 Eq.16 — Σ pde·cp / Σ cp over all token pairs (incl. diag).
        Identical formula across pt2/of3/bt2/af3 writers."""
        if pde is None or contact_probs is None:
            return None
        if isinstance(pde, _torch.Tensor):
            pde = pde.detach().cpu().float().numpy()
        if isinstance(contact_probs, _torch.Tensor):
            contact_probs = contact_probs.detach().cpu().float().numpy()
        pde = _np.asarray(pde, dtype=_np.float32)
        contact_probs = _np.asarray(contact_probs, dtype=_np.float32)
        if pde.shape != contact_probs.shape or pde.ndim < 2:
            return None
        s = float(contact_probs.sum())
        if s <= 0:
            return None
        return float((pde * contact_probs).sum() / s)

    def _unified_write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx,
    ):
        try:
            return _write_unified_inner(
                self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx,
            )
        except Exception:
            import traceback as _tb
            sys.stderr.write("=== [bt2 unified writer] EXCEPTION ===\n")
            _tb.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise

    def _write_unified_inner(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx,
    ):
        if prediction["exception"]:
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

        _bt_seed = _bt_seed_now()
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Seed-level artefacts: distogram (sample-independent) + timing.
            seed_dir_top = self.output_dir / record.id / f"seed_{_bt_seed}"
            seed_dir_top.mkdir(parents=True, exist_ok=True)

            # ── Featurizer-time inputs view (real boltz alignments) ──────
            # Build templates from record.templates (TemplateInfo entries
            # populated by boltz.data.parse.schema.get_template_records_*
            # via Bio.Align.PairwiseAligner — the alignment the model
            # actually consumed, NOT AF3's queryIndices).
            try:
                rt_tpls: dict = {}
                tpl_list = getattr(record, "templates", None) or []
                # We need the source cif path + auth_asym_id to populate
                # `pdb_chain` consistently with the previous schema. The
                # boltz YAML's templates: block has cif paths in the same
                # order as record.templates; pair them up by index.
                yaml_text = ""
                _input_arg = os.environ.get("BIO_TRT_INPUT_PATH")
                if _input_arg and os.path.isfile(_input_arg):
                    yaml_text = Path(_input_arg).read_text()
                import re as _re
                cifs = [m.group(1).strip() for m in _re.finditer(
                    r"-\s*cif:\s*(\S+)\s*\n\s*chain_id:", yaml_text)]
                # boltz aligner can produce multiple local alignments per
                # cif (record.templates len > yaml cif count). Build a
                # lookup keyed by pdb-id (extracted from the cif filename
                # `template_<C>_<NN>_<pdb>.cif`) so we can resolve cif by
                # TemplateInfo.name regardless of order or duplication.
                cif_by_pdb: dict = {}
                for c in cifs:
                    stem = Path(c).stem
                    m = _re.search(r"_([0-9a-z]{4})$", stem)
                    if m:
                        cif_by_pdb.setdefault(m.group(1), c)
                # inputs_meta._read_first_auth_asym_id for pdb_chain letter,
                # _count_entity_poly_seq for template chain residue count.
                try:
                    import sys as _sys
                    _scripts = str(Path(__file__).resolve().parent)
                    if _scripts not in _sys.path:
                        _sys.path.insert(0, _scripts)
                    from inputs_meta import (
                        _read_first_auth_asym_id as _read_chain,
                        _count_entity_poly_seq as _eps_count,
                    )  # type: ignore
                except Exception:
                    _read_chain = lambda p: None  # type: ignore
                    _eps_count = lambda s: None   # type: ignore
                for i, ti in enumerate(tpl_list):
                    # Resolve cif by template_id (TemplateInfo.name) first
                    # because boltz can emit multiple TemplateInfo records
                    # per cif (multiple local alignments) and positional
                    # mapping breaks. Fall back to positional + None.
                    name = str(getattr(ti, "name", "") or "").lower()
                    cif = cif_by_pdb.get(name) or (cifs[i] if i < len(cifs) else None)
                    pdb = "?"
                    cif_text = None
                    if cif:
                        stem = Path(cif).stem
                        m = _re.search(r"_([0-9a-z]{4})$", stem)
                        if m:
                            pdb = m.group(1)
                        try:
                            cif_text = Path(cif).read_text()
                        except Exception:
                            pass
                    elif name:
                        pdb = name   # at least surface the pdb id from TemplateInfo
                    tpl_chain = (_read_chain(cif) if cif else None) or getattr(ti, "template_chain", "?")
                    length = _eps_count(cif_text) if cif_text else None
                    qs = int(getattr(ti, "query_st", 0)) + 1   # 0-idx → 1-idx
                    qe = int(getattr(ti, "query_en", 0))
                    ts = int(getattr(ti, "template_st", 0)) + 1
                    te = int(getattr(ti, "template_en", 0))
                    qchain = getattr(ti, "query_chain", None) or "A"
                    rt_tpls.setdefault(qchain, []).append({
                        "pdb_chain": f"{pdb}_{tpl_chain}",
                        "qs": qs, "qe": qe,
                        "template_st": ts, "template_en": te,
                        "length": length,
                        "selected": True,
                        "cif_path": cif,
                    })
                # Dump per-target runtime view (msa from startup, templates
                # with real boltz alignment). For batch mode we have a
                # per-target msa view in _BT2_RUNTIME_MSA_BY_TARGET; for
                # single-yaml mode use the legacy global _BT2_RUNTIME_MSA.
                _msa_view = (
                    _BT2_RUNTIME_MSA_BY_TARGET.get(record.id) or _BT2_RUNTIME_MSA
                )
                # Write under <self.output_dir>/<record.id>/ so per-target
                # batch runs each get their own inputs_runtime.json (matches
                # af3/pt2 layout). Falls through to legacy top-level when
                # self.output_dir isn't set / writer.output_dir not under
                # a per-target wrapper.
                _per_target_dir = self.output_dir / record.id
                _per_target_dir.mkdir(parents=True, exist_ok=True)
                try:
                    from inputs_meta import write_runtime_inputs as _wri  # type: ignore
                    _payload = {"msa": _msa_view, "templates": rt_tpls}
                    # Variant: read sidecar dropped by _convert_one (the
                    # converted YAML loses `inputs/<sto>/<variant>/` paths).
                    # BIO_TRT_INPUT_PATH may be a single YAML or a directory
                    # of converted YAMLs (batch mode); handle both.
                    try:
                        _yaml = os.environ.get("BIO_TRT_INPUT_PATH", "")
                        _vp = None
                        if _yaml.endswith(".yaml"):
                            _vp = Path(_yaml).with_suffix(".variant")
                        elif _yaml and Path(_yaml).is_dir():
                            _vps = sorted(Path(_yaml).glob("*.variant"))
                            if _vps:
                                _vp = _vps[0]
                        if _vp and _vp.is_file():
                            _payload["variant"] = _vp.read_text().strip()
                    except Exception:
                        pass
                    _wri(_per_target_dir, _payload)
                except Exception as _e:
                    logger.warning(f"[bt2 runtime_inputs] {type(_e).__name__}: {_e}")
            except Exception as _e:
                logger.warning(f"[bt2 record.templates] {type(_e).__name__}: {_e}")
            # Distogram → <seed_dir>/distogram_logits.npz (logit_codec int8_softmax).
            pdist = prediction.get("pdistogram")
            if pdist is not None:
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(_REPO_ROOT / "scripts"))
                    from logit_codec import save_logits as _save_logits  # type: ignore

                    pd = pdist
                    if hasattr(pd, "dim"):
                        if pd.dim() == 5:           # [B, L, L, K, bins]
                            pd = pd[0, ..., 0, :] if pd.shape[3] == 1 else pd[0, ..., 0, :]
                        elif pd.dim() == 4:         # [B, L, L, bins]
                            pd = pd[0]
                    pd_np = pd.detach().cpu().float().numpy() if isinstance(pd, _torch.Tensor) else _np.asarray(pd)
                    _save_logits(seed_dir_top / "distogram_logits.npz", distogram=pd_np)
                except Exception as _e:
                    logger.warning(f"[bt2 distogram dump] {type(_e).__name__}: {_e}")
            data_path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                from boltz.data.types import StructureV2 as _StructureV2
                struct = _StructureV2.load(data_path)
            else:
                from boltz.data.types import Structure as _Structure
                struct = _Structure.load(data_path)
            chain_map = {}
            for i, mask in enumerate(struct.mask):
                if mask:
                    chain_map[len(chain_map)] = i
            struct = struct.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                model_coord = coord[model_idx]
                coord_unpad = model_coord[pad_mask.bool()].cpu().numpy()
                atoms = struct.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    coord_unpad = _np.array([(x,) for x in coord_unpad], dtype=_Coords)
                residues = struct.residues
                residues["is_present"] = True
                interfaces = _np.array([], dtype=_Interface)
                if self.boltz2:
                    new_struct = _replace(struct, atoms=atoms, residues=residues,
                                          interfaces=interfaces, coords=coord_unpad)
                else:
                    new_struct = _replace(struct, atoms=atoms, residues=residues,
                                          interfaces=interfaces)

                rank = idx_to_rank[model_idx]
                seed_dir = self.output_dir / record.id / f"seed_{_bt_seed}"
                sample_dir = seed_dir / f"sample_{rank}"
                arrays_dir = sample_dir / "arrays"
                arrays_dir.mkdir(parents=True, exist_ok=True)

                # Structure cif (B_iso = pLDDT × 100 via to_mmcif)
                plddts = prediction.get("plddt")
                plddt_atom = plddts[model_idx] if plddts is not None else None
                if self.output_format == "pdb":
                    (sample_dir / "model.pdb").write_text(
                        _to_pdb(new_struct, plddts=plddt_atom, boltz2=self.boltz2)
                    )
                else:
                    cif_path = sample_dir / "model.cif"
                    cif_path.write_text(
                        _to_mmcif(new_struct, plddts=plddt_atom, boltz2=self.boltz2)
                    )
                    if _norm_cif is not None:
                        try:
                            _norm_cif(cif_path)
                        except Exception as _e:
                            logger.warning(f"[bt2 cif normalize] {type(_e).__name__}: {_e}")

                # Arrays. boltz returns pLDDT on a 0–1 scale; convert to the
                # 0–100 scale used by af3 / pt2 / of3 for cross-model parity.
                # (to_mmcif already does ×100 internally for B-factor.)
                # bt2's plddt is per-token (Boltz emits residue-level), so we
                # also write plddt_token.npy as a copy for cross-model parity
                # with pt2/of3.
                if plddt_atom is not None:
                    plddt_atom_100 = plddt_atom * 100
                    _save_npy(arrays_dir / "plddt_atom.npy", plddt_atom_100)
                    _save_npy(arrays_dir / "plddt_token.npy", plddt_atom_100)

                # pae / pde as raw logits (logit_codec int8_softmax) into
                # arrays/{pae,pde}_logits.npz. Logits captured from
                # ConfidenceModule.forward; boltz's predict() drops them in
                # favour of the expected-value scalars.
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(_REPO_ROOT / "scripts"))
                    from logit_codec import save_logits as _save_logits  # type: ignore

                    def _slice(t, idx):
                        if t is None:
                            return None
                        a = t.detach().cpu().float().numpy() if isinstance(t, _torch.Tensor) else _np.asarray(t)
                        if a.shape[0] > idx:
                            return a[idx]
                        return None

                    pae_l = _slice(_BT2_LATEST_HEAD_LOGITS.get("pae_logits"), model_idx)
                    pde_l = _slice(_BT2_LATEST_HEAD_LOGITS.get("pde_logits"), model_idx)
                    if pae_l is not None:
                        _save_logits(arrays_dir / "pae_logits.npz", pae=pae_l)
                    if pde_l is not None:
                        _save_logits(arrays_dir / "pde_logits.npz", pde=pde_l)
                except Exception as _e:
                    logger.warning(f"[bt2 sample logits dump] {type(_e).__name__}: {_e}")

                # contact_probs.npy — bt2 has no native head, derive from the
                # seed-level distogram. Two things must match the conventions
                # used by the other models for cross-model parity:
                #   1) Bin grid: Convention A from the training featurizer.
                #      63 boundaries via linspace(2.0, 22.0, 63); bin 0 is open
                #      below 2.0, bin K-1 open above 22.0. Source:
                #      boltz/data/feature/featurizerv2.py:1409 (Boltz-2),
                #      featurizer.py:793 (Boltz-1).
                #   2) Threshold rule: bin_top <= 8.0 Å (AF3/OF3 convention).
                #      Matches AF3 distogram_head.py:65-74 (`is_contact_bin =
                #      bin_tops <= threshold`) and OF3 confidence.py:63-70
                #      (`distogram_bin_ends <= 8.0`). With BT2's wider step
                #      (0.3226), `bin_top <= 8` includes bins 0..18 (19 bins),
                #      same count as AF3/OF3. A naive `centers < 8` rule would
                #      include bin 19 (center 7.97, top 8.13) → 20 bins, off
                #      by one from AF3/OF3.
                contact_probs_np = None
                pdist_full = prediction.get("pdistogram")
                if pdist_full is not None:
                    try:
                        pd = pdist_full
                        if hasattr(pd, "dim"):
                            if pd.dim() == 5:
                                pd = pd[0, ..., 0, :] if pd.shape[3] == 1 else pd[0, ..., 0, :]
                            elif pd.dim() == 4:
                                pd = pd[0]
                        pd_np = pd.detach().cpu().float().numpy() if isinstance(pd, _torch.Tensor) else _np.asarray(pd)
                        pd_shift = pd_np - pd_np.max(axis=-1, keepdims=True)
                        pp = _np.exp(pd_shift)
                        pp /= pp.sum(axis=-1, keepdims=True)
                        K = pp.shape[-1]
                        boundaries = _np.linspace(2.0, 22.0, K - 1, dtype=_np.float32)
                        step = (22.0 - 2.0) / (K - 2)
                        bin_tops = _np.append(boundaries, boundaries[-1] + step)  # 64 upper edges
                        contact_probs_np = pp[..., bin_tops <= 8.0].sum(axis=-1).astype(_np.float32)
                        _save_npy(arrays_dir / "contact_probs.npy", contact_probs_np)
                    except Exception as _e:
                        logger.warning(f"[bt2 contact_probs] {type(_e).__name__}: {_e}")

                # gpde via cross-model unified formula (AF3 SI 5.7 Eq.16).
                pde_per_sample = (
                    prediction["pde"][model_idx] if "pde" in prediction else None
                )
                gpde_unified = _compute_gpde(pde_per_sample, contact_probs_np)

                # Real atom count from the bt2 atom array (post-mask, post-
                # remove_invalid_chains). plddt_atom in bt2 is per-token, so
                # use len(atoms) instead.
                try:
                    n_atoms_real = int(len(new_struct.atoms))
                except Exception:
                    n_atoms_real = None
                n_tokens_real = (
                    int(plddt_atom.shape[-1]) if plddt_atom is not None else None
                )

                # Summary
                _plddt_scalar = _f(prediction["complex_plddt"][model_idx]) if "complex_plddt" in prediction else None
                summary = {
                    "model_name": "bt2",
                    "seed": _bt_seed,
                    "sample_idx": int(rank),
                    "n_tokens": n_tokens_real,
                    "n_atoms": n_atoms_real,
                    "plddt": _plddt_scalar * 100 if _plddt_scalar is not None else None,
                    "ranking_score": _f(prediction["confidence_score"][model_idx]) if "confidence_score" in prediction else None,
                    "ptm": _f(prediction["ptm"][model_idx]) if "ptm" in prediction else None,
                    "iptm": _f(prediction["iptm"][model_idx]) if "iptm" in prediction else None,
                    "gpde": gpde_unified,
                    "has_clash": None,
                }
                # chain_ptm / chain_pair_iptm / chain_iptm. Boltz internally
                # uses numeric chain ids ("0","1",...); remap to letter keys
                # for cross-model parity. chain_pair_iptm is canonicalized to
                # dict-of-dict, full N×N, diagonal = chain self-pTM.
                if "pair_chains_iptm" in prediction:
                    raw_cpi = prediction["pair_chains_iptm"]
                    raw_keys = sorted(raw_cpi.keys(), key=lambda x: int(x))
                    chain_ptm = {
                        _chain_letter(int(k)): _f(raw_cpi[k][k][model_idx])
                        for k in raw_keys
                    }
                    # Boltz's native pair_chains_iptm is asymmetric (separate
                    # A→B and B→A scores from conditional iPTM). Symmetrize
                    # by taking the off-diagonal mean so the canonical
                    # cpi[A][B] == cpi[B][A] invariant holds. Diagonal kept
                    # as chain self-pTM.
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
                    # chain_iptm: per-chain mean iptm over off-diagonal
                    # entries. null for monomer (1 chain).
                    if len(raw_keys) > 1:
                        chain_iptm = {}
                        for L in chain_ptm:
                            offs = [
                                cpi[L][L2] for L2 in cpi[L]
                                if L2 != L and cpi[L][L2] is not None
                            ]
                            chain_iptm[L] = (
                                float(_np.mean(offs)) if offs else None
                            )
                        summary["chain_iptm"] = chain_iptm
                    else:
                        summary["chain_iptm"] = None
                else:
                    summary["chain_ptm"] = None
                    summary["chain_pair_iptm"] = None
                    summary["chain_iptm"] = None
                # MSA + template provenance: prefer featurizer-time view
                # (inputs_runtime.json) dropped at OUTPUT by _log_input_summary;
                # fall back to lazy yaml lookup if the runtime file isn't present.
                try:
                    import sys as _sys
                    _sys.path.insert(0, "/data/bio-trt/scripts")
                    _sys.path.insert(0, "/gpfs/deepfold/casp/casp17/Bio-TRT/scripts")
                    from inputs_meta import read_runtime_inputs as _rri, lookup_inputs_meta as _lim
                    # Walk up from sample_dir (deeper than self.output_dir):
                    # inputs_runtime.json lives at <self.output_dir>/<record.id>/.
                    rt = _rri(sample_dir)
                    if rt is not None:
                        # Merge in post-pairing stats. The construct_paired_msa
                        # monkey-patch publishes via env-pointed JSON cache
                        # (BT2_PAIRING_JSON) since DataLoader workers fork;
                        # fall back to in-process global.
                        _pair = None
                        _p_json = os.environ.get("BT2_PAIRING_JSON")
                        if _p_json and os.path.isfile(_p_json):
                            try:
                                import json as _bt2_json
                                _pair = _bt2_json.loads(open(_p_json).read())
                            except Exception:
                                _pair = None
                        if _pair is None:
                            try:
                                _pair = _get_bt2_pairing_runtime()
                            except Exception:
                                _pair = None
                        if _pair is not None:
                            rt["pairing"] = _pair
                            try:
                                from inputs_meta import update_runtime_inputs as _uri  # type: ignore
                                _uri(sample_dir, "pairing", _pair)
                            except Exception:
                                pass
                        summary["inputs"] = rt
                    else:
                        _entry = _lim("bt2", record.id)
                        if _entry:
                            summary["inputs"] = _entry
                except Exception as _e:
                    logger.warning(f"[bt2 inputs_meta] {type(_e).__name__}: {_e}")
                # Round all summary floats to 3 decimals for readability.
                def _round3(x):
                    if isinstance(x, dict):  return {k: _round3(v) for k, v in x.items()}
                    if isinstance(x, list):  return [_round3(v) for v in x]
                    if isinstance(x, bool):  return x
                    if isinstance(x, float): return round(x, 3)
                    return x
                _save_json(sample_dir / "summary.json", _round3(summary))
                _save_json(sample_dir / "meta.json", {
                    "writer": "BoltzWriter (unified, monkey-patched)",
                    "model": "bt2",
                    "job_id": record.id,
                    "seed": _bt_seed,
                    "sample_idx": int(rank),
                    "native_sample_idx": int(model_idx),
                })

            if self.write_embeddings and "s" in prediction and "z" in prediction:
                emb_dir = self.output_dir / record.id
                _np.savez_compressed(
                    emb_dir / f"embeddings_{record.id}.npz",
                    s=prediction["s"].cpu().numpy(),
                    z=prediction["z"].cpu().numpy(),
                )

            # timing.json LAST — watcher uses its presence as the
            # "seed done" signal, so writing it before all sample_*/
            # complete races the partial rsync.
            if "runtime_s" in prediction:
                (seed_dir_top / "timing.json").write_text(
                    _json.dumps({"runtime_s": float(prediction["runtime_s"])}, indent=4)
                )

    _bw_mod.BoltzWriter.write_on_batch_end = _unified_write_on_batch_end
    logger.info("BoltzWriter monkey-patched → unified cross-model output layout")


# ── Force AF3 alignment for templates ──────────────────────────────────────
# By default boltz computes its own pairwise alignment per template
# (Bio.Align blastp local), giving qs/qe/template_st/template_en that
# differ from AF3's queryIndices/templateIndices. Monkey-patch
# get_template_records_from_{matching,search} to consult any sidecar
# `<yaml_dir>/<name>.alignments.json` keyed by Path(cif).stem and use
# those values verbatim — keeps cross-model alignment metadata
# byte-identical to af3/pt2/of3 when the source was an AF3 input.json.
def _install_af3_alignment_override():
    import json as _json
    from pathlib import Path as _P
    from boltz.data.parse import schema as _bschema
    from boltz.data.types import TemplateInfo as _TI

    _aligns_cache: dict = {}
    yaml_arg = None
    for i, a in enumerate(sys.argv):
        if a == "predict" and i + 1 < len(sys.argv):
            yaml_arg = sys.argv[i + 1]
            break
    if yaml_arg:
        ya = _P(yaml_arg)
        search_dirs = [ya.parent if ya.is_file() else ya]
        for sd in search_dirs:
            for js in sd.glob("*.alignments.json"):
                try:
                    _aligns_cache.update(_json.loads(js.read_text()))
                except Exception:
                    pass
    if not _aligns_cache:
        return

    def _override(template_id, chain_ids, sequences,
                  template_chain_ids, template_sequences,
                  force=False, threshold=None):
        rec = _aligns_cache.get(template_id)
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
    logger.info(f"[bt2] AF3 alignment override active for {len(_aligns_cache)} templates")


_install_af3_alignment_override()
_install_unified_boltz_writer()

# Constraint logger: prints parsed contact/pocket constraints (chain-resolved)
# to stderr right after boltz parses the YAML, so inference.log shows what
# the model actually consumed. Implementation in scripts/boltz_unified_jax.py
# (jax-suffix but the patch is backend-agnostic — pure schema-parse hook).
try:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from boltz_unified_jax import (
        install_constraint_logger as _bt2_install_constraint_logger,
        install_runtime_feats_logger as _bt2_install_runtime_feats_logger,
        install_template_chunking as _bt2_install_template_chunking,
        install_stage_profiler as _bt2_install_stage_profiler,
        install_torch_compile as _bt2_install_torch_compile,
    )
    _bt2_install_constraint_logger()
    _bt2_install_runtime_feats_logger()
    # BT2_TORCH_COMPILE=score_model[,pairformer,…] → torch.compile after load (no-op otherwise).
    _bt2_install_torch_compile()
    # Per-template loop in TemplateV2Module: peak O(T·N²) → O(N²). Backend-agnostic
    # (pure torch reordering of a linear sum); this was the 1627-token OOM on the
    # torch/TRT path, which the JAX entry had already fixed. BIO_TRT_BT2_TEMPLATE_CHUNK=0 disables.
    _bt2_install_template_chunking()
    # LANTERN_ENGINE_PROFILE=1 → per-stage timing/peak-memory JSON (see install_stage_profiler).
    _bt2_install_stage_profiler()
except Exception as _e:
    logger.warning(f"[bt2] shared patch install failed: {type(_e).__name__}: {_e}")


# ── Multi-seed-one-load wrapper around pl.Trainer.predict ─────────────────
# Boltz's CLI takes a scalar --seed, so a normal multi-seed run requires N
# pueue tasks each loading the model from scratch. With BT2_SEEDS=s1,s2,…
# set, we wrap Trainer.predict to internally loop seeds; the loaded
# Boltz2Model is reused, and the unified writer's _bt_seed_now() reads
# BT2_SEED freshly each call so outputs land in seed_<N>/ subdirs.
def _install_multiseed_trainer_predict():
    seeds_env = os.environ.get("BT2_SEEDS", "").strip()
    if not seeds_env:
        return
    try:
        seeds = [int(s) for s in seeds_env.split(",") if s.strip()]
    except ValueError:
        logger.warning(f"[bt2 multi-seed] BT2_SEEDS={seeds_env!r} not all-int; ignoring")
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

    def _multiseed_predict(self, *args, **kwargs):
        import gc as _gc
        result = None
        for s in seeds:
            print(f"[bt2-multiseed] === running seed={s} ===", flush=True)
            os.environ["BT2_SEED"] = str(s)
            if _se is not None:
                _se(int(s), workers=True)
            else:
                _torch.manual_seed(int(s))
            result = _orig_predict(self, *args, **kwargs)
            # Drop the previous seed's predictions and free GPU memory
            # before the next seed runs. Without this, Lightning Trainer
            # internal state + cached tensors accumulate across the 1024
            # seed loop and OOM the pod.
            del result
            result = None
            _gc.collect()
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass
        return result

    _Trainer.predict = _multiseed_predict
    logger.info(f"[bt2 multi-seed] active for seeds={seeds}; one model load")


_install_multiseed_trainer_predict()


# Delegate to boltz CLI
from boltz.main import cli

if __name__ == "__main__":
    cli()
