# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys as _sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from biotite.structure import AtomArray

from protenix.data.utils import save_structure_cif


def _load_cif_normalize():
    """Import `cif_normalize.normalize_model_cif` from core/scripts/.

    residudes-engine's `scripts/` dir is located relative to this module
    (`models/protenix/runner/dumper.py` → ../../../scripts), so the runner
    finds the helper regardless of where the repo is checked out.
    """
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    try:
        from cif_normalize import normalize_model_cif  # type: ignore
        return normalize_model_cif
    except Exception:
        return None


_cif_norm = _load_cif_normalize()


def _normalize_model_cif(p) -> None:
    if _cif_norm is not None:
        try:
            _cif_norm(p)
        except Exception:
            pass


def _compute_gpde(pde, contact_probs) -> Optional[float]:
    """AF3 SI 5.7 Eq.16 — Σ pde·cp / Σ cp over all token pairs (incl. diag).
    Identical formula across pt2/of3/bt2/af3 writers."""
    pde_arr = _to_cpu_numpy(pde)
    cp_arr = _to_cpu_numpy(contact_probs)
    if pde_arr is None or cp_arr is None:
        return None
    pde_arr = pde_arr.astype(np.float32, copy=False)
    cp_arr = cp_arr.astype(np.float32, copy=False)
    if pde_arr.shape != cp_arr.shape or pde_arr.ndim < 2:
        return None
    s = float(cp_arr.sum())
    if s <= 0:
        return None
    return float((pde_arr * cp_arr).sum() / s)


def _atom_to_token_from_atom_array(atom_array: AtomArray) -> Optional[np.ndarray]:
    """Build (chain_id, res_id) → token-index mapping in atom-array order."""
    try:
        chain_ids = list(atom_array.chain_id)
        res_ids = list(atom_array.res_id)
    except Exception:
        return None
    token_idx = []
    seen: dict = {}
    for c, r in zip(chain_ids, res_ids):
        key = (c, int(r))
        if key not in seen:
            seen[key] = len(seen)
        token_idx.append(seen[key])
    return np.asarray(token_idx, dtype=np.int32)


def _aggregate_atom_to_token(atom_arr: np.ndarray, atom_to_token: np.ndarray) -> np.ndarray:
    """Mean-aggregate per-atom array into per-token array."""
    n_token = int(atom_to_token.max()) + 1
    out = np.zeros(n_token, dtype=np.float32)
    counts = np.zeros(n_token, dtype=np.int32)
    np.add.at(out, atom_to_token, atom_arr.astype(np.float32))
    np.add.at(counts, atom_to_token, 1)
    return out / np.maximum(counts, 1)


def _to_cpu_numpy(t) -> Optional[np.ndarray]:
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t.astype(np.float32, copy=False) if t.dtype.kind == "f" else t
    try:
        if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
            t = t.to(torch.float32)
        return t.detach().cpu().numpy()
    except Exception:
        try:
            return t.cpu().numpy()
        except Exception:
            return None


def _save_npy(p: Path, arr) -> None:
    arr = _to_cpu_numpy(arr)
    if arr is None:
        return
    if arr.dtype.kind == "f" and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def _json_default(x):
    # Handle torch / numpy scalars + arrays + nested torch dicts.
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def _save_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, default=_json_default))


# ── MSA / template provenance extraction ──────────────────────────────────

def _count_a3m_records(p: Optional[str]) -> int:
    """Count `>` headers in an a3m file. Returns 0 on missing/error."""
    if not p:
        return 0
    try:
        with open(p) as f:
            return sum(1 for ln in f if ln.startswith(">"))
    except OSError:
        return 0


def _parse_hmmsearch_a3m_hits(p: Optional[str]) -> list[dict]:
    """Extract per-hit metadata from a hmmsearch a3m. Delegates to
    inputs_meta.parse_hmmsearch_a3m_hits so dumper + post_processing share
    a single regex (lazy `.*?` made the local copy drop length:N)."""
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    try:
        from inputs_meta import parse_hmmsearch_a3m_hits  # type: ignore
    except Exception:
        return []
    return parse_hmmsearch_a3m_hits(p)


def build_pt2_inputs_meta_from_json(input_json_path: Optional[str]) -> dict[str, dict]:
    """For each protenix job (top-level list entry with proteinChain
    sequences), return a dict keyed by the job's `name`:

        {
          "<job_name>": {
              "msa":       {"<chain>": {unpaired, paired, depth_unpaired, depth_paired}},
              "templates": {"<chain>": [{pdb_chain, qs, qe, length}, ...]}
          }
        }
    """
    if not input_json_path:
        return {}
    try:
        data = json.loads(open(input_json_path).read())
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        data = [data]
    out: dict[str, dict] = {}
    for job in data:
        name = job.get("name") or "<unnamed>"
        msa: dict = {}
        tpls: dict = {}
        for entry in (job.get("sequences") or []):
            pc = entry.get("proteinChain") or entry.get("protein")
            if not pc:
                continue
            cid = pc.get("id")
            if isinstance(cid, list):
                key = cid[0] if cid else "?"
            else:
                key = cid or "?"
            unp = pc.get("unpairedMsaPath")
            ppm = pc.get("pairedMsaPath")
            tpath = pc.get("templatesPath")
            msa[key] = {
                "unpaired": unp,
                "paired":   ppm,
                "depth_unpaired": _count_a3m_records(unp),
                "depth_paired":   _count_a3m_records(ppm),
            }
            tpls[key] = _parse_hmmsearch_a3m_hits(tpath)
        out[name] = {"msa": msa, "templates": tpls}
    return out


class DataDumper:
    """Write predictions in the unified cross-model layout.

    For each sample, emits:
        <base_dir>/<dataset_name>/<sample_name>/seed_<seed>/sample_<rank>/
            model.cif          # biotite CIF, B_iso_or_equiv = pLDDT
            summary.json       # cross-model schema (plddt, ptm, iptm, ...)
            meta.json          # provenance
            arrays/
                plddt_atom.npy
                plddt_token.npy        (derived via atom→token mean)
                pae.npy
                pde.npy
                contact_probs.npy
                distogram_probs.npy    (uint8, log-quantized; decoded via
                                        inferences/_lib/unify_output.py
                                        _dequantize_distogram_uint8)
                distance_std.npy

    No `predictions/` subdir, no native JSON/NPZ intermediate.
    """

    MODEL_NAME = "pt2"

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,  # kept for compat (unused)
        sorted_by_ranking_score: bool = True,
        inputs_meta: Optional[dict] = None,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score
        # `inputs_meta` is keyed by job name (== sample_name == pdb_id at dump
        # time). Each value: {"msa": {chain: {...}}, "templates": {chain: [...]}}
        # When present, the per-chain MSA + template provenance is included
        # in summary.json so downstream comparisons can audit which inputs
        # produced each prediction.
        self.inputs_meta: dict = inputs_meta or {}

    # ── Public API ────────────────────────────────────────────────────────

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ) -> None:
        dump_dir = Path(self._get_dump_dir(dataset_name, pdb_id, seed))
        dump_dir.mkdir(parents=True, exist_ok=True)

        sorted_indices = self._get_ranker_indices(pred_dict)

        # Pre-compute b_factor list (atom_plddt × 100) for cif writer.
        b_factor_list = self._extract_atom_plddt_per_sample(pred_dict)

        # Distogram is sample-independent; write once at the seed dir.
        self._write_seed_distogram(dump_dir, pred_dict)

        n_samples = pred_dict["coordinate"].shape[0]
        for sample_idx in range(n_samples):
            rank = sorted_indices[sample_idx] if sorted_indices else sample_idx
            sample_dir = dump_dir / f"sample_{rank}"
            arrays_dir = sample_dir / "arrays"
            arrays_dir.mkdir(parents=True, exist_ok=True)

            # 1) Structure CIF (biotite). B_iso_or_equiv = pLDDT × 100.
            self._write_cif(
                sample_dir / "model.cif",
                atom_array=atom_array,
                pred_coord=pred_dict["coordinate"][sample_idx],
                entity_poly_type=entity_poly_type,
                pdb_id=pdb_id,
                b_factor=(
                    b_factor_list[sample_idx]
                    if b_factor_list is not None
                    else None
                ),
            )

            # 2) Arrays (npy) + per-sample logit heads.
            atom_to_token = _atom_to_token_from_atom_array(atom_array)
            self._write_arrays(
                arrays_dir,
                pred_dict=pred_dict,
                sample_idx=sample_idx,
                atom_to_token=atom_to_token,
            )
            self._write_sample_heads(sample_dir, pred_dict, sample_idx)

            # 3) Summary + meta.
            summary = self._build_summary(
                pred_dict=pred_dict,
                sample_idx=sample_idx,
                rank=rank,
                seed=seed,
                atom_array=atom_array,
            )
            # Attach featurizer-time inputs view (post-cap depths + selected
            # template hits). Prefer runtime.json dropped at job dump_dir
            # by inference.py and merge MSA depths from PT2_MSA_DEPTH_JSON
            # cache (written by msa_featurizer's post-load log line) so the
            # depths reflect what the model actually consumed, not raw a3m
            # record counts.
            try:
                from inputs_meta import read_runtime_inputs as _rri  # type: ignore
                rt = _rri(dump_dir)
            except Exception:
                rt = None
            if rt is not None:
                # Override per-chain depths with featurizer-time values.
                msa_depth_path = os.environ.get("PT2_MSA_DEPTH_JSON")
                merged = False
                if msa_depth_path and os.path.isfile(msa_depth_path):
                    try:
                        depths = json.loads(open(msa_depth_path).read())
                        for cid, d in depths.items():
                            chain = (rt.get("msa") or {}).get(cid)
                            if chain is None:
                                continue
                            if chain.get("unpaired") is not None and d.get("unpaired_depth") is not None:
                                chain["unpaired"]["depth"] = d["unpaired_depth"]
                                merged = True
                            if chain.get("paired") is not None and d.get("paired_depth") is not None:
                                chain["paired"]["depth"] = d["paired_depth"]
                                merged = True
                    except Exception:
                        pass
                # No per-seed inputs_runtime.json copy: the unified policy
                # is one file at <target>/<variant>/<model>/, not per-seed.
                # Merge in post-pairing stats. DataLoader workers fork before
                # main, so pairing function publishes via env-pointed JSON
                # cache (PT2_PAIRING_JSON); fall back to in-process global
                # for non-forked deployments.
                _pair = None
                _p_json = os.environ.get("PT2_PAIRING_JSON")
                if _p_json and os.path.isfile(_p_json):
                    try:
                        _pair = json.loads(open(_p_json).read())
                    except Exception:
                        _pair = None
                if _pair is None:
                    try:
                        from protenix.data.msa.msa_utils import get_pairing_runtime as _gpr  # type: ignore
                        _pair = _gpr()
                    except Exception:
                        _pair = None
                if _pair is not None:
                    rt["pairing"] = _pair
                    try:
                        from inputs_meta import update_runtime_inputs as _uri  # type: ignore
                        _uri(dump_dir, "pairing", _pair)
                    except Exception:
                        pass
                summary["inputs"] = rt
            else:
                job_meta = self.inputs_meta.get(pdb_id)
                if job_meta:
                    summary["inputs"] = job_meta
            # Round all summary floats to 3 decimals for readability.
            def _round3(x):
                if isinstance(x, dict):  return {k: _round3(v) for k, v in x.items()}
                if isinstance(x, list):  return [_round3(v) for v in x]
                if isinstance(x, bool):  return x
                if isinstance(x, float): return round(x, 3)
                return x
            _save_json(sample_dir / "summary.json", _round3(summary))
            _save_json(sample_dir / "meta.json", {
                "writer": "DataDumper (unified)",
                "model": self.MODEL_NAME,
                "job_id": pdb_id,
                "seed": seed,
                "sample_idx": rank,
                "native_sample_idx": sample_idx,
            })

        # timing.json LAST — watcher uses its presence as the "seed done"
        # signal, so it must appear only after every sample_*/ has its
        # model.cif + arrays/ written. Otherwise watcher's rsync races
        # the per-sample writes (ends up with partial seed dirs locally).
        runtime_s = pred_dict.get("_runtime_s")
        if runtime_s is not None:
            (dump_dir / "timing.json").write_text(
                json.dumps({"runtime_s": float(runtime_s)}, indent=4)
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        return os.path.join(self.base_dir, dataset_name, sample_name, f"seed_{seed}")

    def _get_ranker_indices(self, data: dict) -> List[int]:
        n = len(data.get("summary_confidence", []))
        if n == 0 or not self.sorted_by_ranking_score:
            return list(range(n))
        scores = torch.tensor([
            data["summary_confidence"][i].get("ranking_score", 0.0)
            for i in range(n)
        ])
        # rank: position of i in descending-by-score order
        return [int(x) for x in torch.argsort(torch.argsort(scores, descending=True))]

    def _extract_atom_plddt_per_sample(self, pred_dict: dict) -> Optional[List[np.ndarray]]:
        if "full_data" not in pred_dict:
            return None
        out = []
        for entry in pred_dict["full_data"]:
            ap = entry.get("atom_plddt")
            if ap is None:
                return None
            arr = _to_cpu_numpy(ap)
            if arr is None:
                return None
            out.append(arr * 100.0)
        return out

    def _write_cif(
        self,
        out_path: Path,
        atom_array: AtomArray,
        pred_coord: torch.Tensor,
        entity_poly_type: dict[str, str],
        pdb_id: str,
        b_factor: Optional[np.ndarray],
    ) -> None:
        if b_factor is not None:
            atom_array.set_annotation("b_factor", np.round(b_factor, 2))
        save_structure_cif(
            atom_array=atom_array,
            pred_coordinate=pred_coord,
            output_fpath=str(out_path),
            entity_poly_type=entity_poly_type,
            pdb_id=pdb_id,
        )
        _normalize_model_cif(out_path)

    def _write_arrays(
        self,
        arrays_dir: Path,
        pred_dict: dict,
        sample_idx: int,
        atom_to_token: Optional[np.ndarray],
    ) -> None:
        fd = pred_dict.get("full_data", [None] * (sample_idx + 1))
        sample_fd = fd[sample_idx] if sample_idx < len(fd) else None
        if sample_fd is None:
            return

        atom_plddt = _to_cpu_numpy(sample_fd.get("atom_plddt"))
        if atom_plddt is not None:
            # Protenix returns atom_plddt on a 0–1 scale; convert to 0–100 to
            # match af3 / bt2 / of3 (and the cif's B_iso column).
            atom_plddt_100 = atom_plddt * 100.0
            _save_npy(arrays_dir / "plddt_atom.npy", atom_plddt_100)
            if atom_to_token is not None and atom_to_token.shape[0] == atom_plddt.shape[-1]:
                _save_npy(
                    arrays_dir / "plddt_token.npy",
                    _aggregate_atom_to_token(atom_plddt_100, atom_to_token),
                )

        # contact_probs only — pae/pde now go through logit_codec as
        # arrays/{pae,pde}_logits.npz (see _write_sample_logits below).
        v = sample_fd.get("contact_probs")
        if v is not None:
            _save_npy(arrays_dir / "contact_probs.npy", v)

    def _write_seed_distogram(self, dump_dir: Path, pred_dict: dict) -> None:
        """Distogram logits → <seed_dir>/distogram_logits.npz (sample-independent)."""
        try:
            import sys as _sys
            _scripts = str(Path(__file__).resolve().parents[3] / "scripts")
            if _scripts not in _sys.path:
                _sys.path.insert(0, _scripts)
            from logit_codec import save_logits as _save_logits  # type: ignore

            dl = pred_dict.get("distogram")
            if dl is None:
                return
            if hasattr(dl, "dim"):
                if dl.dim() == 5:           # [B, S, N, N, K] → take sample 0
                    dl = dl[0, 0]
                elif dl.dim() == 4:         # [B, N, N, K]
                    dl = dl[0]
            arr = _to_cpu_numpy(dl)
            if arr is not None:
                _save_logits(dump_dir / "distogram_logits.npz", distogram=arr)
        except Exception as e:
            import sys
            print(f"[pt2 distogram dump fail] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    def _write_sample_heads(self, sample_dir: Path, pred_dict: dict, sample_idx: int) -> None:
        """Per-sample pae/pde logits → <sample_dir>/arrays/{pae,pde}_logits.npz.
        plddt logits are skipped (atom plddt expected-value lives at
        arrays/plddt_atom.npy already)."""
        try:
            import sys as _sys
            _scripts = str(Path(__file__).resolve().parents[3] / "scripts")
            if _scripts not in _sys.path:
                _sys.path.insert(0, _scripts)
            from logit_codec import save_logits as _save_logits  # type: ignore

            def _slice(t):
                """Pull this sample's logits from a [B, S, ...] tensor."""
                if t is None:
                    return None
                arr = _to_cpu_numpy(t)
                if arr is None:
                    return None
                # protenix lays out as [B, S, ...] → take batch 0, sample sample_idx
                if arr.ndim >= 2 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.shape[0] > sample_idx:
                    arr = arr[sample_idx]
                return arr

            arrays_dir = sample_dir / "arrays"
            arrays_dir.mkdir(parents=True, exist_ok=True)
            pae_l = _slice(pred_dict.get("pae"))
            pde_l = _slice(pred_dict.get("pde"))
            if pae_l is not None:
                _save_logits(arrays_dir / "pae_logits.npz", pae=pae_l)
            if pde_l is not None:
                _save_logits(arrays_dir / "pde_logits.npz", pde=pde_l)
        except Exception as e:
            import sys
            print(f"[pt2 sample logits dump fail] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    def _build_summary(
        self,
        pred_dict: dict,
        sample_idx: int,
        rank: int,
        seed: int,
        atom_array: AtomArray,
    ) -> dict:
        s = (pred_dict.get("summary_confidence") or [])[sample_idx] or {}
        # Number of atoms / tokens (best-effort).
        n_atoms = None
        try:
            n_atoms = int(len(atom_array))
        except Exception:
            pass
        n_tokens = None
        cp = (pred_dict.get("full_data") or [None])[sample_idx]
        if cp is not None and isinstance(cp.get("contact_probs", None), torch.Tensor):
            n_tokens = int(cp["contact_probs"].shape[-1])

        # gpde via cross-model unified formula (AF3 SI 5.7 Eq.16).
        gpde_unified = _compute_gpde(
            cp.get("token_pair_pde") if cp is not None else None,
            cp.get("contact_probs") if cp is not None else None,
        )

        def _f(x):
            try:
                return float(x) if x is not None else None
            except Exception:
                return None

        # Canonicalize chain_*/chain_pair_iptm to letter-keyed dicts. pt2's
        # natives are positional Tensors:
        #   chain_ptm:        scalar (monomer) or 1-D tensor (multimer)
        #   chain_iptm:       scalar (monomer = 0) or 1-D tensor (multimer)
        #   chain_pair_iptm:  scalar (monomer = 0) or 2-D tensor with diag=0
        # Chain letter order = unique atom_array.chain_id in token order.
        try:
            seen_chains: list[str] = []
            for c in atom_array.chain_id:
                cs = str(c)
                if cs not in seen_chains:
                    seen_chains.append(cs)
        except Exception:
            seen_chains = []

        def _to_list(v):
            if v is None:
                return None
            if torch.is_tensor(v):
                v = v.detach().cpu().tolist()
            if isinstance(v, np.ndarray):
                v = v.tolist()
            return v

        cptm_raw = _to_list(s.get("chain_ptm"))
        ciptm_raw = _to_list(s.get("chain_iptm"))
        cpi_raw = _to_list(s.get("chain_pair_iptm"))

        if not isinstance(cptm_raw, list):
            cptm_raw = [cptm_raw] if cptm_raw is not None else []
        if not isinstance(ciptm_raw, list):
            ciptm_raw = [ciptm_raw] if ciptm_raw is not None else []

        n = len(seen_chains)
        chain_ptm = {seen_chains[i]: _f(cptm_raw[i])
                     for i in range(min(n, len(cptm_raw)))}
        # Monomer: pt2 emits chain_iptm = 0; canonical = null.
        if n > 1 and len(ciptm_raw) == n:
            chain_iptm = {seen_chains[i]: _f(ciptm_raw[i]) for i in range(n)}
        else:
            chain_iptm = None

        cpi_canon = None
        if n == 1:
            cpi_canon = {seen_chains[0]: {seen_chains[0]: chain_ptm.get(seen_chains[0])}}
        elif (
            n > 1
            and isinstance(cpi_raw, list)
            and len(cpi_raw) == n
            and all(isinstance(r, list) and len(r) == n for r in cpi_raw)
        ):
            cpi_canon = {
                seen_chains[i]: {
                    seen_chains[j]: (
                        chain_ptm.get(seen_chains[i]) if i == j
                        else _f(cpi_raw[i][j])
                    )
                    for j in range(n)
                }
                for i in range(n)
            }

        return {
            "model_name": self.MODEL_NAME,
            "seed": seed,
            "sample_idx": rank,
            "n_tokens": n_tokens,
            "n_atoms": n_atoms,
            "plddt": _f(s.get("plddt")),
            "gpde": gpde_unified,
            "ptm": _f(s.get("ptm")),
            "iptm": _f(s.get("iptm")),
            "has_clash": bool(s["has_clash"]) if s.get("has_clash") is not None else None,
            "ranking_score": _f(s.get("ranking_score")),
            "chain_ptm": chain_ptm or None,
            "chain_iptm": chain_iptm,
            "chain_pair_iptm": cpi_canon,
        }
