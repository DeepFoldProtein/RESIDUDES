# Copyright 2026 AlQuraishi Laboratory
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

"""A module for containing writing tools and callbacks for model outputs.

The output writer emits a cross-model **unified** layout:

    <output_dir>/<query_id>/seed_<seed>/sample_<idx>/   (idx is 0-indexed)
        model.cif
        summary.json   (cross-model schema)
        meta.json
        arrays/
            plddt_atom.npy
            plddt_token.npy        (mean over atoms-in-residue)
            pae.npy                (token×token)
            pde.npy                (token×token)
            contact_probs.npy      (token×token)
            distogram_probs.npy    (uint8, log-quantized; decompress via
                                    inferences/_lib/unify_output.py
                                    _dequantize_distogram_uint8)
            distance_std.npy       (token×token, σ of model's distance dist.)

Native per-atom `_confidences.json` (O(N²) and ~3-4 GB for ntok≥500) is
NOT written — the same data is captured in arrays/ as quantized npy.
"""

import json
import logging
import sys as _sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.utils.tensor_utils import tensor_tree_map


def _load_cif_normalize():
    """Import cif_normalize.normalize_model_cif from core/scripts/.

    residudes-engine's scripts/ dir is located relative to this module
    (models/openfold-3/openfold3/core/runners/writer.py → 5 levels up).
    """
    scripts_dir = str(Path(__file__).resolve().parents[5] / "scripts")
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

logger = logging.getLogger(__name__)


def _compute_gpde(pde, contact_probs) -> Optional[float]:
    """AF3 SI 5.7 Eq.16 — Σ pde·cp / Σ cp over all token pairs (incl. diag).
    Identical formula across pt2/of3/bt2/af3 writers."""
    if pde is None or contact_probs is None:
        return None
    pde_arr = np.asarray(pde, dtype=np.float32) if not isinstance(pde, np.ndarray) else pde.astype(np.float32, copy=False)
    cp_arr = np.asarray(contact_probs, dtype=np.float32) if not isinstance(contact_probs, np.ndarray) else contact_probs.astype(np.float32, copy=False)
    if pde_arr.shape != cp_arr.shape or pde_arr.ndim < 2:
        return None
    s = float(cp_arr.sum())
    if s <= 0:
        return None
    return float((pde_arr * cp_arr).sum() / s)


def _atom_to_token_from_atom_array(atom_array) -> Optional[np.ndarray]:
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
    n_token = int(atom_to_token.max()) + 1
    out = np.zeros(n_token, dtype=np.float32)
    counts = np.zeros(n_token, dtype=np.int32)
    np.add.at(out, atom_to_token, atom_arr.astype(np.float32))
    np.add.at(counts, atom_to_token, 1)
    return out / np.maximum(counts, 1)


def _save_npy(p: Path, arr) -> None:
    if arr is None:
        return
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype.kind == "f" and arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)


def _save_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, cls=NumpyEncoder))


def _to_python(x):
    if x is None:
        return None
    if isinstance(x, (np.ndarray,)):
        return x.tolist() if x.ndim > 0 else x.item()
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "item") and not isinstance(x, (str, bytes, list, tuple, dict)):
        try:
            return x.item()
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return x


class NumpyEncoder(json.JSONEncoder):
    r"""Custom JSON encoder for handling numpy data types.

    https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _take_batch_dim(x, b: int):
    if isinstance(x, torch.Tensor):
        if len(x.shape) > 1:
            return x[b].cpu().float().numpy()
        else:
            return x
    if isinstance(x, dict):
        return {k: _take_batch_dim(v, b) for k, v in x.items()}
    return x


def _take_sample_dim(x, s: int):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.shape[0] == 1:
            return x[0]
        return x[s]
    if isinstance(x, dict):
        return {k: _take_sample_dim(v, s) for k, v in x.items()}
    return x


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(
        self,
        output_dir: Path,
        structure_format: str = "pdb",
        full_confidence_output_format: str = "json",
        write_features: bool = False,
        write_latent_outputs: bool = False,
        write_full_confidence_scores: bool = True,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_output_format
        self.write_features = write_features
        self.write_latent_outputs = write_latent_outputs
        self.write_full_confidence_scores = write_full_confidence_scores

        # Track successfully predicted samples
        self.success_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.failed_queries = []

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
        make_ost_compatible: bool = True,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(
            atom_array,
            output_file,
            include_bonds=True,
            make_ost_compatible=make_ost_compatible,
        )

    def get_pae_confidence_scores(self, confidence_scores, atom_array):
        pae_confidence_scores = {}
        single_value_keys = [
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
        ]

        for key in single_value_keys:
            pae_confidence_scores[key] = confidence_scores[key]

        # Get map from asym id to chain id
        renum_ids = np.unique(atom_array.chain_id, return_inverse=True)[1] + 1
        asym_id_to_chain_id = {
            k: v
            for (k, v) in set(
                [
                    (int(x[0]), str(x[1]))
                    for x in zip(renum_ids, atom_array.chain_id, strict=True)
                ]
            )
        }

        # Asym id -> chain id for chain_ptm
        pae_confidence_scores["chain_ptm"] = {
            asym_id_to_chain_id[int(k)]: v
            for k, v in confidence_scores["chain_ptm"].items()
        }

        # Asym id -> chain id for chain_pair_iptm
        pae_confidence_scores["chain_pair_iptm"] = {}
        for k, v in confidence_scores["chain_pair_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["chain_pair_iptm"][f"({k1}, {k2})"] = v

        # Asym id -> chain id for bespoke_iptm
        pae_confidence_scores["bespoke_iptm"] = {}
        for k, v in confidence_scores["bespoke_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["bespoke_iptm"][f"({k1}, {k2})"] = v
        return pae_confidence_scores

    def write_unified_sample(
        self,
        sample_dir: Path,
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        confidence_scores: dict,
        seed: int,
        sample_idx: int,
        query_id: str,
        native_sample_idx: Optional[int] = None,
    ) -> None:
        """Write one sample in the unified cross-model layout.

        Replaces the legacy per-sample _model.cif + _confidences.json +
        _confidences_aggregated.json triplet.  Avoids serialising the huge
        per-atom JSON entirely; the same arrays land directly in arrays/*.npy.
        """
        sample_dir.mkdir(parents=True, exist_ok=True)
        arrays_dir = sample_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        # 1) Structure cif (B_iso = pLDDT × 100 in the existing convention).
        plddt_atom = confidence_scores["plddt"]
        cif_path = sample_dir / f"model.{self.structure_format}"
        self.write_structure_prediction(
            atom_array=atom_array,
            predicted_coords=predicted_coords,
            plddt=plddt_atom,
            output_file=cif_path,
        )
        _normalize_model_cif(cif_path)

        # 2) Arrays npy.
        _save_npy(arrays_dir / "plddt_atom.npy", plddt_atom)

        atom_to_token = _atom_to_token_from_atom_array(atom_array)
        if atom_to_token is not None and atom_to_token.shape[0] == np.asarray(plddt_atom).shape[-1]:
            _save_npy(
                arrays_dir / "plddt_token.npy",
                _aggregate_atom_to_token(np.asarray(plddt_atom), atom_to_token),
            )

        # contact_probs only — pae/pde now go through logit_codec as
        # arrays/{pae,pde}_logits.npz alongside seed-level distogram_logits.npz.
        v = confidence_scores.get("contact_probs")
        if v is not None:
            _save_npy(arrays_dir / "contact_probs.npy", np.asarray(v))

        # Heads as raw logits (logit_codec int8_softmax, ~1e-4 KL recovery):
        #   <seed_dir>/distogram_logits.npz       (sample-independent)
        #   <sample_dir>/arrays/pae_logits.npz    (per-sample)
        #   <sample_dir>/arrays/pde_logits.npz    (per-sample)
        try:
            import sys as _sys
            _scripts = str(Path(__file__).resolve().parents[5] / "scripts")
            if _scripts not in _sys.path:
                _sys.path.insert(0, _scripts)
            from logit_codec import save_logits as _save_logits  # type: ignore

            def _to_np(t):
                if t is None:
                    return None
                arr = np.asarray(t.cpu() if hasattr(t, "cpu") else t)
                while arr.ndim > 4 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr

            # Seed-level distogram, written once at sample_idx == 0.
            if int(sample_idx) == 0:
                d = _to_np(confidence_scores.get("distogram_logits"))
                if d is not None:
                    seed_dir = sample_dir.parent
                    _save_logits(seed_dir / "distogram_logits.npz", distogram=d)

            pae_l = _to_np(confidence_scores.get("pae_logits"))
            pde_l = _to_np(confidence_scores.get("pde_logits"))
            if pae_l is not None:
                _save_logits(arrays_dir / "pae_logits.npz", pae=pae_l)
            if pde_l is not None:
                _save_logits(arrays_dir / "pde_logits.npz", pde=pde_l)
        except Exception as e:
            logger.warning(f"[of3 logits dump fail] {type(e).__name__}: {e}")

        # 3) summary.json (cross-model schema).
        summary = {
            "model_name": "of3",
            "seed": int(seed) if seed is not None else None,
            "sample_idx": int(sample_idx),
            "n_tokens": int(atom_to_token.max()) + 1 if atom_to_token is not None else None,
            "n_atoms": int(len(atom_array)),
            "plddt": float(np.mean(plddt_atom)),
            # Re-derived from saved arrays — AF3 SI 5.7 Eq.16 unified across
            # af3/pt2/of3/bt2 (Σ pde·cp / Σ cp over all token pairs).
            "gpde": _compute_gpde(
                confidence_scores.get("pde"),
                confidence_scores.get("contact_probs"),
            ),
            "ptm": _to_python(confidence_scores.get("ptm")),
            "iptm": _to_python(confidence_scores.get("iptm")),
            "has_clash": bool(confidence_scores["has_clash"])
                if confidence_scores.get("has_clash") is not None else None,
            "ranking_score": _to_python(confidence_scores.get("sample_ranking_score")),
        }
        # Asym-id → chain-id remapped chain fields (uses existing helper).
        try:
            chain_dict = self.get_pae_confidence_scores(confidence_scores, atom_array)
            chain_ptm = chain_dict.get("chain_ptm") or {}
            summary["chain_ptm"] = chain_ptm or None
            # Native of3 chain_pair_iptm is a dict with "(A, B)" keys for
            # off-diagonal pairs only. Canonicalize to dict-of-dict with
            # letter keys, full N×N, diagonal = chain self-pTM.
            native_cpi = chain_dict.get("chain_pair_iptm") or {}
            chain_keys = sorted(chain_ptm.keys())
            cpi_canon = {c: {c2: None for c2 in chain_keys} for c in chain_keys}
            for c in chain_keys:
                cpi_canon[c][c] = float(chain_ptm[c]) if chain_ptm.get(c) is not None else None
            for k, v in native_cpi.items():
                k1, k2 = [x.strip() for x in k.strip("()").split(",")]
                if k1 in cpi_canon and k2 in cpi_canon[k1]:
                    cpi_canon[k1][k2] = float(v)
                    cpi_canon[k2][k1] = float(v)
            summary["chain_pair_iptm"] = cpi_canon if chain_keys else None
            # Aggregate per-chain mean iptm (excl. self) from off-diagonal.
            per_chain = {}
            for k, v in native_cpi.items():
                k1, k2 = [x.strip() for x in k.strip("()").split(",")]
                if k1 == k2:
                    continue
                per_chain.setdefault(k1, []).append(float(v))
                per_chain.setdefault(k2, []).append(float(v))
            summary["chain_iptm"] = (
                {c: float(np.mean(vs)) for c, vs in per_chain.items()} or None
            )
        except Exception:
            summary["chain_ptm"] = None
            summary["chain_pair_iptm"] = None
            summary["chain_iptm"] = None

        # MSA + template provenance: prefer featurizer-time view dropped
        # by single_datasets/inference.py at $OF3_OUTPUT_DIR; fall back to
        # input.json provenance when the runtime file isn't present.
        try:
            import os as _os, sys as _sys
            _sys.path.insert(0, _os.path.join(
                _os.environ.get("RESIDUDES_ENGINE_CORE")
                or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[_os.pardir] * 5)), "scripts"))
            from inputs_meta import read_runtime_inputs as _rri, lookup_inputs_meta as _lim
            rt = _rri(sample_dir)
            if rt is not None:
                # Merge in post-pairing stats. DataLoader workers fork before
                # main, so pairing function publishes via env-pointed JSON
                # cache (OF3_PAIRING_JSON); fall back to in-process global.
                _pair = None
                import os as _os, json as _json
                _p_json = _os.environ.get("OF3_PAIRING_JSON")
                if _p_json and _os.path.isfile(_p_json):
                    try:
                        _pair = _json.loads(open(_p_json).read())
                    except Exception:
                        _pair = None
                if _pair is None:
                    try:
                        from openfold3.core.data.primitives.sequence.msa import (
                            get_pairing_runtime as _gpr,
                        )
                        _pair = _gpr()
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
                _entry = _lim("of3", query_id)
                if _entry:
                    summary["inputs"] = _entry
        except Exception as _e:
            logger.warning(f"[of3 inputs_meta] {type(_e).__name__}: {_e}")

        # Round all summary floats to 3 decimals for readability.
        def _round3(x):
            if isinstance(x, dict):  return {k: _round3(v) for k, v in x.items()}
            if isinstance(x, list):  return [_round3(v) for v in x]
            if isinstance(x, bool):  return x
            if isinstance(x, float): return round(x, 3)
            return x
        _save_json(sample_dir / "summary.json", _round3(summary))
        _save_json(sample_dir / "meta.json", {
            "writer": "OF3OutputWriter (unified)",
            "model": "of3",
            "job_id": query_id,
            "seed": int(seed) if seed is not None else None,
            "sample_idx": int(sample_idx),
            "native_sample_idx": (
                int(native_sample_idx) if native_sample_idx is not None
                else int(sample_idx)
            ),
        })

    def write_all_outputs(self, batch: dict, outputs: dict, confidence_scores: dict):
        """Writes all outputs for a given batch in the unified layout."""

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]
            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            atom_array_batch = batch["atom_array"][b]
            predicted_coords_batch = (
                outputs["atom_positions_predicted"][b].cpu().float().numpy()
            )
            confidence_scores_batch = _take_batch_dim(confidence_scores, b)

            # Sort samples by sample_ranking_score so sample_0 is always
            # the top-ranked sample (cross-model parity with pt2/bt2; af3
            # is similarly sorted in run_alphafold.py).
            try:
                rscores = np.asarray(_take_batch_dim(
                    confidence_scores.get("sample_ranking_score"), b
                ))
                if rscores.ndim == 0:
                    sample_order = list(range(sample_size))
                else:
                    sample_order = list(np.argsort(-rscores).astype(int))
            except Exception:
                sample_order = list(range(sample_size))

            for new_s, native_s in enumerate(sample_order):
                # Store as 0-indexed for cross-model parity (af3/pt2/bt2 use 0..S-1).
                sample_dir = output_subdir / f"sample_{new_s}"
                confidence_scores_sample = _take_sample_dim(confidence_scores_batch, native_s)
                predicted_coords_sample = predicted_coords_batch[native_s]

                self.write_unified_sample(
                    sample_dir=sample_dir,
                    atom_array=atom_array_batch,
                    predicted_coords=predicted_coords_sample,
                    confidence_scores=confidence_scores_sample,
                    seed=seed,
                    sample_idx=new_s,
                    query_id=query_id,
                    native_sample_idx=native_s,
                )

            def fetch_cur_batch(t):
                # Get tensor for current batch dim
                # Remove expanded sample dim if it exists to get original tensor shapes
                if t.ndim < 2:
                    return t

                cur_feats = t[b : b + 1].squeeze(1)  # noqa: B023
                return cur_feats.detach().clone().cpu()

            file_prefix = output_subdir / f"{query_id}_seed_{seed}"

            # Write out input feature dictionary
            if self.write_features:
                out_file = Path(f"{file_prefix}_batch.pt")
                cur_batch = tensor_tree_map(fetch_cur_batch, batch, strict_type=False)
                torch.save(cur_batch, out_file)
                del cur_batch

            # Write out latent reps / raw model outputs
            if self.write_latent_outputs:
                out_file = Path(f"{file_prefix}_latent_output.pt")
                cur_output = tensor_tree_map(
                    fetch_cur_batch, outputs, strict_type=False
                )
                torch.save(cur_output, out_file)
                del cur_output

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Skip repeated samples
        if batch.get("repeated_sample"):
            return

        self.total_count += 1

        # Skip and track failed samples
        if outputs is None:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        # Write predictions and confidence scores
        # Optionally write out input features and latent outputs
        try:
            self.write_all_outputs(
                batch=batch,
                outputs=outputs,
                confidence_scores=confidence_scores,
            )
            self.success_count += 1
        except Exception as e:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            logger.exception(
                f"Failed to write predictions for query_id(s) "
                f"{', '.join(batch['query_id'])}: {e}"
            )

        del batch, outputs

    def on_predict_end(self, trainer, pl_module):
        """
        Print summary of inference run. Includes a timeout failsafe
        for distributed runs.
        """

        try:
            # Gather summary data from all processes
            final_summary_data = {
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "failed_queries": self.failed_queries,
            }

            gathered_data = [final_summary_data]
            if dist.is_available() and dist.is_initialized():
                gathered_data = [None] * trainer.world_size
                dist.all_gather_object(gathered_data, final_summary_data)

            if trainer.is_global_zero:
                # Aggregate the results from all processes on Rank 0
                total_queries = sum(data["total"] for data in gathered_data)
                success_count = sum(data["success"] for data in gathered_data)
                failed_count = sum(data["failed"] for data in gathered_data)
                final_failed_list = [
                    item for data in gathered_data for item in data["failed_queries"]
                ]

                self._write_summary(
                    total_queries=total_queries,
                    success_count=success_count,
                    failed_count=failed_count,
                    failed_list=final_failed_list,
                    global_rank=trainer.global_rank,
                    is_complete=True,
                )

        except RuntimeError as e:
            # TODO: Due to additional sync PL does outside of this callback,
            #  this won't be reached before the timeout error occurs.
            #  Leaving this here for now in case we refactor the prediction
            #  logic to avoid the extra syncs.
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logger.warning(
                    f"[Rank {trainer.global_rank}] Distributed sync timed out! "
                    f"Writing local results to a fallback log."
                )

                self._write_summary(
                    total_queries=self.total_count,
                    success_count=self.success_count,
                    failed_count=self.failed_count,
                    failed_list=self.failed_queries,
                    global_rank=trainer.global_rank,
                    is_complete=False,
                )

            else:
                # Re-raise unexpected runtime errors
                raise e

    def _write_summary(
        self,
        total_queries: int,
        success_count: int,
        failed_count: int,
        failed_list: list,
        global_rank: int = 0,
        is_complete=True,
    ):
        """Helper to format the final summary."""
        if is_complete:
            status = "COMPLETE"
            out_file = self.output_dir / "summary.txt"
        else:
            status = f"INCOMPLETE (Rank {global_rank})"
            out_file = self.output_dir / f"fallback_summary_rank_{global_rank}.txt"

        summary = [
            "\n" + "=" * 50,
            f"    PREDICTION SUMMARY ({status})    ",
            "=" * 50,
            f"Total Queries Processed: {total_queries}",
            f"  - Successful Queries:  {success_count}",
            f"  - Failed Queries:      {failed_count}",
        ]

        if failed_list:
            failed_str = ", ".join(sorted(list(set(failed_list))))
            summary.append(f"\nFailed Queries: {failed_str}")

        summary.append("=" * 50 + "\n")
        summary = "\n".join(summary)

        out_file.write_text(summary)

        if is_complete:
            print(summary)
        else:
            logger.warning(
                f"Fallback summary for Rank {global_rank} saved to: {out_file}"
            )
