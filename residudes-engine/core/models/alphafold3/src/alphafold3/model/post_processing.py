# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Post-processing utilities for AlphaFold inference results.

Output layout (unified across af3/pt2/of3/bt2):
    <output_dir>/seed_<seed>/sample_<idx>/
        model.cif
        summary.json     (cross-model schema)
        meta.json
        arrays/
            plddt_atom.npy
            plddt_token.npy        (mean over atoms-in-residue, derived)
            pae.npy
            pde.npy
            contact_probs.npy

Native per-pair `confidences.json` (~O(N²)) is no longer written; the same
arrays land directly in arrays/*.npy. Distogram is saved once per seed at
<output_dir>/seed_<seed>/distogram_probs.npy (uint8, log-quantized).
"""

import dataclasses
import datetime
import json
import os
from pathlib import Path

from alphafold3 import version
from alphafold3.model import confidence_types
from alphafold3.model import mmcif_metadata
from alphafold3.model import model
import numpy as np
import zstandard


def _atom_to_token(atom_chain_ids, token_chain_ids, token_res_ids,
                   atom_res_ids) -> np.ndarray | None:
  """Map each atom to its token (residue) index using parallel chain/res ids."""
  try:
    token_keys = list(zip(token_chain_ids, token_res_ids, strict=True))
    key_to_idx = {k: i for i, k in enumerate(token_keys)}
    out = np.empty(len(atom_chain_ids), dtype=np.int32)
    for i, (c, r) in enumerate(zip(atom_chain_ids, atom_res_ids, strict=True)):
      out[i] = key_to_idx.get((c, int(r)), -1)
    if (out < 0).any():
      return None
    return out
  except Exception:
    return None


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
  if arr.dtype.kind == 'f' and arr.dtype != np.float32:
    arr = arr.astype(np.float32, copy=False)
  p.parent.mkdir(parents=True, exist_ok=True)
  np.save(p, arr)


def _compute_gpde(pde, contact_probs):
  """AF3 SI 5.7 Eq.16 — Σ pde·cp / Σ cp over all token pairs (incl. diag).
  Identical formula across pt2/of3/bt2/af3 writers."""
  if pde is None or contact_probs is None:
    return None
  pde = np.asarray(pde, dtype=np.float32)
  cp = np.asarray(contact_probs, dtype=np.float32)
  if pde.shape != cp.shape or pde.ndim < 2:
    return None
  s = float(cp.sum())
  if s <= 0:
    return None
  return float((pde * cp).sum() / s)


# Canonical _atom_site column order (AF3-native order kept verbatim) and the
# noise categories we drop on normalize. Mirrors core/scripts/cif_normalize.py
# so all four runners produce byte-equivalent model.cif.
_AF3_ATOM_SITE_COLS = (
    'group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id',
    'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
    'pdbx_PDB_ins_code',
    'Cartn_x', 'Cartn_y', 'Cartn_z',
    'occupancy', 'B_iso_or_equiv',
    'auth_seq_id', 'auth_asym_id', 'pdbx_PDB_model_num',
)
_DROP_CIF_CATEGORIES = (
    '_audit_author', '_audit_conform',
    '_citation', '_citation_author',
    '_pdbx_data_usage',
    '_struct', '_struct_keywords',
)


def _normalize_cif_via_gemmi(cif_path: Path) -> bool:
  """Strip provenance/legal noise + add _ma_qa_metric* (B_iso = pLDDT)."""
  try:
    import gemmi
  except Exception:
    return False
  try:
    doc = gemmi.cif.read_file(str(cif_path))
  except Exception:
    return False
  if not len(doc):
    return False
  b = doc[0]
  for cat in _DROP_CIF_CATEGORIES:
    if b.get_mmcif_category(cat + '.'):
      b.set_mmcif_category(cat + '.', {}, raw=True)
  site = b.get_mmcif_category('_atom_site.')
  if not site:
    return False
  n = len(next(iter(site.values())))
  canon = {}
  for col in _AF3_ATOM_SITE_COLS:
    if col in site:
      canon[col] = site[col]
    elif col == 'auth_seq_id' and 'label_seq_id' in site:
      canon[col] = site['label_seq_id']
    elif col == 'auth_asym_id' and 'label_asym_id' in site:
      canon[col] = site['label_asym_id']
    elif col == 'id':
      canon[col] = [str(i + 1) for i in range(n)]
    elif col == 'occupancy':
      canon[col] = ['1.00'] * n
    elif col in ('label_alt_id', 'pdbx_PDB_ins_code'):
      canon[col] = ['?'] * n
    elif col in ('pdbx_PDB_model_num', 'label_entity_id'):
      canon[col] = ['1'] * n
    elif col == 'group_PDB':
      canon[col] = ['ATOM'] * n
    else:
      canon[col] = ['?'] * n
  b.set_mmcif_category('_atom_site.', canon, raw=False)
  if not b.get_mmcif_category('_ma_qa_metric.'):
    groups: dict = {}
    order: list = []
    b_iso = canon['B_iso_or_equiv']
    for i in range(n):
      try:
        v = float(b_iso[i])
      except (TypeError, ValueError):
        continue
      key = (canon['label_asym_id'][i], canon['label_seq_id'][i],
             canon['label_comp_id'][i])
      if key not in groups:
        groups[key] = []
        order.append(key)
      groups[key].append(v)
    if order:
      b.set_mmcif_category('_ma_qa_metric.', {
          'id': ['1', '2'],
          'mode': ['global', 'local'],
          'name': ['pLDDT', 'pLDDT'],
          'software_group_id': ['1', '1'],
          'type': ['pLDDT', 'pLDDT'],
      }, raw=False)
      global_pl = sum(sum(v) / len(v) for v in groups.values()) / len(order)
      b.set_mmcif_category('_ma_qa_metric_global.', {
          'metric_id': ['1'],
          'metric_value': [f'{global_pl:.2f}'],
          'model_id': ['1'],
          'ordinal_id': ['1'],
      }, raw=False)
      b.set_mmcif_category('_ma_qa_metric_local.', {
          'ordinal_id':    [str(o + 1) for o in range(len(order))],
          'model_id':      ['1'] * len(order),
          'label_asym_id': [k[0] for k in order],
          'label_seq_id':  [k[1] for k in order],
          'label_comp_id': [k[2] for k in order],
          'metric_id':     ['2'] * len(order),
          'metric_value':  [f'{sum(groups[k]) / len(groups[k]):.2f}' for k in order],
      }, raw=False)
      if not b.get_mmcif_category('_ma_data.'):
        b.set_mmcif_category('_ma_data.', {
            'content_type': ['model coordinates'],
            'id': ['1'],
            'name': ['Model'],
        }, raw=False)
      if not b.get_mmcif_category('_ma_model_list.'):
        b.set_mmcif_category('_ma_model_list.', {
            'data_id': ['1'],
            'model_group_id': ['1'],
            'model_group_name': ['Predicted models'],
            'model_id': ['1'],
            'model_name': ['Model'],
            'model_type': ['Ab initio model'],
            'ordinal_id': ['1'],
        }, raw=False)
  doc.write_file(str(cif_path))
  return True


def _save_json(p: Path, d: dict) -> None:
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(d, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else None))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedInferenceResult:
  """Stores attributes of a processed inference result.

  Attributes:
    cif: CIF file containing an inference result.
    mean_confidence_1d: Mean 1D confidence calculated from confidence_1d.
    ranking_score: Ranking score extracted from CIF metadata.
    structure_confidence_summary_json: Content of JSON file with structure
      confidences summary calculated from CIF file.
    structure_full_data_json: Content of JSON file with structure full
      confidences calculated from CIF file.
    model_id: Identifier of the model that produced the inference result.
  """

  cif: bytes
  mean_confidence_1d: float
  ranking_score: float
  structure_confidence_summary_json: bytes
  structure_full_data_json: bytes
  model_id: bytes


def post_process_inference_result(
    inference_result: model.InferenceResult,
) -> ProcessedInferenceResult:
  """Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence."""

  # Add mmCIF metadata fields.
  timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
  cif_with_metadata = mmcif_metadata.add_metadata_to_mmcif(
      old_cif=inference_result.predicted_structure.to_mmcif_dict(),
      version=f'{version.__version__} @ {timestamp}',
      model_id=inference_result.model_id,
  )
  cif = mmcif_metadata.add_legal_comment(cif_with_metadata.to_string())
  cif = cif.encode('utf-8')
  confidence_1d = confidence_types.AtomConfidence.from_inference_result(
      inference_result
  )
  mean_confidence_1d = np.mean(confidence_1d.confidence)
  structure_confidence_summary_json = (
      confidence_types.StructureConfidenceSummary.from_inference_result(
          inference_result
      )
      .to_json()
      .encode('utf-8')
  )
  structure_full_data_json = (
      confidence_types.StructureConfidenceFull.from_inference_result(
          inference_result
      )
      .to_json()
      .encode('utf-8')
  )
  return ProcessedInferenceResult(
      cif=cif,
      mean_confidence_1d=mean_confidence_1d,
      ranking_score=float(inference_result.metadata['ranking_score']),
      structure_confidence_summary_json=structure_confidence_summary_json,
      structure_full_data_json=structure_full_data_json,
      model_id=inference_result.model_id,
  )


def write_output(
    inference_result: model.InferenceResult,
    output_dir: os.PathLike[str] | str,
    terms_of_use: str | None = None,
    name: str | None = None,
    compress: bool = False,
    sample_idx: int | None = None,
    seed: int | None = None,
    job_name: str | None = None,
) -> None:
  """Writes processed inference result in the unified cross-model layout.

  When called from `run_alphafold.write_outputs` per sample, `sample_idx` and
  `seed` should be set; the writer then emits unified files directly under
  `<output_dir>` (which is expected to be `<run_dir>/seed_<seed>/sample_<idx>/`).

  Legacy compat: when `sample_idx` is None this writes the legacy max-rank
  triplet (model.cif, summary_confidences.json, confidences.json) flat at
  `<output_dir>/{prefix}{file}` — used by run_alphafold for the top-level
  "best across all seeds" copy.
  """
  processed_result = post_process_inference_result(inference_result)

  prefix = f'{name}_' if name is not None else ''
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # ── Legacy / top-level path (no sample_idx given) ───────────────────────
  if sample_idx is None:
    if compress:
      opener = zstandard.open
      path_transform = lambda path: f'{path}.zst'
    else:
      opener = open
      path_transform = lambda path: path

    mmcif_path = output_dir / f'{prefix}model.cif'
    with opener(path_transform(str(mmcif_path)), 'wb') as f:
      f.write(processed_result.cif)
    summary_path = output_dir / f'{prefix}summary_confidences.json'
    with open(str(summary_path), 'wb') as f:
      f.write(processed_result.structure_confidence_summary_json)
    if terms_of_use is not None:
      (output_dir / 'TERMS_OF_USE.md').write_text(terms_of_use)
    return

  # ── Unified per-sample path ─────────────────────────────────────────────
  arrays_dir = output_dir / 'arrays'
  arrays_dir.mkdir(parents=True, exist_ok=True)

  # 1) Structure cif (then normalize via gemmi for cross-model parity).
  if compress:
    (output_dir / 'model.cif.zst').write_bytes(b'')  # placeholder if needed
    with zstandard.open(str(output_dir / 'model.cif.zst'), 'wb') as f:
      f.write(processed_result.cif)
  else:
    cif_path = output_dir / 'model.cif'
    cif_path.write_bytes(processed_result.cif)
    _normalize_cif_via_gemmi(cif_path)

  # 2) Arrays from full data
  full = confidence_types.StructureConfidenceFull.from_inference_result(inference_result)
  atom_plddts = np.asarray(full.atom_plddts, dtype=np.float32)
  _save_npy(arrays_dir / 'plddt_atom.npy', atom_plddts)
  _save_npy(arrays_dir / 'contact_probs.npy', full.contact_probs)
  # pae / pde / plddt as raw pre-softmax logits via logit_codec
  # (matches pt2/of3/bt2 unified format).
  if (full.pae_logits is not None or full.pde_logits is not None
      or full.plddt_logits is not None):
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _utils = _os.path.normpath(_os.path.join(_here, '..', 'utils'))
    if _utils not in _sys.path:
      _sys.path.insert(0, _utils)
    import logit_codec  # noqa: I001
    if full.pae_logits is not None:
      logit_codec.save_logits(
          arrays_dir / 'pae_logits.npz', pae=np.asarray(full.pae_logits))
    if full.pde_logits is not None:
      logit_codec.save_logits(
          arrays_dir / 'pde_logits.npz', pde=np.asarray(full.pde_logits))
    if full.plddt_logits is not None:
      # plddt_logits: [num_tokens, num_atom_per_token, num_plddt_bins].
      # Reduce the dense atom axis (mean) to match the ab/ag-canonical
      # (N_token, num_plddt_bins) shape used by the unified logit npz.
      pl = np.asarray(full.plddt_logits, dtype=np.float32)
      if pl.ndim == 3:
        pl = pl.mean(axis=1)
      logit_codec.save_logits(
          arrays_dir / 'plddt_logits.npz', plddt=pl)

  # plddt_token (mean atom pLDDT per token).
  # AF3's Structure exposes per-atom `chain_id` and `res_id` directly; use
  # those to build the atom→token mapping (parallels of3/pt2 which use
  # AtomArray.chain_id / AtomArray.res_id).
  struct = inference_result.predicted_structure
  try:
    atom_res_ids = np.asarray(struct.res_id).tolist()
    a2t = _atom_to_token(
        atom_chain_ids=full.atom_chain_ids,
        token_chain_ids=full.token_chain_ids,
        token_res_ids=full.token_res_ids,
        atom_res_ids=atom_res_ids,
    )
    if a2t is not None:
      _save_npy(arrays_dir / 'plddt_token.npy',
                _aggregate_atom_to_token(atom_plddts, a2t))
  except Exception:
    pass

  # 3) Summary (cross-model schema). Read full-precision values directly
  # from inference_result.metadata; af3's StructureConfidenceSummary.to_json
  # rounds to 2 decimals for human readers, but pt2/of3/bt2 emit full
  # float32 precision — match them.
  meta = inference_result.metadata

  def _f(x):
    try:
      v = float(x)
    except Exception:
      return None
    return v if not np.isnan(v) else None

  # Canonicalize chain_*/chain_pair_iptm to letter-keyed dicts (the
  # cross-model unified shape). af3's native arrays are positional:
  #   iptm_ichain     → chain_ptm   [num_chains]
  #   iptm_xchain     → chain_iptm  [num_chains] (NaN-filled for monomer)
  #   chain_pair_iptm                [num_chains, num_chains]  (diag = self-pTM)
  # Chain letter order = unique tokens in full.token_chain_ids (order-preserving).
  seen = []
  for c in full.token_chain_ids:
    if c not in seen:
      seen.append(c)
  chain_keys = seen
  ncpt = np.asarray(meta.get('iptm_ichain')) if 'iptm_ichain' in meta else None
  nciptm = np.asarray(meta.get('iptm_xchain')) if 'iptm_xchain' in meta else None
  ncpi = np.asarray(meta.get('chain_pair_iptm')) if 'chain_pair_iptm' in meta else None

  cptm_dict = None
  if ncpt is not None and ncpt.shape == (len(chain_keys),):
    cptm_dict = {chain_keys[i]: _f(ncpt[i]) for i in range(len(chain_keys))}

  ciptm_dict = None
  if nciptm is not None and nciptm.shape == (len(chain_keys),):
    if not np.any(np.isnan(nciptm)):
      ciptm_dict = {chain_keys[i]: _f(nciptm[i]) for i in range(len(chain_keys))}

  cpi_canon = None
  if ncpi is not None and ncpi.shape == (len(chain_keys), len(chain_keys)):
    cpi_canon = {
        chain_keys[i]: {chain_keys[j]: _f(ncpi[i][j])
                        for j in range(len(chain_keys))}
        for i in range(len(chain_keys))
    }

  # gpde via cross-model unified formula (AF3 SI 5.7 Eq.16):
  #   gpde = Σ_ij contact_probs[i,j] * pde[i,j] / Σ_ij contact_probs[i,j]
  # over all token pairs (incl. diagonal). Identical formula in pt2/of3/bt2
  # writers — fully reproducible from the saved arrays.
  gpde_scalar = _compute_gpde(full.pde, full.contact_probs)

  unified_summary = {
      'model_name': 'af3',
      'seed': int(seed) if seed is not None else None,
      'sample_idx': int(sample_idx),
      'n_tokens': len(full.token_chain_ids),
      'n_atoms': int(atom_plddts.shape[-1]),
      'plddt': _f(np.mean(atom_plddts)),
      'gpde': gpde_scalar,  # AF3 SI Eq.16, unified across pt2/of3/bt2
      'ptm': _f(meta.get('ptm')),
      'iptm': _f(meta.get('iptm')),
      'has_clash': bool(meta.get('has_clash')) if meta.get('has_clash') is not None else None,
      'ranking_score': _f(meta.get('ranking_score')),
      'fraction_disordered': _f(meta.get('fraction_disordered')),
      'chain_ptm': cptm_dict,
      'chain_iptm': ciptm_dict,
      'chain_pair_iptm': cpi_canon,
  }
  # MSA + template provenance: prefer featurizer-time view dropped at
  # <seed_dir>/inputs_runtime.json by run_alphafold.write_outputs; fall
  # back to lookup_inputs_meta (input.json provenance) if not present.
  try:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(
        _os.environ.get('RESIDUDES_ENGINE_CORE')
        or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[_os.pardir] * 5)), 'scripts'))
    from inputs_meta import read_runtime_inputs as _rri, lookup_inputs_meta as _lim  # type: ignore
    rt = _rri(output_dir)
    if rt is not None:
      unified_summary['inputs'] = rt
    else:
      _entry = _lim('af3', job_name or '')
      if _entry:
        unified_summary['inputs'] = _entry
  except Exception:
    pass
  # Round all float values in the summary tree to 3 decimals for readability
  # (matches pt2/of3/bt2 behaviour). int/bool/str/None passthrough unchanged.
  def _round3(x):
    if isinstance(x, dict):  return {k: _round3(v) for k, v in x.items()}
    if isinstance(x, list):  return [_round3(v) for v in x]
    if isinstance(x, bool):  return x  # bool is int subclass — keep as-is
    if isinstance(x, float): return round(x, 3)
    return x
  _save_json(output_dir / 'summary.json', _round3(unified_summary))
  _save_json(output_dir / 'meta.json', {
      'writer': 'af3 post_processing.write_output (unified)',
      'model': 'af3',
      'job_id': job_name,
      'seed': int(seed) if seed is not None else None,
      'sample_idx': int(sample_idx),
  })

  if terms_of_use is not None:
    (output_dir / 'TERMS_OF_USE.md').write_text(terms_of_use)


def write_distogram(
    distogram: np.ndarray,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Save the per-seed distogram as raw pre-softmax logits via logit_codec
  (int8_softmax, ~1e-4 KL recovery; ~256 KB at N=200 bins=64 vs ~5 MB
  uint8 prob)."""
  if distogram is None:
    return
  arr = np.asarray(distogram, dtype=np.float32)
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  import sys, os
  # Vendored at alphafold3/src/alphafold3/utils/logit_codec.py — accessible
  # both at host (.../alphafold3/src/alphafold3/utils/) and inside the SIF
  # (.../app/alphafold/src/alphafold3/utils/) since the SIF binds
  # <checkout>/alphafold3 -> /app/alphafold. Resolve via relative path so the
  # exact same code path works in either location.
  _here = os.path.dirname(os.path.abspath(__file__))
  _utils = os.path.normpath(os.path.join(_here, '..', 'utils'))
  if _utils not in sys.path:
    sys.path.insert(0, _utils)
  import logit_codec  # noqa: I001
  logit_codec.save_logits(out_dir / 'distogram_logits.npz', distogram=arr)


def write_embeddings(
    embeddings: dict[str, np.ndarray],
    output_dir: os.PathLike[str] | str,
    name: str | None = None,
) -> None:
  """Writes embeddings to a directory."""
  prefix = f'{name}_' if name is not None else ''

  with open(os.path.join(output_dir, f'{prefix}embeddings.npz'), 'wb') as f:
    np.savez_compressed(f, **embeddings)
