"""
TCR docking angle distance — Pierce-based, with auto CSV mode.

The Pierce angle binary needs an ANARCI + biopython stack in a separate
venv (`analysis/.venv-cdr/`). This module supports two modes:

  Mode A — pre-computed CSV (default when --tcr-angle-csv exists):
      load <csv>; build wrap-safe sin/cos distance matrix; cluster as usual.

  Mode B — auto-invoke Pierce when CSV missing:
      spawn _pierce_worker.py inside `.venv-cdr/bin/python`, write a fresh
      CSV at the requested path, then load it.

The two modes are transparent to the caller: `compute_tcr_angle_matrix`
takes a list of CIF paths AND an optional CSV path. If the CSV exists it
is consumed; otherwise it is produced and consumed.

CSV format (compatible with target_notes/<target>/analysis_outputs/<target>_angles_pierce.csv):

    group_dir,predictor,seed,sample,crossing_deg,incident_deg,model_path[,...]

We derive `pdb_name = <predictor>_seed<seed>_sample<sample>.pdb` per row to
match against MassiveClustering's structure_names.

Distance metric: wrap-safe L2 on (sin θ, cos θ), stacked across crossing
+ incident:

    feat_i = [sin θc, cos θc, sin θi, cos θi]
    dist(i, j) = || feat_i - feat_j ||_2

Two angles differing by Δθ contribute 2|sin(Δθ/2)| to the L2.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Union, Tuple, Optional
import os
import subprocess
import sys
import tempfile
import numpy as np


# Default venv that has biopython 1.81 + ANARCI for Pierce.
# residudes-qa: optional CDR env (ANARCI + biopython 1.81); override with $RESIDUDES_QA_CDR_BIN.
_DEFAULT_VENV_CDR_PYTHON = __import__("os").path.join(
    __import__("os").environ.get("RESIDUDES_QA_CDR_BIN", "/gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin"), "python")


def load_angles_csv(
    csv_path: Union[str, Path],
    group_filter: Optional[str] = None,
) -> dict:
    """Read Pierce CSV → {pdb_name: (crossing_deg, incident_deg)}.

    If group_filter is given, restrict rows to that group_dir
    (e.g. 'af3_msa' to drop 'af3_msa_8yiv' rows for a target where
    multiple MSA variant trees were processed in the same CSV).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if group_filter and "group_dir" in df.columns:
        df = df[df["group_dir"] == group_filter]
    df = df.copy()
    # Build two indexed lookups: canonical (<predictor>_seedN_sampleM.pdb)
    # for AF3-style trees, plus model_path basename so flat-dir (MassiveFold)
    # callers can match against the source filename instead of a synthesised
    # AF3-tree name.
    df["pdb_name"] = (
        df["predictor"].astype(str)
        + "_seed" + df["seed"].astype(int).astype(str)
        + "_sample" + df["sample"].astype(int).astype(str)
        + ".pdb"
    )
    from os.path import basename
    out = {}
    for _, row in df.iterrows():
        c = row.get("crossing_deg")
        i = row.get("incident_deg")
        try:
            c = float(c); i = float(i)
        except (TypeError, ValueError):
            continue
        if np.isnan(c) or np.isnan(i):
            continue
        # Pierce binary outputs crossing in 0–180° but the Rudolph/Singh
        # convention used elsewhere in this project (and TCR-pMHC literature)
        # is 0–90°: crossings >90° are reverse-side projections of the same
        # docking pose and should be folded back via x → 180 − x. Pierce often
        # falls into this branch with "MHC norm vector suspicious" warnings;
        # the geometric angle is still the folded value.
        if c > 90.0:
            c = 180.0 - c
        # Same logic for incident, though incidents > 90° are rarer
        if i > 90.0:
            i = 180.0 - i
        out[row["pdb_name"]] = (c, i)
        mp = row.get("model_path")
        if isinstance(mp, str) and mp:
            # bare basename (with extension)
            bn = basename(mp)
            out[bn] = (c, i)
            # and stem + ".pdb" — the form most matrix-time callers pass in
            # (e.g. flat-dir model_id was built as `<stem>.pdb` in indexers
            # while the actual source was `<stem>.cif`).
            stem = bn.rsplit(".", 1)[0]
            out[f"{stem}.pdb"] = (c, i)
    return out


def run_pierce_worker(
    cif_paths: Sequence[Union[str, Path]],
    output_csv: Union[str, Path],
    chain_layout: str = "ours_5chain",
    target_name: Optional[str] = None,
    venv_python: str = _DEFAULT_VENV_CDR_PYTHON,
    workers: int = 16,
) -> Path:
    """Spawn `_pierce_worker.py` inside the .venv-cdr Python and wait."""
    here = Path(__file__).parent
    worker = here / "_pierce_worker.py"
    if not Path(venv_python).is_file():
        raise FileNotFoundError(
            f"Pierce venv python not found: {venv_python}. "
            "Run analysis/setup.sh --with-cdr to build it."
        )
    if not worker.is_file():
        raise FileNotFoundError(f"missing pierce worker: {worker}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in cif_paths:
            f.write(f"{p}\n")
        list_path = f.name

    cmd = [
        venv_python, str(worker),
        "--cifs-list", list_path,
        "--output-csv", str(output_csv),
        "--chain-layout", chain_layout,
        "--workers", str(workers),
    ]
    if target_name:
        cmd += ["--target-name", target_name]

    print(f"[tcr-angle] invoking Pierce worker on {len(cif_paths)} CIFs "
          f"via {venv_python}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"Pierce worker failed (exit {r.returncode}). "
            f"cmd: {' '.join(cmd)}"
        )
    return Path(output_csv)


def compute_tcr_angle_features(
    structure_names: Sequence[str],
    angle_csv: Union[str, Path],
    group_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Match names against CSV → (features_deg, features_sincos, kept_indices)."""
    table = load_angles_csv(angle_csv, group_filter=group_filter)
    deg, kept = [], []
    for k, name in enumerate(structure_names):
        if name in table:
            deg.append(table[name])
            kept.append(k)
    deg = np.asarray(deg, dtype=np.float32)
    if len(deg):
        rad = np.deg2rad(deg)
        sc = np.stack([
            np.sin(rad[:, 0]), np.cos(rad[:, 0]),
            np.sin(rad[:, 1]), np.cos(rad[:, 1]),
        ], axis=1).astype(np.float32)
    else:
        sc = np.zeros((0, 4), dtype=np.float32)
    return deg, sc, kept


def compute_tcr_angle_matrix(
    structure_names: Sequence[str],
    cif_paths: Optional[Sequence[Union[str, Path]]] = None,
    angle_csv: Optional[Union[str, Path]] = None,
    group_filter: Optional[str] = None,
    chain_layout: str = "ours_5chain",
    target_name: Optional[str] = None,
    recompute: bool = False,
    venv_python: str = _DEFAULT_VENV_CDR_PYTHON,
    workers: int = 16,
    output_path: Union[str, Path, None] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build N×N wrap-safe angular distance matrix from TCR docking angles.

    Args:
      structure_names : ordered names matching cluster_results convention
                        (<predictor>_seed<N>_sample<M>.pdb).
      cif_paths       : aligned CIF/PDB paths — required only when CSV
                        is missing or recompute=True.
      angle_csv       : path to Pierce CSV. If exists and not recomputing,
                        it is loaded directly. If missing, Pierce is invoked
                        via _pierce_worker.py to produce one here.
      group_filter    : optional CSV group_dir filter (e.g. 'af3_msa').
      chain_layout    : 'ours_5chain' or 'ours_4chain' (passed to Pierce).
      target_name     : optional HLA-offset key (e.g. '9RU5').
      recompute       : force re-running Pierce even if CSV exists.
      venv_python     : path to .venv-cdr python (must have ANARCI).
      workers         : parallel workers for Pierce.
      output_path     : where to save the resulting distance matrix.

    Returns:
      distance_matrix : (N_kept, N_kept) float32
      names_kept      : filtered names (rows with no CSV entry are dropped)
      features_deg    : (N_kept, 2) [crossing_deg, incident_deg]
    """
    csv = Path(angle_csv) if angle_csv else None
    needs_compute = recompute or (csv is None) or (not csv.is_file())
    if needs_compute:
        if cif_paths is None:
            raise ValueError(
                "tcr-angle CSV missing and cif_paths not provided — cannot "
                "auto-invoke Pierce."
            )
        if csv is None:
            raise ValueError(
                "tcr-angle CSV missing and angle_csv path not provided — "
                "specify output location via --tcr-angle-csv"
            )
        if show_progress:
            print(f"[tcr-angle] CSV missing/recompute → invoking Pierce "
                  f"on {len(cif_paths)} models", flush=True)
        run_pierce_worker(
            cif_paths, csv,
            chain_layout=chain_layout, target_name=target_name,
            venv_python=venv_python, workers=workers,
        )

    deg, sc, kept = compute_tcr_angle_features(
        structure_names, csv, group_filter=group_filter,
    )
    N_orig = len(structure_names)
    N = len(kept)
    if show_progress:
        print(f"[tcr-angle] kept {N}/{N_orig} models with angle data")
        if N < N_orig:
            missing = [structure_names[i] for i in range(N_orig)
                       if i not in set(kept)]
            print(f"  WARN: {N_orig - N} models missing CSV entries "
                  f"(first 3: {missing[:3]})")
    if N == 0:
        raise RuntimeError(
            "No structure_names matched the angle CSV — expected "
            "'<predictor>_seed<N>_sample<M>.pdb' convention"
        )

    # Pairwise L2 on the 4D sin/cos representation.
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.tensor(sc, device="cuda", dtype=torch.float32)
            D = torch.cdist(t, t).cpu().numpy().astype(np.float32)
            del t
            torch.cuda.empty_cache()
        else:
            raise RuntimeError("no cuda")
    except Exception:
        from scipy.spatial.distance import cdist
        D = cdist(sc, sc).astype(np.float32)

    names_kept = [structure_names[i] for i in kept]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, D)
        np.save(output_path.with_name(output_path.stem + "_features_deg.npy"),
                deg)
        np.save(output_path.with_name(output_path.stem + "_features_sincos.npy"),
                sc)
        np.save(output_path.with_name(output_path.stem + "_names.npy"),
                np.array(names_kept))

    return D, names_kept, deg


__all__ = [
    "load_angles_csv",
    "run_pierce_worker",
    "compute_tcr_angle_features",
    "compute_tcr_angle_matrix",
]
