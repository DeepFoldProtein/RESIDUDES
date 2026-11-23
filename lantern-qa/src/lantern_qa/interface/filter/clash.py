"""Proper heavy-atom clash detection (1.7 Å, disulfide-aware).

Generalized from H1335 ``46b_proper_clash.py``.

A clash is a heavy-atom pair at less than ``threshold`` Å where:

* the two atoms are NOT in the same residue,
* the two atoms are NOT in adjacent residues of the same chain (peptide bond
  / 1-2 backbone neighbours produce trivial sub-2 Å N-CA distances), and
* the two atoms are NOT a CYS-SG / CYS-SG disulfide pair (S-S bond up to
  ``disulfide_max`` Å).

Returned per-CIF dict::

    {
        'n_clash_17': int,         # total clash count
        'n_inter_clash_17': int,   # inter-chain subset
        'clash_examples': str,     # 'ch.res.atom-ch.res.atom=d Å; ...'  (top 3)
    }
"""
from __future__ import annotations

import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def detect_clashes(
    cif_path: str | Path,
    threshold: float = 1.7,
    disulfide_max: float = 2.5,
) -> Optional[dict[str, Any]]:
    """Run clash detection on a single CIF.  Returns ``None`` on read failure."""
    import gemmi
    from scipy.spatial import cKDTree

    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return None

    atoms: list[tuple[str, int, str, str, np.ndarray]] = []
    cys_sg: list[int] = []
    for model in st:
        for ch in model:
            for r in ch:
                for a in r:
                    if a.element.name == "H":
                        continue
                    coord = np.array([a.pos.x, a.pos.y, a.pos.z])
                    atoms.append((ch.name, r.seqid.num, r.name, a.name, coord))
                    if r.name == "CYS" and a.name == "SG":
                        cys_sg.append(len(atoms) - 1)
        break
    if not atoms:
        return None

    xyz = np.array([a[4] for a in atoms])
    tree = cKDTree(xyz)

    # Discover all disulfides (CYS SG-SG pairs within disulfide_max).
    ssbonds: set[tuple[int, int]] = set()
    for ii, i in enumerate(cys_sg):
        for j in cys_sg[ii + 1 :]:
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if d <= disulfide_max:
                a, b = (i, j) if i < j else (j, i)
                ssbonds.add((a, b))

    pairs = tree.query_pairs(threshold, output_type="ndarray")
    n_clash = 0
    n_inter = 0
    examples: list[str] = []
    for i, j in pairs:
        if i > j:
            i, j = j, i
        ci, ri, _, ai, _ = atoms[i]
        cj, rj, _, aj, _ = atoms[j]
        # Same residue or adjacent same-chain — peptide / 1-2 neighbours.
        if ci == cj and abs(ri - rj) <= 1:
            continue
        if (int(i), int(j)) in ssbonds:
            continue
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        n_clash += 1
        if ci != cj:
            n_inter += 1
        if len(examples) < 5:
            examples.append(f"{ci}.{ri}.{ai}-{cj}.{rj}.{aj}={d:.2f}Å")
    return {
        "n_clash_17": int(n_clash),
        "n_inter_clash_17": int(n_inter),
        "clash_examples": "; ".join(examples[:3]),
    }


def _process_one(
    arg: tuple[str, str, float, float],
) -> dict[str, Any]:
    model_id, cif_path, threshold, disulfide_max = arg
    out: dict[str, Any] = {"model_id": model_id}
    r = detect_clashes(cif_path, threshold=threshold, disulfide_max=disulfide_max)
    if r is None:
        out["clash_err"] = "read_failed_or_empty"
    else:
        out.update(r)
    return out


def run_clash_audit_parallel(
    index_df: pd.DataFrame,
    threshold: float = 1.7,
    disulfide_max: float = 2.5,
    workers: int = 8,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Pool-parallel clash audit.  Returns DataFrame keyed on ``model_id``."""
    records = [
        (r["model_id"], r["src_cif"], threshold, disulfide_max)
        for _, r in index_df.iterrows()
    ]
    n = len(records)
    print(
        f"  clash audit: {n} cifs  workers={workers}  threshold={threshold} Å",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict] = []
    n_err = 0
    with Pool(workers) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=10), 1
        ):
            rows.append(res)
            if "clash_err" in res:
                n_err += 1
            if i % progress_every == 0 or i == n:
                rate = i / (time.time() - t0 + 1e-6)
                eta = (n - i) / rate / 60 if rate else 0.0
                print(
                    f"    [{i}/{n}] {rate:.1f} cif/s  ETA {eta:.1f} min  err={n_err}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    print(
        f"  clash audit: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_clash_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/clash_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "clash_features.parquet"
    df.to_parquet(out, index=False)
    return out
