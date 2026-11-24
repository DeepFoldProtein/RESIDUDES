"""
Chain canonicalisation for homo-oligomer ensembles.

Global-Kabsch TM / dRMSD use a fixed file-order residue correspondence, so for a
**homo-oligomer** (identical chains) they are sensitive to how each model labels
its interchangeable chains. A *cyclic* relabel of a symmetric assembly is a rigid
rotation (TM stays ~1.0, Kabsch absorbs it), but a *transposition* is improper
(TM collapses, e.g. ~0.15 for a C3 trimer). Different predictors label chains
arbitrarily, so a large fraction of a pool can be mutually mis-scored — which
fragments clusters.

This module fixes that as a preprocessing step: each structure's chains are
relabelled to the permutation (best of all chain orderings) that maximises the
global TM to a common reference, putting every model in one consistent frame.
After this, TM/dRMSD become valid quaternary-structure distances and the normal
clustering pipeline can run.

Hetero-oligomers don't need this (distinct chains => unambiguous correspondence).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# structure IO + geometry
# --------------------------------------------------------------------------- #
def _read(path) -> "object":
    """Read .pdb or .cif (incl. PDB content under a .cif name). None on failure."""
    import gemmi
    try:
        return gemmi.read_structure(str(path))
    except Exception:
        try:
            return gemmi.read_pdb(str(path))
        except Exception:
            return None


def chain_ca_by_resnum(model) -> Dict[str, Dict[int, tuple]]:
    """{chain_id: {resnum: (x,y,z)}} over CA atoms of the first model."""
    out: Dict[str, Dict[int, tuple]] = {}
    for ch in model:
        d = {}
        for r in ch:
            ca = r.find_atom("CA", "*")
            if ca is None:
                ca = next((a for a in r if a.name == "CA"), None)
            if ca is not None:
                d[r.seqid.num] = (ca.pos.x, ca.pos.y, ca.pos.z)
        if d:
            out[ch.name] = d
    return out


def kabsch_tm(P: np.ndarray, Q: np.ndarray, l_norm: int) -> float:
    """TM-score of P superposed onto Q (single-shot Kabsch). For ranking
    permutations one shot is enough; the correct permutation dominates clearly."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    R = Wt.T @ np.diag([1.0, 1.0, d]) @ V.T
    dist = np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1))
    d0 = 1.24 * ((l_norm - 15) ** (1.0 / 3.0)) - 1.8 if l_norm > 21 else 0.5
    return float((1.0 / (1.0 + (dist / d0) ** 2)).sum() / l_norm)


def permutation_parity(perm: Sequence, base: Sequence) -> int:
    """0 = even (identity/cyclic), 1 = odd (involves a transposition)."""
    idx = [list(base).index(p) for p in perm]
    inv = sum(1 for i in range(len(idx)) for j in range(i + 1, len(idx)) if idx[i] > idx[j])
    return inv % 2


def best_chain_permutation(
    model_chains: Dict[str, Dict[int, tuple]],
    ref_chains: Dict[str, Dict[int, tuple]],
    max_perms: int = 5040,
) -> Tuple[Optional[tuple], Optional[float]]:
    """Best-matching chain ordering of ``model_chains`` vs ``ref_chains``.

    Returns ``(best_perm, best_tm)`` where ``best_perm[i]`` is the model chain
    name placed in reference slot ``sorted(ref)[i]``. None if chain counts differ
    or the permutation count exceeds ``max_perms``.
    """
    ref_slots = sorted(ref_chains.keys())
    m_names = sorted(model_chains.keys())
    if len(m_names) != len(ref_slots):
        return None, None
    if math.factorial(len(m_names)) > max_perms:
        return None, None
    l_norm = sum(len(ref_chains[s]) for s in ref_slots)
    best_perm, best_tm = None, -1.0
    for perm in itertools.permutations(m_names):
        P, Q = [], []
        for slot, mc in zip(ref_slots, perm):
            for rn in (ref_chains[slot].keys() & model_chains[mc].keys()):
                Q.append(ref_chains[slot][rn]); P.append(model_chains[mc][rn])
        if len(P) < 10:
            continue
        tm = kabsch_tm(np.array(P, float), np.array(Q, float), l_norm)
        if tm > best_tm:
            best_tm, best_perm = tm, perm
    return best_perm, best_tm


def write_canonical(src_path, perm: Sequence, out_path) -> None:
    """Rewrite ``src_path`` with chains reordered to ``perm`` and relabelled A,B,C..."""
    import gemmi
    st = _read(src_path)
    model = st[0]
    chains = {ch.name: ch for ch in model}
    new_st = gemmi.Structure()
    new_st.name = Path(out_path).stem
    nm = gemmi.Model("1")
    for lab, src_name in zip((chr(ord("A") + i) for i in range(len(perm))), perm):
        nc = gemmi.Chain(lab)
        for res in chains[src_name]:
            nc.add_residue(res)
        nm.add_chain(nc)
    new_st.add_model(nm)
    new_st.write_pdb(str(out_path))


# --------------------------------------------------------------------------- #
# batch driver (parallel)
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init_worker(ref_path, max_perms, require_len, out_dir):
    st = _read(ref_path)
    ref = chain_ca_by_resnum(st[0])
    _W.update(ref=ref, slots=sorted(ref.keys()),
              reflen=sum(len(v) for v in ref.values()),
              max_perms=max_perms, require_len=require_len, out_dir=Path(out_dir))


def _work(item):
    src, outname = item
    st = _read(src)
    if st is None or len(st) == 0:
        return dict(id=Path(outname).stem, status="read_fail", n_chains=0,
                    parity=-1, tm_to_ref=float("nan"), written=False)
    m = chain_ca_by_resnum(st[0])
    nres = sum(len(v) for v in m.values())
    if _W["require_len"] and nres != _W["reflen"]:
        return dict(id=Path(outname).stem, status="difflen", n_chains=len(m),
                    parity=-1, tm_to_ref=float("nan"), written=False)
    perm, tm = best_chain_permutation(m, _W["ref"], _W["max_perms"])
    if perm is None:
        return dict(id=Path(outname).stem, status="noperm", n_chains=len(m),
                    parity=-1, tm_to_ref=float("nan"), written=False)
    write_canonical(src, perm, _W["out_dir"] / outname)
    return dict(id=Path(outname).stem, status="ok", n_chains=len(m),
                parity=permutation_parity(perm, sorted(m.keys())),
                tm_to_ref=round(tm, 4), written=True)


def canonicalize_directory(
    input_dir,
    output_dir,
    *,
    ref: Optional[str] = None,
    ext: Sequence[str] = ("pdb", "cif"),
    max_perms: int = 5040,
    require_ref_length: bool = False,
    workers: int = 8,
):
    """Canonicalise every structure in ``input_dir`` to a common chain frame.

    Writes canonical .pdb files (chains A,B,C...) to ``output_dir`` + returns a
    manifest DataFrame (id, status, n_chains, parity, tm_to_ref, written). The
    reference defaults to the first complete structure (most common chain count).
    """
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lstrip(".") in ext)
    if not files:
        raise SystemExit(f"no .{'/.'.join(ext)} files in {input_dir}")

    if ref is None:
        # pick reference = a structure with the modal chain count
        counts = {}
        for p in files[: min(200, len(files))]:
            st = _read(p)
            if st is not None and len(st):
                counts[p] = len(chain_ca_by_resnum(st[0]))
        if not counts:
            raise SystemExit("could not read any structure to choose a reference")
        modal = max(set(counts.values()), key=list(counts.values()).count)
        ref = str(next(p for p, c in counts.items() if c == modal))
    print(f"[canon] {len(files)} structures | reference={Path(ref).name} | workers={workers}")

    tasks = [(p, p.stem + ".pdb") for p in files]
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(ref, max_perms, require_ref_length, output_dir)) as ex:
            for r in ex.map(_work, tasks, chunksize=16):
                rows.append(r)
    else:
        _init_worker(ref, max_perms, require_ref_length, output_dir)
        rows = [_work(t) for t in tasks]

    mdf = pd.DataFrame(rows)
    n_ok = int((mdf.status == "ok").sum())
    n_odd = int(((mdf.status == "ok") & (mdf.parity == 1)).sum())
    print(f"[canon] written {n_ok}/{len(mdf)} | "
          f"odd-chirality (needed transposition fix): {n_odd} "
          f"({100*n_odd/max(n_ok,1):.0f}% of written)")
    drop = mdf[mdf.status != "ok"].status.value_counts().to_dict()
    if drop:
        print(f"[canon] not written: {drop}")
    return mdf


__all__ = [
    "chain_ca_by_resnum", "kabsch_tm", "permutation_parity",
    "best_chain_permutation", "write_canonical", "canonicalize_directory",
]
