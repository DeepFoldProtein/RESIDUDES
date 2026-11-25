"""Repair mis-placed C-terminal carboxyl oxygens (the Protenix terminal-O bug).

Some predictors (notably pt2 / Protenix) place the terminal residue's carboxyl
oxygen ``O`` / ``OXT`` tens to >100 Å away from its own CA. The backbone, CA
trace, and interface are otherwise fine — so we **keep the model** and only
snap the offending oxygen(s) back onto sane sp2 carboxyl geometry, recording
exactly which models were touched.

Geometry: at the terminal carbon ``C`` (sp2), the three bonds to ``CA``, ``O``,
``OXT`` lie ~120° apart in the local backbone plane (N–CA–C). We place each bad
oxygen 1.25 Å from ``C`` along ``C→CA`` rotated ±120° about the plane normal.

Downstream analysis (contact map, clustering, xgb) is already robust to the
artifact; this repair is for *submission-quality* CIFs and any geometry-sensitive
scorer.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd


def _rotate(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of vector v about unit axis by angle_deg."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.radians(angle_deg)
    return (v * np.cos(th)
            + np.cross(axis, v) * np.sin(th)
            + axis * np.dot(axis, v) * (1 - np.cos(th)))


def repair_terminal_oxygens(
    cif_in: str | Path,
    cif_out: str | Path,
    *,
    max_outlier_dist: float = 10.0,
    bond_len: float = 1.25,
) -> dict:
    """Reposition any terminal ``O``/``OXT`` farther than ``max_outlier_dist`` Å
    from its CA. Writes a repaired CIF only if something was fixed.

    Returns ``{repaired, atoms_fixed, max_orig_dist}``.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_in))
    fixed: list[str] = []
    max_orig = 0.0
    if len(st) == 0:
        return {"repaired": False, "atoms_fixed": [], "max_orig_dist": 0.0}
    model = st[0]
    for ch in model:
        residues = [r for r in ch]
        if not residues:
            continue
        last = residues[-1]
        ca = last.find_atom("CA", "*")
        c = last.find_atom("C", "*")
        n = last.find_atom("N", "*")
        if ca is None or c is None:
            continue
        ca_p = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
        c_p = np.array([c.pos.x, c.pos.y, c.pos.z])
        u = ca_p - c_p
        u = u / (np.linalg.norm(u) + 1e-12)            # C -> CA
        if n is not None:
            n_p = np.array([n.pos.x, n.pos.y, n.pos.z])
            normal = np.cross(ca_p - c_p, n_p - c_p)
        else:
            # arbitrary normal perpendicular to u
            normal = np.cross(u, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(normal) < 1e-6:
                normal = np.cross(u, np.array([0.0, 1.0, 0.0]))
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        for atom_name, angle in (("O", 120.0), ("OXT", -120.0)):
            at = last.find_atom(atom_name, "*")
            if at is None:
                continue
            at_p = np.array([at.pos.x, at.pos.y, at.pos.z])
            d = float(np.linalg.norm(at_p - ca_p))
            if d > max_outlier_dist:
                max_orig = max(max_orig, d)
                direction = _rotate(u, normal, angle)
                new_p = c_p + bond_len * direction
                at.pos = gemmi.Position(float(new_p[0]), float(new_p[1]), float(new_p[2]))
                fixed.append(f"{ch.name}.{last.seqid.num}.{atom_name}({d:.0f}A)")

    if fixed:
        cif_out = Path(cif_out)
        cif_out.parent.mkdir(parents=True, exist_ok=True)
        st.setup_entities()
        st.make_mmcif_document().write_file(str(cif_out))
    return {"repaired": bool(fixed), "atoms_fixed": fixed, "max_orig_dist": float(max_orig)}


def _repair_worker(arg):
    model_id, cif_in, cif_out, max_outlier_dist = arg
    try:
        res = repair_terminal_oxygens(cif_in, cif_out, max_outlier_dist=max_outlier_dist)
        return {
            "model_id": model_id,
            "repaired": res["repaired"],
            "atoms_fixed": ";".join(res["atoms_fixed"]),
            "max_orig_dist": round(res["max_orig_dist"], 1),
            "effective_cif": str(cif_out) if res["repaired"] else str(cif_in),
        }
    except Exception as e:
        return {"model_id": model_id, "repaired": False, "atoms_fixed": "",
                "max_orig_dist": np.nan, "effective_cif": str(cif_in),
                "err": f"{type(e).__name__}: {e}"}


def repair_batch(
    records: pd.DataFrame,
    out_dir: str | Path,
    *,
    model_col: str = "model_id",
    cif_col: str = "src_cif",
    max_outlier_dist: float = 10.0,
    workers: int = 4,
) -> pd.DataFrame:
    """Repair a batch; write repaired CIFs to ``out_dir/cifs/`` and a manifest.

    Returns the manifest DataFrame: ``model_id, repaired, atoms_fixed,
    max_orig_dist, effective_cif`` (effective_cif = repaired path if fixed, else
    the original — so a pool can point at one column and get clean geometry).
    """
    out_dir = Path(out_dir)
    cif_dir = out_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    args = []
    for _, r in records.iterrows():
        mid = r[model_col]
        safe = str(mid).replace("/", "__")
        args.append((mid, r[cif_col], cif_dir / f"{safe}.cif", max_outlier_dist))

    if workers > 1 and len(args) > 1:
        with mp.Pool(workers) as pool:
            rows = pool.map(_repair_worker, args)
    else:
        rows = [_repair_worker(a) for a in args]

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "repair_manifest.csv", index=False)
    return manifest


__all__ = ["repair_terminal_oxygens", "repair_batch"]
