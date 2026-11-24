"""PPI v4 feature extractor — produces the EXACT ``master_ppi_v4`` schema.

Background
----------
``xgb_ppi_v4_full`` was trained on ``master_ppi_v4.parquet``, built by the
benchmark pipeline at ``/store/gunwoong/Benchmark/PPI/scripts``
(``p2_extract_features`` + ``p2b_extract_extended`` + ``p2c_dfire_gbind`` +
``p2e_dihedrals`` + ``p3_mpnn`` + ``p5_feature_engineering``).  The model
expects 224 feature columns under those exact names.

The earlier XGBDimer ``base``/``cpu``/``dfire``/``mpnn`` feature groups were a
*re-implementation* that emitted a different (smaller, renamed) column set —
so when the model was bumped to v4 only ~32/224 features matched and the rest
were imputed to train means (≈59 % of the model's gain went dead, scores
collapsed into a narrow band).  This module ports the original extractors
**verbatim** so the master parquet carries every column the model was trained
on, under the training names.

Scope of one row: a single ``(chain_a, chain_b)`` interface — the model is a
*dimer* scorer.  Whole-complex quality features (global plDDT, DFIRE complex,
Ramachandran, H-bond, MPNN-global) are computed over the full predicted
structure; interface-specific features over the ``a × b`` pair; this mirrors
how the benchmark pipeline handled multi-chain predictions.

MPNN sequence-recovery (``overall_recovery`` / ``mpnn_iface_recovery_mean``)
is intentionally **not** computed (requires sampling, negligible gain) — those
two columns stay absent and are imputed to train means at score time.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi

# EMA scorers live in the analysis tree (bundled); resolve via $EMA_PATH.
_EMA_PATH = os.environ.get("EMA_PATH", str(__import__("pathlib").Path(__file__).resolve().parents[3]))
if _EMA_PATH not in sys.path:
    sys.path.insert(0, _EMA_PATH)

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Kyte-Doolittle hydropathy (p2b)
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGED_POS = {"R", "K", "H"}
CHARGED_NEG = {"D", "E"}
AROMATIC = {"F", "W", "Y"}
HYDROPHOBIC = {"A", "V", "L", "I", "M", "F", "W", "Y", "C"}

CONTACT_CB = 8.0
CONTACT_HEAVY = 5.0


# --------------------------------------------------------------------------- #
# p2_extract_features.py  — basic plDDT / iface / JSON confidence
# --------------------------------------------------------------------------- #
def parse_structure(cif_path):
    st = gemmi.read_structure(str(cif_path))
    chains_data, n_breaks = {}, 0
    for model in st:
        for ch in model:
            pl, cb, hv, hri = [], [], [], []
            ridx = 0
            prev = None
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                pl.append(ca.b_iso)
                cb_a = r.find_atom("CB", "*") or ca
                cb.append([cb_a.pos.x, cb_a.pos.y, cb_a.pos.z])
                for at in r:
                    if at.element and at.element.name != "H":
                        hv.append([at.pos.x, at.pos.y, at.pos.z])
                        hri.append(ridx)
                if prev is not None and (r.seqid.num - prev) > 1:
                    n_breaks += 1
                prev = r.seqid.num
                ridx += 1
            if pl:
                pa = np.array(pl, dtype=np.float32)
                if pa.max() < 2.0:
                    pa = pa * 100
                chains_data[ch.name] = dict(
                    plddt=pa,
                    cb=np.array(cb, dtype=np.float32),
                    heavy=np.array(hv, dtype=np.float32),
                    heavy_res=np.array(hri, dtype=np.int32),
                )
        break
    return chains_data, n_breaks


def iface_pair(c1, c2):
    out = dict(n_iface_a=0, n_iface_b=0, n_contacts=0,
               iface_plddt_a=None, iface_plddt_b=None, iface_plddt_combined=None)
    if len(c1["cb"]) == 0 or len(c2["cb"]) == 0:
        return out
    d2 = np.sum((c1["cb"][:, None, :] - c2["cb"][None, :, :]) ** 2, axis=-1)
    cut2 = CONTACT_CB ** 2
    am = (d2 <= cut2).any(1)
    bm = (d2 <= cut2).any(0)
    out["n_iface_a"] = int(am.sum())
    out["n_iface_b"] = int(bm.sum())
    if out["n_iface_a"] > 0:
        out["iface_plddt_a"] = float(c1["plddt"][am].mean())
    if out["n_iface_b"] > 0:
        out["iface_plddt_b"] = float(c2["plddt"][bm].mean())
    if out["n_iface_a"] + out["n_iface_b"] > 0:
        comb = []
        if out["n_iface_a"] > 0:
            comb.append(c1["plddt"][am])
        if out["n_iface_b"] > 0:
            comb.append(c2["plddt"][bm])
        out["iface_plddt_combined"] = float(np.concatenate(comb).mean())
    if len(c1["heavy"]) and len(c2["heavy"]):
        h_d2 = np.sum((c1["heavy"][:, None, :] - c2["heavy"][None, :, :]) ** 2, axis=-1)
        cm = h_d2 <= CONTACT_HEAVY ** 2
        if cm.any():
            ai, bi = np.nonzero(cm)
            pset = set(zip(c1["heavy_res"][ai].tolist(), c2["heavy_res"][bi].tolist()))
            out["n_contacts"] = len(pset)
    return out


def pdockq_b(p, n):
    if n <= 0 or p is None or not np.isfinite(p):
        return 0.0
    x = p * np.log10(n)
    return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018


def _json_features(summary):
    """p2 JSON-confidence features (whole-complex)."""
    out = {}
    if not summary:
        return out
    out["json_iptm"] = summary.get("iptm", np.nan)
    out["json_ptm"] = summary.get("ptm", np.nan)
    out["json_ranking_score"] = summary.get("ranking_score", summary.get("confidence_score", np.nan))
    out["json_plddt"] = summary.get("plddt", summary.get("complex_plddt", np.nan))
    out["json_gpde"] = summary.get("gpde", np.nan)
    out["json_fraction_disordered"] = summary.get("fraction_disordered", summary.get("disorder", np.nan))
    out["json_complex_iplddt"] = summary.get("complex_iplddt", np.nan)
    out["json_complex_pde"] = summary.get("complex_pde", np.nan)
    out["json_complex_ipde"] = summary.get("complex_ipde", np.nan)
    hc = summary.get("has_clash")
    out["json_has_clash"] = int(hc) if hc is not None else np.nan

    cp = summary.get("chain_pair_iptm", summary.get("pair_chains_iptm", None))
    if cp is not None:
        try:
            if isinstance(cp, list):
                arr = np.array(cp, dtype=float)
                if arr.ndim == 2:
                    n = arr.shape[0]
                    vals = arr[~np.eye(n, dtype=bool)]
                    if len(vals):
                        out["chain_pair_iptm_mean"] = float(np.mean(vals))
                        out["chain_pair_iptm_max"] = float(np.max(vals))
                        out["chain_pair_iptm_min"] = float(np.min(vals))
                        out["chain_pair_iptm_n"] = len(vals)
            elif isinstance(cp, dict):
                all_vals = []
                for c1 in cp:
                    sub = cp[c1] if isinstance(cp[c1], dict) else {}
                    for c2 in sub:
                        if c1 != c2 and sub[c2] is not None:
                            all_vals.append(sub[c2])
                if all_vals:
                    out["chain_pair_iptm_mean"] = float(np.mean(all_vals))
                    out["chain_pair_iptm_max"] = float(np.max(all_vals))
                    out["chain_pair_iptm_min"] = float(np.min(all_vals))
                    out["chain_pair_iptm_n"] = len(all_vals)
        except Exception:
            pass

    cp_pae = summary.get("chain_pair_pae_min", summary.get("chain_pair_pae", None))
    if cp_pae is not None:
        try:
            arr = np.array(cp_pae, dtype=float)
            if arr.ndim == 2:
                n = arr.shape[0]
                vals = arr[~np.eye(n, dtype=bool)]
                if len(vals):
                    out["chain_pair_pae_mean"] = float(np.mean(vals))
                    out["chain_pair_pae_min_overall"] = float(np.min(vals))
                    out["chain_pair_pae_max"] = float(np.max(vals))
        except Exception:
            pass

    # chain_iptm / chain_ptm: AF3/Protenix CASP output = dict {chain: value};
    # deepfold3 benchmark = list.  Handle both (training af3 had these populated).
    def _per_chain_vals(x):
        if isinstance(x, dict):
            return [v for v in x.values() if v is not None]
        if isinstance(x, list):
            return [v for v in x if v is not None]
        return []

    ci = _per_chain_vals(summary.get("chain_iptm", None))
    cpt = _per_chain_vals(summary.get("chain_ptm", summary.get("chains_ptm", None)))
    try:
        if ci:
            out["chain_iptm_mean"] = float(np.mean(ci))
            out["chain_iptm_min"] = float(np.min(ci))
            out["chain_iptm_max"] = float(np.max(ci))
        if cpt:
            out["chain_ptm_mean"] = float(np.mean(cpt))
            out["chain_ptm_min"] = float(np.min(cpt))
            out["chain_ptm_max"] = float(np.max(cpt))
    except Exception:
        pass
    return out


def _basic_features(cif_path, summary, chain_a, chain_b):
    chains, nbreaks = parse_structure(cif_path)
    out = {}
    if not chains:
        return out
    all_p = np.concatenate([c["plddt"] for c in chains.values()])
    out["cif_plddt_mean"] = float(all_p.mean())
    out["cif_plddt_std"] = float(all_p.std())
    out["cif_plddt_min"] = float(all_p.min())
    for q in (10, 25, 50, 75, 90):
        out[f"cif_plddt_q{q}"] = float(np.quantile(all_p, q / 100))
    out["cif_plddt_lt70_frac"] = float((all_p < 70).mean())
    out["cif_plddt_lt50_frac"] = float((all_p < 50).mean())
    out["n_chain_breaks"] = nbreaks
    out["n_residues_total"] = int(len(all_p))
    out["n_chains"] = len(chains)

    for label, cid in [("a", chain_a), ("b", chain_b)]:
        c = chains.get(cid)
        if c is None:
            for k in chains:
                if k.startswith(cid):
                    c = chains[k]
                    break
        if c is not None:
            out[f"chain_{label}_plddt_mean"] = float(c["plddt"].mean())
            out[f"chain_{label}_plddt_std"] = float(c["plddt"].std())
            out[f"chain_{label}_n_residues"] = int(len(c["plddt"]))

    if chain_a in chains and chain_b in chains:
        s = iface_pair(chains[chain_a], chains[chain_b])
        out["iface_n_residues_a"] = s["n_iface_a"]
        out["iface_n_residues_b"] = s["n_iface_b"]
        out["iface_n_contacts"] = s["n_contacts"]
        out["iface_plddt_a"] = s["iface_plddt_a"]
        out["iface_plddt_b"] = s["iface_plddt_b"]
        out["iface_plddt_combined"] = s["iface_plddt_combined"]
        out["iface_pdockq"] = pdockq_b(s["iface_plddt_combined"], s["n_contacts"])
        out["iface_anticonf"] = pdockq_b(s["iface_plddt_a"], s["n_contacts"])

    out.update(_json_features(summary))
    return out


# --------------------------------------------------------------------------- #
# p2b_extract_extended.py  — iface chemistry, topology, ss bonds
# --------------------------------------------------------------------------- #
def parse_residues(cif_path):
    st = gemmi.read_structure(str(cif_path))
    residues = []
    for model in st:
        for ch in model:
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                cb = r.find_atom("CB", "*") or ca
                heavy = []
                for at in r:
                    if at.element and at.element.name != "H":
                        heavy.append((at.name, np.array([at.pos.x, at.pos.y, at.pos.z])))
                residues.append({
                    "chain": ch.name, "name": r.name, "aa": AA_3TO1[r.name],
                    "seqid": r.seqid.num,
                    "ca": np.array([ca.pos.x, ca.pos.y, ca.pos.z]),
                    "cb": np.array([cb.pos.x, cb.pos.y, cb.pos.z]),
                    "heavy": heavy, "plddt": ca.b_iso,
                })
        break
    return residues


def _iface_res_idx(residues, chain_a, chain_b, cb_cutoff=CONTACT_CB):
    a_res = [i for i, r in enumerate(residues) if r["chain"] == chain_a]
    b_res = [i for i, r in enumerate(residues) if r["chain"] == chain_b]
    if not a_res or not b_res:
        return [], []
    a_cb = np.array([residues[i]["cb"] for i in a_res])
    b_cb = np.array([residues[i]["cb"] for i in b_res])
    d2 = np.sum((a_cb[:, None, :] - b_cb[None, :, :]) ** 2, axis=-1)
    cut2 = cb_cutoff ** 2
    a_iface = [a_res[i] for i in range(len(a_res)) if (d2[i] <= cut2).any()]
    b_iface = [b_res[i] for i in range(len(b_res)) if (d2[:, i] <= cut2).any()]
    return a_iface, b_iface


def _iface_chemistry(residues, a_iface, b_iface, atom_cutoff=4.0):
    out = {}
    a_res = [residues[i] for i in a_iface]
    b_res = [residues[i] for i in b_iface]
    out["iface_hydropathy_a"] = float(np.mean([HYDROPATHY.get(r["aa"], 0) for r in a_res])) if a_res else None
    out["iface_hydropathy_b"] = float(np.mean([HYDROPATHY.get(r["aa"], 0) for r in b_res])) if b_res else None
    out["iface_hydropathy_combined"] = float(np.mean([HYDROPATHY.get(r["aa"], 0) for r in a_res + b_res])) if (a_res or b_res) else None

    n_salt = n_hphob_pair = n_aromatic_pair = n_charge_complement = 0
    cut2 = atom_cutoff ** 2
    for a in a_res:
        for b in b_res:
            if np.sum((a["ca"] - b["ca"]) ** 2) > 225:
                continue
            close = False
            for _, ac in a["heavy"]:
                for _, bc in b["heavy"]:
                    if np.sum((ac - bc) ** 2) <= cut2:
                        close = True
                        break
                if close:
                    break
            if not close:
                continue
            aa_a, aa_b = a["aa"], b["aa"]
            if (aa_a in CHARGED_POS and aa_b in CHARGED_NEG) or (aa_a in CHARGED_NEG and aa_b in CHARGED_POS):
                n_salt += 1
                n_charge_complement += 1
            if aa_a in HYDROPHOBIC and aa_b in HYDROPHOBIC:
                n_hphob_pair += 1
            if aa_a in AROMATIC and aa_b in AROMATIC:
                n_aromatic_pair += 1
    out["iface_n_salt_bridges"] = n_salt
    out["iface_n_hydrophobic_contacts"] = n_hphob_pair
    out["iface_n_aromatic_contacts"] = n_aromatic_pair
    out["iface_charge_complement"] = n_charge_complement

    aa_a_count = {aa: sum(1 for r in a_res if r["aa"] == aa) for aa in AA_3TO1.values()}
    aa_b_count = {aa: sum(1 for r in b_res if r["aa"] == aa) for aa in AA_3TO1.values()}
    out["iface_n_charged_pos_a"] = sum(aa_a_count[aa] for aa in CHARGED_POS)
    out["iface_n_charged_neg_a"] = sum(aa_a_count[aa] for aa in CHARGED_NEG)
    out["iface_n_charged_pos_b"] = sum(aa_b_count[aa] for aa in CHARGED_POS)
    out["iface_n_charged_neg_b"] = sum(aa_b_count[aa] for aa in CHARGED_NEG)
    out["iface_n_aromatic_a"] = sum(aa_a_count[aa] for aa in AROMATIC)
    out["iface_n_aromatic_b"] = sum(aa_b_count[aa] for aa in AROMATIC)
    return out


def _topology(residues):
    out = {}
    if not residues:
        return out
    cas = np.array([r["ca"] for r in residues])
    centroid = cas.mean(0)
    out["radius_of_gyration"] = float(np.sqrt(((cas - centroid) ** 2).sum(axis=1).mean()))
    centered = cas - centroid
    inertia = np.einsum("ni,nj->ij", centered, centered) / len(centered)
    eigs = sorted(np.linalg.eigvalsh(inertia), reverse=True)
    out["asphericity"] = float(eigs[0] - 0.5 * (eigs[1] + eigs[2]))
    for cid in set(r["chain"] for r in residues):
        ch_cas = np.array([r["ca"] for r in residues if r["chain"] == cid])
        if len(ch_cas) < 5:
            continue
        c = ch_cas.mean(0)
        out[f"radius_of_gyration_{cid}"] = float(np.sqrt(((ch_cas - c) ** 2).sum(axis=1).mean()))
    return out


def _ssbonds(residues):
    cys = [r for r in residues if r["name"] == "CYS"]
    n_intra = n_inter = 0
    for i in range(len(cys)):
        sg1 = next((ac for an, ac in cys[i]["heavy"] if an == "SG"), None)
        if sg1 is None:
            continue
        for j in range(i + 1, len(cys)):
            sg2 = next((ac for an, ac in cys[j]["heavy"] if an == "SG"), None)
            if sg2 is None:
                continue
            if np.linalg.norm(sg1 - sg2) < 2.5:
                if cys[i]["chain"] == cys[j]["chain"]:
                    n_intra += 1
                else:
                    n_inter += 1
    return {"n_ssbond_intra": n_intra, "n_ssbond_inter": n_inter, "n_cys_total": len(cys)}


def _extended_geo(cif_path, chain_a, chain_b):
    out = {}
    residues = parse_residues(cif_path)
    if not residues:
        return out
    a_iface, b_iface = _iface_res_idx(residues, chain_a, chain_b)
    out["iface_n_residues_a_v2"] = len(a_iface)
    out["iface_n_residues_b_v2"] = len(b_iface)
    out.update(_iface_chemistry(residues, a_iface, b_iface))
    out.update(_topology(residues))
    out.update(_ssbonds(residues))
    return out


# --------------------------------------------------------------------------- #
# p2e_dihedrals.py  — phi/psi/omega + Ramachandran
# --------------------------------------------------------------------------- #
def _compute_dihedral(p1, p2, p3, p4):
    b0 = p1 - p2
    b1 = p3 - p2
    b2 = p4 - p3
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def _is_rama_favored(phi, psi, aa):
    if -120 <= phi <= -30 and -75 <= psi <= 0:
        return "favored"
    if -180 <= phi <= -90 and 90 <= psi <= 180:
        return "favored"
    if 30 <= phi <= 90 and 0 <= psi <= 60:
        return "favored"
    if -180 <= phi <= 0 and -180 <= psi <= 180:
        return "allowed"
    if 0 <= phi <= 180 and -90 <= psi <= 90:
        return "allowed"
    return "outlier"


def _extract_phi_psi_omega(cif_path):
    st = gemmi.read_structure(str(cif_path))
    out = []
    for model in st:
        for ch in model:
            res_list = []
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                n = r.find_atom("N", "*")
                ca = r.find_atom("CA", "*")
                c = r.find_atom("C", "*")
                if not (n and ca and c):
                    continue
                cb_a = r.find_atom("CB", "*") or ca
                res_list.append({
                    "chain": ch.name, "seqid": r.seqid.num, "aa": AA_3TO1[r.name],
                    "N": np.array([n.pos.x, n.pos.y, n.pos.z]),
                    "CA": np.array([ca.pos.x, ca.pos.y, ca.pos.z]),
                    "C": np.array([c.pos.x, c.pos.y, c.pos.z]),
                    "CB": np.array([cb_a.pos.x, cb_a.pos.y, cb_a.pos.z]),
                })
            for i, r in enumerate(res_list):
                phi = psi = omega = None
                if i > 0 and res_list[i - 1]["chain"] == r["chain"]:
                    prev = res_list[i - 1]
                    phi = _compute_dihedral(prev["C"], r["N"], r["CA"], r["C"])
                    omega = _compute_dihedral(prev["CA"], prev["C"], r["N"], r["CA"])
                if i < len(res_list) - 1 and res_list[i + 1]["chain"] == r["chain"]:
                    nxt = res_list[i + 1]
                    psi = _compute_dihedral(r["N"], r["CA"], r["C"], nxt["N"])
                r["phi"] = phi
                r["psi"] = psi
                r["omega"] = omega
                out.append(r)
        break
    return out


def _iface_res_idx_dihedral(residues, chain_a, chain_b, cb_cutoff=CONTACT_CB):
    a_res = [(i, r) for i, r in enumerate(residues) if r["chain"] == chain_a]
    b_res = [(i, r) for i, r in enumerate(residues) if r["chain"] == chain_b]
    if not a_res or not b_res:
        return set()
    a_cb = np.array([r["CB"] for _, r in a_res])
    b_cb = np.array([r["CB"] for _, r in b_res])
    d2 = np.sum((a_cb[:, None, :] - b_cb[None, :, :]) ** 2, axis=-1)
    cut2 = cb_cutoff ** 2
    idx = set()
    for j, (i, _) in enumerate(a_res):
        if (d2[j] <= cut2).any():
            idx.add(i)
    for j, (i, _) in enumerate(b_res):
        if (d2[:, j] <= cut2).any():
            idx.add(i)
    return idx


def _dihedral_global(residues):
    """Whole-structure φ/ψ/ω + Ramachandran (pair-independent)."""
    out = {}
    valid = [r for r in residues if r["phi"] is not None and r["psi"] is not None]
    if not valid:
        return out
    phis = np.array([r["phi"] for r in valid])
    psis = np.array([r["psi"] for r in valid])
    out["phi_mean"] = float(phis.mean())
    out["phi_std"] = float(phis.std())
    out["psi_mean"] = float(psis.mean())
    out["psi_std"] = float(psis.std())
    omegas = np.array([r["omega"] for r in residues if r["omega"] is not None])
    if len(omegas):
        out["omega_mean"] = float(omegas.mean())
        out["omega_std"] = float(omegas.std())
        out["omega_n_cis"] = int(np.sum(np.abs(omegas) < 90))
        non_planar = int(np.sum(np.abs(np.abs(omegas) - 180) > 10))
        out["omega_n_nonplanar"] = non_planar
        out["omega_nonplanar_frac"] = non_planar / len(omegas)
    cats = [_is_rama_favored(r["phi"], r["psi"], r["aa"]) for r in valid]
    out["rama_favored_frac"] = cats.count("favored") / len(cats)
    out["rama_allowed_frac"] = cats.count("allowed") / len(cats)
    out["rama_outlier_frac"] = cats.count("outlier") / len(cats)
    out["rama_n_residues"] = len(cats)
    out["rama_n_outliers"] = cats.count("outlier")
    return out


def _dihedral_iface(residues, iface_idx):
    """Interface-residue Ramachandran (pair-specific)."""
    out = {}
    iface_valid = [residues[i] for i in iface_idx
                   if residues[i]["phi"] is not None and residues[i]["psi"] is not None]
    if iface_valid:
        ic = [_is_rama_favored(r["phi"], r["psi"], r["aa"]) for r in iface_valid]
        out["rama_iface_favored_frac"] = ic.count("favored") / len(ic)
        out["rama_iface_outlier_frac"] = ic.count("outlier") / len(ic)
        out["rama_iface_n_outliers"] = ic.count("outlier")
        out["rama_iface_n_residues"] = len(ic)
        out["phi_iface_std"] = float(np.array([r["phi"] for r in iface_valid]).std())
        out["psi_iface_std"] = float(np.array([r["psi"] for r in iface_valid]).std())
    return out


def _dihedral_features(cif_path, chain_a, chain_b):
    out = {}
    residues = _extract_phi_psi_omega(cif_path)
    iface_idx = _iface_res_idx_dihedral(residues, chain_a, chain_b)
    valid = [r for r in residues if r["phi"] is not None and r["psi"] is not None]
    if not valid:
        return out
    phis = np.array([r["phi"] for r in valid])
    psis = np.array([r["psi"] for r in valid])
    out["phi_mean"] = float(phis.mean())
    out["phi_std"] = float(phis.std())
    out["psi_mean"] = float(psis.mean())
    out["psi_std"] = float(psis.std())

    omegas = np.array([r["omega"] for r in residues if r["omega"] is not None])
    if len(omegas):
        out["omega_mean"] = float(omegas.mean())
        out["omega_std"] = float(omegas.std())
        out["omega_n_cis"] = int(np.sum(np.abs(omegas) < 90))
        non_planar = int(np.sum(np.abs(np.abs(omegas) - 180) > 10))
        out["omega_n_nonplanar"] = non_planar
        out["omega_nonplanar_frac"] = non_planar / len(omegas)

    cats = [_is_rama_favored(r["phi"], r["psi"], r["aa"]) for r in valid]
    out["rama_favored_frac"] = cats.count("favored") / len(cats)
    out["rama_allowed_frac"] = cats.count("allowed") / len(cats)
    out["rama_outlier_frac"] = cats.count("outlier") / len(cats)
    out["rama_n_residues"] = len(cats)
    out["rama_n_outliers"] = cats.count("outlier")

    iface_valid = [residues[i] for i in iface_idx
                   if residues[i]["phi"] is not None and residues[i]["psi"] is not None]
    if iface_valid:
        ic = [_is_rama_favored(r["phi"], r["psi"], r["aa"]) for r in iface_valid]
        out["rama_iface_favored_frac"] = ic.count("favored") / len(ic)
        out["rama_iface_outlier_frac"] = ic.count("outlier") / len(ic)
        out["rama_iface_n_outliers"] = ic.count("outlier")
        out["rama_iface_n_residues"] = len(ic)
        out["phi_iface_std"] = float(np.array([r["phi"] for r in iface_valid]).std())
        out["psi_iface_std"] = float(np.array([r["psi"] for r in iface_valid]).std())
    return out


# --------------------------------------------------------------------------- #
# EMA scorers — bbdep / ndrd / hbond (p2b naming) + DFIRE (p2c gbind)
# --------------------------------------------------------------------------- #
class Scorers:
    """Lazy-init heavy scorers, reused across cifs in one process."""

    def __init__(self, device="cuda"):
        # Resolve the requested device against what's actually visible so we
        # never force "cuda" on a CPU-only node (MPNN/DFIRE would crash on .to).
        # When a GPU IS present, this pins every scorer to it instead of leaving
        # MPNN on its "auto" default — that ambiguity is why GPU util read 0%.
        if device in ("auto", "cuda"):
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self._dfire = None
        self._bbdep = None
        self._mpnn = None

    @property
    def dfire(self):
        if self._dfire is None:
            warnings.filterwarnings("ignore")
            from residudes_qa.ema.scorers.dfire_v2_scorer import DFIREv2Scorer
            self._dfire = DFIREv2Scorer(device=self.device)
        return self._dfire

    @property
    def bbdep(self):
        if self._bbdep is None:
            from residudes_qa.ema.scorers.bbdep_chi_spline_v2 import get_lib
            self._bbdep = get_lib()
        return self._bbdep

    @property
    def mpnn(self):
        if self._mpnn is None:
            from residudes_qa.ema.scorers.proteinmpnn_score import MPNNScorer
            # recovery NOT needed → skip sampling for speed. Pass device explicitly
            # so MPNN lands on the GPU (not its CPU-ish "auto" guess) when present.
            try:
                self._mpnn = MPNNScorer(num_sequences=1, sampling_temp=0.1,
                                        skip_sampling=True, device=self.device)
            except TypeError:
                self._mpnn = MPNNScorer(num_sequences=1, sampling_temp=0.1,
                                        device=self.device)
        return self._mpnn


def _bbdep_features(cif_path, scorers):
    try:
        _ = scorers.bbdep  # ensure lib loaded
        from residudes_qa.ema.scorers.bbdep_chi_spline_v2 import score_bbdep_chi_v2
        r = score_bbdep_chi_v2(str(cif_path))
        return {f"bbdep_{k}": float(v) for k, v in r.items()
                if isinstance(v, (int, float, np.number))}
    except Exception:
        return {}


def _ndrd_features(cif_path):
    try:
        from residudes_qa.ema.scorers.ndrd_scorer import score_ndrd
        r = score_ndrd(str(cif_path))
        return {f"ndrd_{k}": float(v) for k, v in r.items()
                if isinstance(v, (int, float, np.number))}
    except Exception:
        return {}


def _hbond_features(cif_path):
    try:
        from residudes_qa.ema.scorers.hbond_network import analyze_hbond_network
        r = analyze_hbond_network(str(cif_path))
        return {f"hbond_{k}": float(v) for k, v in r.items()
                if isinstance(v, (int, float, np.number))}
    except Exception:
        return {}


def _write_chains_cif(src_cif, keep_chains, dst_cif):
    st = gemmi.read_structure(str(src_cif))
    keep = set(keep_chains)
    for model in st:
        for cname in [ch.name for ch in model if ch.name not in keep]:
            model.remove_chain(cname)
        break
    st.make_mmcif_document().write_file(str(dst_cif))


def _score_dfire(scorers, cif_path):
    try:
        r = scorers.dfire.score(str(cif_path))
        return {
            "E_dfire": float(r.get("E_dfire_v2", 0)),
            "E_dfire_per_atom": float(r.get("E_dfire_v2_per_atom", 0)),
            "n_atoms": int(r.get("n_atoms", 0)),
            "n_pairs": int(r.get("n_pairs", 0)),
        }
    except Exception:
        return {}


def _dfire_gbind_features(cif_path, chain_a, chain_b, scorers, tmpdir, tag):
    """p2c: DFIRE on complex / a / b / pair + G_bind decomposition."""
    rec = {}
    comp = _score_dfire(scorers, cif_path)
    for k, v in comp.items():
        rec[f"complex_{k}"] = v
    try:
        ca_path = Path(tmpdir) / f"a_{tag}.cif"
        _write_chains_cif(cif_path, [chain_a], ca_path)
        for k, v in _score_dfire(scorers, ca_path).items():
            rec[f"chain_a_{k}"] = v
        ca_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        cb_path = Path(tmpdir) / f"b_{tag}.cif"
        _write_chains_cif(cif_path, [chain_b], cb_path)
        for k, v in _score_dfire(scorers, cb_path).items():
            rec[f"chain_b_{k}"] = v
        cb_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        pair_path = Path(tmpdir) / f"pair_{tag}.cif"
        _write_chains_cif(cif_path, [chain_a, chain_b], pair_path)
        for k, v in _score_dfire(scorers, pair_path).items():
            rec[f"pair_{k}"] = v
        pair_path.unlink(missing_ok=True)
    except Exception:
        pass
    if all(k in rec for k in ("complex_E_dfire", "chain_a_E_dfire", "chain_b_E_dfire")):
        rec["gbind_dfire_global"] = rec["complex_E_dfire"] - rec["chain_a_E_dfire"] - rec["chain_b_E_dfire"]
    if all(k in rec for k in ("pair_E_dfire", "chain_a_E_dfire", "chain_b_E_dfire")):
        rec["gbind_dfire_pair"] = rec["pair_E_dfire"] - rec["chain_a_E_dfire"] - rec["chain_b_E_dfire"]
    if "gbind_dfire_pair" in rec and rec.get("pair_n_atoms", 0) > 0:
        rec["gbind_dfire_pair_per_atom"] = rec["gbind_dfire_pair"] / rec["pair_n_atoms"]
    return rec


def _chain_pair_pae_features(cif_path):
    """Reconstruct chain_pair_pae_* from ``arrays/pae_logits.npz`` (CASP output).

    CASP ``summary.json`` does not carry a chain-pair PAE matrix, but the
    per-sample ``arrays/pae_logits.npz`` holds the token×token PAE distribution
    (``int8_softmax`` encoded, 64 bins).  We decode expected PAE, aggregate to a
    chain×chain **min**-PAE matrix (the ``chain_pair_pae_min`` semantics the
    training pipeline read from the AF3 summary), then take off-diagonal
    mean/min/max — matching ``p2_extract_features``.
    """
    cif = Path(cif_path)
    npz = cif.parent / "arrays" / "pae_logits.npz"
    if not npz.exists():
        return {}
    try:
        import json as _json
        z = np.load(npz)
        if "pae" not in z:
            return {}
        meta = {}
        if "_meta" in z:
            try:
                meta = _json.loads(bytes(z["_meta"].tolist()).decode("utf-8", "replace"))
            except Exception:
                meta = {}
        scale = float(meta.get("heads", {}).get("pae", {}).get("scale", 16.0 / 255.0))
        logits = z["pae"].astype(np.float32) * scale
        # softmax over bin axis
        mx = logits.max(-1, keepdims=True)
        e = np.exp(logits - mx)
        p = e / e.sum(-1, keepdims=True)
        no_bins = logits.shape[-1]
        max_bin = 31.0
        breaks = np.linspace(0, max_bin, no_bins - 1)
        step = breaks[1] - breaks[0]
        centers = np.concatenate([breaks + step / 2, [breaks[-1] + step / 2]])
        ep = (p * centers).sum(-1)  # (n_tok, n_tok) expected PAE

        # token -> chain slices (CA-bearing amino acids, structure order)
        st = gemmi.read_structure(str(cif_path))
        order = []
        for model in st:
            for ch in model:
                n = sum(1 for r in ch
                        if r.name in AA_3TO1 and r.find_atom("CA", "*") is not None)
                if n:
                    order.append((ch.name, n))
            break
        if sum(n for _, n in order) != ep.shape[0]:
            return {}  # token/residue mismatch — skip rather than misalign
        idx = {}
        cur = 0
        for name, n in order:
            idx[name] = (cur, cur + n)
            cur += n
        names = [c for c, _ in order]
        k = len(names)
        if k < 2:
            return {}
        M = np.zeros((k, k))
        for i, a in enumerate(names):
            sa, ea = idx[a]
            for j, b in enumerate(names):
                sb, eb = idx[b]
                M[i, j] = float(ep[sa:ea, sb:eb].min())
        off = M[~np.eye(k, dtype=bool)]
        return {
            "chain_pair_pae_mean": float(off.mean()),
            "chain_pair_pae_min_overall": float(off.min()),
            "chain_pair_pae_max": float(off.max()),
        }
    except Exception:
        return {}


def _chain_residue_idx(cif_path):
    st = gemmi.read_structure(str(cif_path))
    chain_idx = {}
    cur = 0
    for model in st:
        for ch in model:
            idxs = []
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                if r.find_atom("CA", "*") is None:
                    continue
                idxs.append(cur)
                cur += 1
            if idxs:
                chain_idx[ch.name] = np.array(idxs, dtype=np.int32)
        break
    return chain_idx


def _seed_mpnn():
    """ProteinMPNN uses an unseeded random decoding order (torch.randn) → its
    NLL varies run-to-run by ~0.4%.  Seed before each score for reproducibility
    (and so whole-complex mpnn_score is identical across a complex's interfaces)."""
    try:
        import torch
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except Exception:
        pass


def _mpnn_features(cif_path, chain_a, chain_b, scorers):
    """p3: MPNN global + interface NLL.  Recovery intentionally omitted."""
    rec = {}
    try:
        _seed_mpnn()
        res = scorers.mpnn.score_structure(str(cif_path))
        rec["mpnn_score"] = float(res.mpnn_score)
        per_res = np.asarray(res.per_residue_score, dtype=np.float32)
        chain_idx = _chain_residue_idx(cif_path)
        residues = parse_residues(cif_path)
        a_iface, b_iface = _iface_res_idx(residues, chain_a, chain_b)
        # iface mask over global residue index (residues list order == mpnn order)
        mask = np.zeros(len(per_res), dtype=bool)
        for i in a_iface + b_iface:
            if i < len(mask):
                mask[i] = True
        if mask.any() and len(mask) == len(per_res):
            rec["mpnn_iface_all_mean"] = float(per_res[mask].mean())
            rec["mpnn_iface_n_residues"] = int(mask.sum())
    except Exception:
        pass
    return rec


# --------------------------------------------------------------------------- #
# Top-level row extractor + runner
# --------------------------------------------------------------------------- #
def extract_ppi_v4_row(cif_path, summary, chain_a, chain_b, scorers, tmpdir, tag):
    """Full master_ppi_v4 base-feature dict for one (chain_a, chain_b)."""
    out = {}
    out.update(_basic_features(cif_path, summary, chain_a, chain_b))
    try:
        out.update(_extended_geo(cif_path, chain_a, chain_b))
    except Exception:
        pass
    try:
        out.update(_dihedral_features(cif_path, chain_a, chain_b))
    except Exception:
        pass
    out.update(_bbdep_features(cif_path, scorers))
    out.update(_ndrd_features(cif_path))
    out.update(_hbond_features(cif_path))
    out.update(_dfire_gbind_features(cif_path, chain_a, chain_b, scorers, tmpdir, tag))
    out.update(_mpnn_features(cif_path, chain_a, chain_b, scorers))
    out.update(_chain_pair_pae_features(cif_path))
    return out


def detect_interface_pairs(cif_path, min_contacts=10):
    """Auto-detect contacting chain pairs in a structure.

    Returns ``[(a, b), …]`` for every chain pair whose heavy-atom contact count
    (``iface_pair``, 5 Å) is ``>= min_contacts``, sorted by descending contacts
    so the strongest interfaces come first.  Used by ``run-ppi-v4`` when the
    caller does not pass ``--pairs`` / ``--chain-a``.
    """
    chains, _ = parse_structure(cif_path)
    names = list(chains.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            s = iface_pair(chains[a], chains[b])
            if s["n_contacts"] >= min_contacts:
                pairs.append((a, b, s["n_contacts"]))
    pairs.sort(key=lambda x: -x[2])
    return [(a, b) for a, b, _ in pairs]


def find_summary_json(cif_path):
    """Locate the summary/confidence JSON paired with a cif (p2 logic)."""
    import json as _json
    cif = Path(cif_path)
    stem = cif.stem.replace("_postprocessed", "")
    candidates = [
        cif.parent / "summary.json",
        cif.parent / f"{stem}_summary_confidences.json",
        cif.parent / f"{stem}_confidences.json",
        cif.parent / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    for pat in ("*summary*.json", "*confidences*.json"):
        for p in cif.parent.glob(pat):
            return p
    return None


# method_master canonicalization (mirror core.master / xgb training universe).
# MUST match METHOD_MASTER_MAP in xgbdimer/core/master.py — bt2 (plain Boltz-2)
# was previously missing here, so bt2 rows stayed 'bt2' (unknown to the model)
# and got unfairly penalized at score time. bt2 and bt2_jax are both Boltz-2 → b2.
_METHOD_MASTER = {
    "af3": "af3",
    "bt2": "b2", "bt2_jax": "b2", "b2": "b2",
    "of3": "of3",
    "pt2": "v2", "v2": "v2", "ptx": "ptx", "ptx_v2": "ptx_v2",
    "ef2": "ef2", "esmfold2": "ef2", "single_seq": "ef2",
    "af2m1": "af3", "af2m2": "af3", "af2m3": "af3",
    "af2m4": "af3", "af2m5": "af3", "afm": "af3",
}


def run_ppi_v4(index_df, chain_a, chain_b, device="cuda", progress_every=50):
    """Extract the full v4 base schema for every cif in ``index_df``.

    ``index_df`` columns expected: ``model_id``, ``src_cif`` (+ optional meta
    ``target``/``pdb``/``method``/``seed``/``sample``/``variant``).  One output
    row per cif for the single ``(chain_a, chain_b)`` interface.
    """
    import json as _json
    scorers = Scorers(device=device)
    rows = []
    n = len(index_df)
    meta_cols = [c for c in ("target", "pdb", "method", "seed", "sample",
                             "variant", "candidate_uid") if c in index_df.columns]
    with tempfile.TemporaryDirectory() as tmpdir:
        for k, r in enumerate(index_df.itertuples(index=False)):
            d = r._asdict()
            rec = {"model_id": d["model_id"], "chain_a": chain_a, "chain_b": chain_b}
            for c in meta_cols:
                rec[c] = d.get(c)
            cif = d.get("src_cif") or d.get("cif_path")
            try:
                if not cif or not Path(cif).exists():
                    rec["err"] = "cif missing"
                    rows.append(rec)
                    continue
                sj_path = find_summary_json(cif)
                summary = {}
                if sj_path:
                    try:
                        summary = _json.loads(Path(sj_path).read_text())
                    except Exception:
                        summary = {}
                feats = extract_ppi_v4_row(cif, summary, chain_a, chain_b,
                                           scorers, tmpdir, tag=str(k))
                rec.update(feats)
            except Exception as e:
                rec["err"] = f"{type(e).__name__}: {str(e)[:80]}"
            rows.append(rec)
            if (k + 1) % progress_every == 0:
                print(f"    [ppi_v4 {k+1}/{n}] errors="
                      f"{sum('err' in x for x in rows)}", flush=True)
    df = pd.DataFrame(rows)
    # method_master for one-hot at score time
    if "method" in df.columns:
        df["method_master"] = df["method"].astype(str).map(
            lambda m: _METHOD_MASTER.get(m, m))
    return df


# --------------------------------------------------------------------------- #
# Optimized multi-interface extraction.
#
# Scoring N interfaces of the SAME complex with the per-pair path above
# recomputes every whole-complex feature N times (global plDDT, JSON conf,
# chain_pair_iptm/pae, topology, ssbond, global Ramachandran, bbdep/ndrd/hbond,
# complex DFIRE, whole-structure MPNN, the 155 MB pae_logits decode) — and the
# shared chain's per-chain DFIRE too.  ``run_ppi_v4_multi`` computes each cif's
# whole-complex block + per-chain DFIRE ONCE and reuses them across pairs, so
# only genuinely pair-specific features (iface geometry/chemistry/Ramachandran,
# iface MPNN, the a+b pair DFIRE, ΔG_bind) are recomputed.  Results are
# bit-identical to the per-pair path.
# --------------------------------------------------------------------------- #
def _global_plddt_block(chains, nbreaks):
    out = {}
    all_p = np.concatenate([c["plddt"] for c in chains.values()])
    out["cif_plddt_mean"] = float(all_p.mean())
    out["cif_plddt_std"] = float(all_p.std())
    out["cif_plddt_min"] = float(all_p.min())
    for q in (10, 25, 50, 75, 90):
        out[f"cif_plddt_q{q}"] = float(np.quantile(all_p, q / 100))
    out["cif_plddt_lt70_frac"] = float((all_p < 70).mean())
    out["cif_plddt_lt50_frac"] = float((all_p < 50).mean())
    out["n_chain_breaks"] = nbreaks
    out["n_residues_total"] = int(len(all_p))
    out["n_chains"] = len(chains)
    return out


def _chain_plddt_block(chains, label, cid):
    out = {}
    c = chains.get(cid)
    if c is None:
        for k in chains:
            if k.startswith(cid):
                c = chains[k]
                break
    if c is not None:
        out[f"chain_{label}_plddt_mean"] = float(c["plddt"].mean())
        out[f"chain_{label}_plddt_std"] = float(c["plddt"].std())
        out[f"chain_{label}_n_residues"] = int(len(c["plddt"]))
    return out


def _iface_basic_block(chains, a, b):
    out = {}
    if a in chains and b in chains:
        s = iface_pair(chains[a], chains[b])
        out["iface_n_residues_a"] = s["n_iface_a"]
        out["iface_n_residues_b"] = s["n_iface_b"]
        out["iface_n_contacts"] = s["n_contacts"]
        out["iface_plddt_a"] = s["iface_plddt_a"]
        out["iface_plddt_b"] = s["iface_plddt_b"]
        out["iface_plddt_combined"] = s["iface_plddt_combined"]
        out["iface_pdockq"] = pdockq_b(s["iface_plddt_combined"], s["n_contacts"])
        out["iface_anticonf"] = pdockq_b(s["iface_plddt_a"], s["n_contacts"])
    return out


def run_ppi_v4_multi(index_df, pairs, device="cuda", progress_every=50):
    """Extract the v4 base schema for MANY interfaces, sharing whole-complex work.

    Returns ``{(a, b): DataFrame}`` — one frame per interface, each with the
    same columns/values the per-pair :func:`run_ppi_v4` would produce.
    """
    import json as _json
    scorers = Scorers(device=device)
    n = len(index_df)
    meta_cols = [c for c in ("target", "pdb", "method", "seed", "sample",
                             "variant", "candidate_uid") if c in index_df.columns]
    pair_rows = {p: [] for p in pairs}

    with tempfile.TemporaryDirectory() as tmpdir:
        for k, r in enumerate(index_df.itertuples(index=False)):
            d = r._asdict()
            base_meta = {"model_id": d["model_id"]}
            for c in meta_cols:
                base_meta[c] = d.get(c)
            cif = d.get("src_cif") or d.get("cif_path")

            if not cif or not Path(cif).exists():
                for (a, b) in pairs:
                    pair_rows[(a, b)].append({**base_meta, "chain_a": a,
                                              "chain_b": b, "err": "cif missing"})
                continue

            try:
                # ---- parse + summary (once) ----
                sj_path = find_summary_json(cif)
                summary = {}
                if sj_path:
                    try:
                        summary = _json.loads(Path(sj_path).read_text())
                    except Exception:
                        summary = {}
                chains, nbreaks = parse_structure(cif)
                residues = parse_residues(cif)
                dih = _extract_phi_psi_omega(cif)

                # ---- whole-complex block (once) ----
                W = {}
                if chains:
                    W.update(_global_plddt_block(chains, nbreaks))
                W.update(_json_features(summary))
                try:
                    W.update(_topology(residues))
                    W.update(_ssbonds(residues))
                except Exception:
                    pass
                try:
                    W.update(_dihedral_global(dih))
                except Exception:
                    pass
                W.update(_bbdep_features(cif, scorers))
                W.update(_ndrd_features(cif))
                W.update(_hbond_features(cif))
                # complex DFIRE (whole)
                comp = _score_dfire(scorers, cif)
                for kk, vv in comp.items():
                    W[f"complex_{kk}"] = vv
                # whole-structure MPNN (per-residue NLL reused for every iface mask)
                per_res = None
                try:
                    _seed_mpnn()
                    res = scorers.mpnn.score_structure(str(cif))
                    W["mpnn_score"] = float(res.mpnn_score)
                    per_res = np.asarray(res.per_residue_score, dtype=np.float32)
                except Exception:
                    pass
                W.update(_chain_pair_pae_features(cif))

                # ---- per-chain DFIRE cache (each chain scored once) ----
                dfire_chain = {}

                def _chain_dfire(name):
                    if name not in dfire_chain:
                        try:
                            cpath = Path(tmpdir) / f"c_{k}_{name}.cif"
                            _write_chains_cif(cif, [name], cpath)
                            dfire_chain[name] = _score_dfire(scorers, cpath)
                            cpath.unlink(missing_ok=True)
                        except Exception:
                            dfire_chain[name] = {}
                    return dfire_chain[name]

                # ---- per pair: only pair-specific features ----
                for (a, b) in pairs:
                    rec = {**base_meta, "chain_a": a, "chain_b": b}
                    rec.update(W)
                    rec.update(_chain_plddt_block(chains, "a", a))
                    rec.update(_chain_plddt_block(chains, "b", b))
                    rec.update(_iface_basic_block(chains, a, b))
                    # iface chemistry
                    try:
                        ai, bi = _iface_res_idx(residues, a, b)
                        rec["iface_n_residues_a_v2"] = len(ai)
                        rec["iface_n_residues_b_v2"] = len(bi)
                        rec.update(_iface_chemistry(residues, ai, bi))
                    except Exception:
                        ai = bi = []
                    # iface Ramachandran
                    try:
                        idx = _iface_res_idx_dihedral(dih, a, b)
                        rec.update(_dihedral_iface(dih, idx))
                    except Exception:
                        pass
                    # iface MPNN (reuse whole per_res)
                    if per_res is not None:
                        mask = np.zeros(len(per_res), dtype=bool)
                        for i in list(ai) + list(bi):
                            if i < len(mask):
                                mask[i] = True
                        if mask.any():
                            rec["mpnn_iface_all_mean"] = float(per_res[mask].mean())
                            rec["mpnn_iface_n_residues"] = int(mask.sum())
                    # DFIRE pair + ΔG_bind (chain a/b from cache, pair recomputed)
                    sa = _chain_dfire(a)
                    sb = _chain_dfire(b)
                    for kk, vv in sa.items():
                        rec[f"chain_a_{kk}"] = vv
                    for kk, vv in sb.items():
                        rec[f"chain_b_{kk}"] = vv
                    try:
                        ppath = Path(tmpdir) / f"pair_{k}_{a}{b}.cif"
                        _write_chains_cif(cif, [a, b], ppath)
                        for kk, vv in _score_dfire(scorers, ppath).items():
                            rec[f"pair_{kk}"] = vv
                        ppath.unlink(missing_ok=True)
                    except Exception:
                        pass
                    if all(x in rec for x in ("complex_E_dfire", "chain_a_E_dfire", "chain_b_E_dfire")):
                        rec["gbind_dfire_global"] = rec["complex_E_dfire"] - rec["chain_a_E_dfire"] - rec["chain_b_E_dfire"]
                    if all(x in rec for x in ("pair_E_dfire", "chain_a_E_dfire", "chain_b_E_dfire")):
                        rec["gbind_dfire_pair"] = rec["pair_E_dfire"] - rec["chain_a_E_dfire"] - rec["chain_b_E_dfire"]
                    if "gbind_dfire_pair" in rec and rec.get("pair_n_atoms", 0) > 0:
                        rec["gbind_dfire_pair_per_atom"] = rec["gbind_dfire_pair"] / rec["pair_n_atoms"]
                    pair_rows[(a, b)].append(rec)
            except Exception as e:
                for (a, b) in pairs:
                    pair_rows[(a, b)].append({**base_meta, "chain_a": a,
                                              "chain_b": b,
                                              "err": f"{type(e).__name__}: {str(e)[:80]}"})

            if (k + 1) % progress_every == 0:
                done = sum(len(v) for v in pair_rows.values())
                print(f"    [ppi_v4_multi cif {k+1}/{n}] rows={done}", flush=True)

    out = {}
    for p, rows in pair_rows.items():
        df = pd.DataFrame(rows)
        if "method" in df.columns:
            df["method_master"] = df["method"].astype(str).map(
                lambda m: _METHOD_MASTER.get(m, m))
        out[p] = df
    return out


# --------------------------------------------------------------------------- #
# p5_feature_engineering.py — base-level derived columns the model expects.
# (method one-hot, z_method, z_target are produced by score.prep_xgb_ppi_input;
#  here we add only the raw/base-level engineered columns it looks up directly.)
# --------------------------------------------------------------------------- #
TARGET_RANK_HIGH_GOOD = [
    "json_iptm", "json_ptm", "json_ranking_score", "json_plddt",
    "cif_plddt_mean", "iface_plddt_combined", "iface_pdockq", "pdockq_max",
    "chain_pair_iptm_mean", "chain_pair_iptm_min", "iface_n_contacts",
    "mpnn_score", "overall_recovery", "mpnn_iface_all_mean",
]
TARGET_RANK_LOW_GOOD = [
    "chain_pair_pae_mean", "chain_pair_pae_min_overall",
    "rama_outlier_frac", "rama_iface_outlier_frac", "omega_nonplanar_frac",
    "gbind_dfire_pair", "gbind_dfire_global",
    "complex_E_dfire_per_atom",
    "cif_plddt_lt70_frac", "cif_plddt_lt50_frac",
]


def engineer_ppi_v4(df, target_col="target"):
    """Add p5 base-level engineered columns (size logs, iptm-ptm, target ranks).

    Ranks are computed within ``target_col`` (mirrors p5 ``groupby('pdb')``);
    if the column is absent the whole frame is treated as one target.
    """
    df = df.copy()
    if "n_residues_total" in df:
        df["log_L_total"] = np.log1p(df["n_residues_total"].astype(float))
    if "n_chains" in df:
        df["log_N_chain"] = np.log1p(df["n_chains"].astype(float))
    if "complex_n_atoms" in df:
        df["log_N_atom"] = np.log1p(df["complex_n_atoms"].astype(float))
    if "iface_n_contacts" in df:
        df["log_iface_n_contacts"] = np.log1p(df["iface_n_contacts"].astype(float))
    if "json_iptm" in df and "json_ptm" in df:
        df["iptm_minus_ptm"] = df["json_iptm"] - df["json_ptm"]

    grp = df[target_col] if target_col in df.columns else pd.Series(0, index=df.index)
    for c in TARGET_RANK_HIGH_GOOD:
        if c in df.columns:
            df[f"{c}_rank_target_high"] = df.groupby(grp)[c].transform(lambda x: x.rank(pct=True))
    for c in TARGET_RANK_LOW_GOOD:
        if c in df.columns:
            df[f"{c}_rank_target_low"] = df.groupby(grp)[c].transform(lambda x: (-x).rank(pct=True))
    return df
