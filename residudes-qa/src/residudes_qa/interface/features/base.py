"""Base feature extraction — xgb_v8 schema, target-agnostic.

Adapted from H1335 ``20_xgb_base_features.py``.  The H1335 version hard-coded
the chain layout (Fab = D/E, antigen = A/B/C); this module accepts a
:class:`ChainConfig` so the same code works for arbitrary CASP17 targets and
even PPI (no antibody) mode.

Produced columns (a subset; the full set follows the spec in the module
docstring):

* pLDDT distribution
    ``cif_plddt_{mean,std,min,q10,q25,q50,q75,q90,lt70_frac}``,
    ``n_chain_breaks``, ``n_residues_total``

* AF3 summary.json
    ``json_iptm``, ``json_ptm``, ``json_ranking_score``, ``json_mean_plddt``,
    ``json_fraction_disordered``, ``json_gpde``, ``json_has_clash``,
    ``json_complex_iplddt``, ``json_complex_ipde``, ``json_complex_pde``,
    ``json_num_recycles``

* Interface (Ab × Ag pairs in Ab mode; all-vs-all in PPI mode)
    ``iface_n_residues_ab_total``, ``iface_n_residues_ag_total``,
    ``iface_n_contacts_total``,
    ``iface_plddt_{ab,ag,combined}_{max,mean}``,
    ``pdockq_{max,mean}``, ``anticonf_{max,mean}``

* chain-pair iPTM aggregates
    ``ab_ag_chain_pair_iptm_{mean,max,min,n}``, ``iface_chain_pair_iptm``,
    ``ab_only_chain_iptm``, ``ab_only_chain_ptm``,
    ``antigen_chain_iptm_mean``, ``antigen_chain_ptm_mean``

* Per-chain confidence (Ab-mode only when ``heavy`` / ``light`` provided)
    ``heavy_chain_{plddt_cif,iptm,ptm}``, ``light_chain_{plddt_cif,iptm,ptm}``,
    ``vh_vl_pair_iptm``, ``ab_only_chain_plddt_cif``,
    ``ab_scfv_chain_plddt_cif``, ``antigen_chain_plddt_cif_mean``

All output dicts include ``model_id`` so the resulting DataFrame can be joined
back onto the index from :mod:`residudes_qa.interface.core.discover`.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# Distance cutoffs (Å) used by H1335 stage-1.
CONTACT_CB_CUTOFF = 8.0
CONTACT_HEAVY_CUTOFF = 5.0
DEFAULT_SANITIZE_CUTOFF = 10.0  # pt2 terminal-O outliers — see parse_structure

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class ChainConfig:
    """Chain-role configuration for one target.

    For an antibody-antigen complex set both ``ab_chains`` and ``ag_chains``
    (and optionally ``heavy_chain`` / ``light_chain``).  For protein-protein
    (PPI) mode set ``ab_chains=[]`` — interface pairs are then formed by all
    pairs of distinct chains and only the chain-pair-iPTM / pair-iface
    aggregates are produced (the Ab/Ag named columns stay NaN).

    ``scfv_chain`` is used for single-chain Fv targets that contribute to
    ``ab_scfv_chain_plddt_cif``.
    """

    ab_chains: list[str] = field(default_factory=list)
    ag_chains: list[str] = field(default_factory=list)
    heavy_chain: Optional[str] = None
    light_chain: Optional[str] = None
    scfv_chain: Optional[str] = None

    @property
    def mode(self) -> str:
        return "ab" if self.ab_chains else "ppi"


# ----- structure parsing -----------------------------------------------------


def parse_structure(cif_path: str | Path, sanitize_cutoff: float = DEFAULT_SANITIZE_CUTOFF):
    """Read CIF, return ``(chains_data, n_chain_breaks)``.

    ``chains_data[chain_id]`` keys: ``plddt`` (np.float32, scaled to 0-100 if
    stored as fraction), ``cb`` (CB or CA-fallback xyz), ``heavy``
    (non-hydrogen xyz), ``heavy_res`` (residue index per heavy atom), ``com``
    (CB centroid).

    ``sanitize_cutoff`` drops residues whose CA is more than this many Å from
    the chain CA-centroid — guards against pt2 / AF3 terminal-O blow-ups that
    otherwise contaminate the interface CB distance matrix.  Set to ``None``
    or a very large value to disable.
    """
    import gemmi  # local: keep module import-time cheap

    st = gemmi.read_structure(str(cif_path))
    chains_data: dict[str, dict[str, Any]] = {}
    n_breaks = 0
    for model in st:
        for ch in model:
            pl: list[float] = []
            cb: list[list[float]] = []
            ca_xyz: list[list[float]] = []
            hv: list[list[float]] = []
            hri: list[int] = []
            ridx = 0
            prev: Optional[int] = None
            for r in ch:
                if r.name not in AA_3TO1:
                    continue
                ca = r.find_atom("CA", "*")
                if ca is None:
                    continue
                pl.append(ca.b_iso)
                cb_a = r.find_atom("CB", "*") or ca
                cb.append([cb_a.pos.x, cb_a.pos.y, cb_a.pos.z])
                ca_xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
                for at in r:
                    if at.element and at.element.name != "H":
                        hv.append([at.pos.x, at.pos.y, at.pos.z])
                        hri.append(ridx)
                if prev is not None and (r.seqid.num - prev) > 1:
                    n_breaks += 1
                prev = r.seqid.num
                ridx += 1
            if not pl:
                continue
            pa = np.asarray(pl, dtype=np.float32)
            if pa.max() < 2.0:  # AF3 sometimes writes 0..1
                pa = pa * 100
            cb_arr = np.asarray(cb, dtype=np.float32)
            ca_arr = np.asarray(ca_xyz, dtype=np.float32)

            # ---- sanitize: drop CA outliers > sanitize_cutoff from centroid
            keep = np.ones(len(ca_arr), dtype=bool)
            if sanitize_cutoff is not None and len(ca_arr) > 2:
                centroid = ca_arr.mean(axis=0)
                d = np.linalg.norm(ca_arr - centroid, axis=1)
                # We only flag points absurdly far from BOTH centroid AND the
                # bulk (3 * median).  pt2 outliers tend to be 10+ Å detached
                # from the chain trace.
                med = np.median(d) if len(d) else 0.0
                bad = (d > sanitize_cutoff) & (d > 3 * med + 5.0)
                if bad.any():
                    keep = ~bad
                    pa = pa[keep]
                    cb_arr = cb_arr[keep]
                    ca_arr = ca_arr[keep]
                    # Heavy atoms: drop those whose residue was filtered.
                    kept_idx = np.flatnonzero(keep)
                    keep_set = set(int(i) for i in kept_idx)
                    remap = {old: new for new, old in enumerate(kept_idx)}
                    hv_new, hri_new = [], []
                    for xyz, ri in zip(hv, hri):
                        if ri in keep_set:
                            hv_new.append(xyz)
                            hri_new.append(remap[ri])
                    hv, hri = hv_new, hri_new

            chains_data[ch.name] = dict(
                plddt=pa,
                cb=cb_arr,
                heavy=np.asarray(hv, dtype=np.float32) if hv else np.zeros((0, 3), np.float32),
                heavy_res=np.asarray(hri, dtype=np.int32),
                com=cb_arr.mean(axis=0) if len(cb_arr) else np.zeros(3, np.float32),
            )
        break  # only model 1
    return chains_data, n_breaks


# ----- interface pair geometry ----------------------------------------------


def iface_pair(ab: dict, ag: dict) -> dict:
    """Per-pair interface stats (mirrors H1335 stage-1 ``iface_pair``)."""
    out: dict[str, Any] = dict(
        n_iface_ab=0, n_iface_ag=0, n_contacts=0,
        iface_plddt_ab=None, iface_plddt_ag=None, iface_plddt_combined=None,
    )
    if len(ab["cb"]) == 0 or len(ag["cb"]) == 0:
        return out
    d2 = np.sum((ab["cb"][:, None, :] - ag["cb"][None, :, :]) ** 2, axis=-1)
    cut2 = CONTACT_CB_CUTOFF ** 2
    am = (d2 <= cut2).any(1)
    gm = (d2 <= cut2).any(0)
    out["n_iface_ab"] = int(am.sum())
    out["n_iface_ag"] = int(gm.sum())
    if out["n_iface_ab"] > 0:
        out["iface_plddt_ab"] = float(ab["plddt"][am].mean())
    if out["n_iface_ag"] > 0:
        out["iface_plddt_ag"] = float(ag["plddt"][gm].mean())
    if out["n_iface_ab"] + out["n_iface_ag"] > 0:
        comb = []
        if out["n_iface_ab"] > 0:
            comb.append(ab["plddt"][am])
        if out["n_iface_ag"] > 0:
            comb.append(ag["plddt"][gm])
        out["iface_plddt_combined"] = float(np.concatenate(comb).mean())
    if len(ab["heavy"]) and len(ag["heavy"]):
        h_d2 = np.sum((ab["heavy"][:, None, :] - ag["heavy"][None, :, :]) ** 2, axis=-1)
        cm = h_d2 <= CONTACT_HEAVY_CUTOFF ** 2
        if cm.any():
            ai, gi = np.nonzero(cm)
            pset = set(zip(ab["heavy_res"][ai].tolist(), ag["heavy_res"][gi].tolist()))
            out["n_contacts"] = len(pset)
    return out


def pdockq_b(p: Optional[float], n: int) -> float:
    """Bryant et al. pDockQ; returns 0.0 when n=0 or plddt is missing."""
    if n <= 0 or p is None or not np.isfinite(p):
        return 0.0
    x = p * np.log10(n)
    return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018


def _agg(pairs: list[dict], key: str, fn):
    vals = [p[key] for p in pairs if p.get(key) is not None]
    return fn(vals) if vals else None


# ----- main extraction -------------------------------------------------------


def extract_base_features(
    cif_path: str | Path,
    summary_path: str | Path,
    chain_config: ChainConfig,
    sanitize_cutoff: float = DEFAULT_SANITIZE_CUTOFF,
) -> dict:
    """Extract the full xgb_v8 base-feature row for one (cif, summary).

    The returned dict always contains the input paths and any error info under
    ``extract_err``.  Callers should attach ``model_id`` from the index.
    """
    out: dict[str, Any] = {
        "cif_path": str(cif_path),
        "summary_path": str(summary_path),
    }

    # ---- summary.json -------------------------------------------------------
    summary: dict[str, Any] = {}
    if summary_path:
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception as e:  # noqa: BLE001
            out["extract_err"] = f"summary read: {type(e).__name__}"
            return out

    # ---- structure ----------------------------------------------------------
    try:
        chains, nbreaks = parse_structure(cif_path, sanitize_cutoff=sanitize_cutoff)
    except Exception as e:  # noqa: BLE001
        out["extract_err"] = f"cif parse: {type(e).__name__}: {str(e)[:80]}"
        return out
    if not chains:
        out["extract_err"] = "no chains"
        return out

    cfg = chain_config
    ab_chains = [c for c in cfg.ab_chains if c in chains]
    ag_chains = [c for c in cfg.ag_chains if c in chains]

    # ---- pLDDT distribution over modelled chains ----------------------------
    used = set(ab_chains) | set(ag_chains)
    if not used:  # PPI mode w/ no whitelisting: use everything
        used = set(chains.keys())
    if cfg.mode == "ppi" and not cfg.ag_chains:
        # PPI without ag_chains hint: form interface pairs from all chains.
        ag_chains = sorted(used)
    sel_p = np.concatenate([chains[c]["plddt"] for c in used])
    out["cif_plddt_mean"] = float(sel_p.mean())
    out["cif_plddt_std"] = float(sel_p.std())
    out["cif_plddt_min"] = float(sel_p.min())
    for q in (10, 25, 50, 75, 90):
        out[f"cif_plddt_q{q}"] = float(np.quantile(sel_p, q / 100))
    out["cif_plddt_lt70_frac"] = float((sel_p < 70).mean())
    out["n_chain_breaks"] = int(nbreaks)
    out["n_residues_total"] = int(len(sel_p))

    # ---- per-chain pLDDT means ---------------------------------------------
    if ab_chains:
        out["ab_only_chain_plddt_cif"] = float(
            np.mean([chains[c]["plddt"].mean() for c in ab_chains])
        )
    if ag_chains:
        out["antigen_chain_plddt_cif_mean"] = float(
            np.mean([chains[c]["plddt"].mean() for c in ag_chains])
        )
    if cfg.scfv_chain and cfg.scfv_chain in chains:
        out["ab_scfv_chain_plddt_cif"] = float(chains[cfg.scfv_chain]["plddt"].mean())

    # ---- interface pairs ----------------------------------------------------
    pairs: list[dict] = []
    if cfg.mode == "ab":
        pair_iter = [(ab, ag) for ab in ab_chains for ag in ag_chains]
    else:
        # PPI: all unordered pairs among the chains we have
        chains_iter = ag_chains  # populated above
        pair_iter = [
            (chains_iter[i], chains_iter[j])
            for i in range(len(chains_iter))
            for j in range(i + 1, len(chains_iter))
        ]
    for a, b in pair_iter:
        s = iface_pair(chains[a], chains[b])
        s["pdockq"] = pdockq_b(s["iface_plddt_combined"], s["n_contacts"])
        s["anticonf"] = pdockq_b(s["iface_plddt_ab"], s["n_contacts"])
        pairs.append(s)

    if pairs:
        out["n_ab_ag_pairs"] = len(pairs)
        out["iface_n_residues_ab_total"] = int(sum(p["n_iface_ab"] for p in pairs))
        out["iface_n_residues_ag_total"] = int(sum(p["n_iface_ag"] for p in pairs))
        out["iface_n_contacts_total"] = int(sum(p["n_contacts"] for p in pairs))
        out["iface_plddt_ab_max"] = _agg(pairs, "iface_plddt_ab", max)
        out["iface_plddt_ag_max"] = _agg(pairs, "iface_plddt_ag", max)
        out["iface_plddt_combined_max"] = _agg(pairs, "iface_plddt_combined", max)
        out["iface_plddt_combined_mean"] = _agg(pairs, "iface_plddt_combined",
                                                lambda v: sum(v) / len(v))
        out["iface_plddt_ab_mean"] = _agg(pairs, "iface_plddt_ab",
                                          lambda v: sum(v) / len(v))
        out["iface_plddt_ag_mean"] = _agg(pairs, "iface_plddt_ag",
                                          lambda v: sum(v) / len(v))
        out["pdockq_max"] = _agg(pairs, "pdockq", max)
        out["pdockq_mean"] = _agg(pairs, "pdockq", lambda v: sum(v) / len(v))
        out["anticonf_max"] = _agg(pairs, "anticonf", max)
        out["anticonf_mean"] = _agg(pairs, "anticonf", lambda v: sum(v) / len(v))

    # ---- "primary" pair: first ab × first ag (matches H1335 schema) --------
    if ab_chains and ag_chains:
        s = iface_pair(chains[ab_chains[0]], chains[ag_chains[0]])
        out["iface_n_residues_ab"] = s["n_iface_ab"]
        out["iface_n_residues_ag"] = s["n_iface_ag"]
        out["iface_n_contacts"] = s["n_contacts"]
        out["iface_plddt_ab"] = s["iface_plddt_ab"]
        out["iface_plddt_ag"] = s["iface_plddt_ag"]
        out["iface_plddt_combined"] = s["iface_plddt_combined"]
        out["iface_pdockq"] = pdockq_b(s["iface_plddt_combined"], s["n_contacts"])
        out["iface_anticonf"] = pdockq_b(s["iface_plddt_ab"], s["n_contacts"])

    # ---- AF3 summary.json scalars ------------------------------------------
    out["json_iptm"] = summary.get("iptm", np.nan)
    out["json_ptm"] = summary.get("ptm", np.nan)
    out["json_ranking_score"] = summary.get("ranking_score", np.nan)
    out["json_mean_plddt"] = summary.get("plddt", np.nan)
    out["json_fraction_disordered"] = summary.get("fraction_disordered", np.nan)
    out["json_gpde"] = summary.get("gpde", np.nan)
    hc = summary.get("has_clash")
    out["json_has_clash"] = int(bool(hc)) if hc is not None else np.nan
    out["json_complex_iplddt"] = summary.get("complex_iplddt", np.nan)
    out["json_complex_ipde"] = summary.get("complex_ipde", np.nan)
    out["json_complex_pde"] = summary.get("complex_pde", np.nan)
    out["json_num_recycles"] = summary.get("num_recycles", np.nan)

    # ---- chain-pair iPTM aggregates ----------------------------------------
    cp = summary.get("chain_pair_iptm", {}) or {}
    if cp and ab_chains and ag_chains:
        vals: list[float] = []
        for ab in ab_chains:
            for ag in ag_chains:
                v = cp.get(ab, {}).get(ag) if isinstance(cp.get(ab, {}), dict) else None
                if v is not None:
                    vals.append(float(v))
        if vals:
            out["ab_ag_chain_pair_iptm_mean"] = float(np.mean(vals))
            out["ab_ag_chain_pair_iptm_max"] = float(np.max(vals))
            out["ab_ag_chain_pair_iptm_min"] = float(np.min(vals))
            out["ab_ag_chain_pair_iptm_n"] = int(len(vals))
            # Training (c3_per_iface_features.py:228) sets this per-iface = single cell.
            # Inference at candidate-level → take the best-cell (max) as the
            # "primary iface" proxy. Averaging dilutes the strong cell across
            # off-iface chain pairs and shifts xgb predictions significantly.
            out["iface_chain_pair_iptm"] = float(np.max(vals))

    ci = summary.get("chain_iptm", {}) or {}
    cpt = summary.get("chain_ptm", {}) or {}
    if ab_chains:
        ab_o = ab_chains[0]
        if ab_o in ci:
            out["ab_only_chain_iptm"] = ci[ab_o]
        if ab_o in cpt:
            out["ab_only_chain_ptm"] = cpt[ab_o]
    if ag_chains:
        ag_iptm = [ci.get(c) for c in ag_chains if ci.get(c) is not None]
        ag_ptm = [cpt.get(c) for c in ag_chains if cpt.get(c) is not None]
        if ag_iptm:
            out["antigen_chain_iptm_mean"] = float(np.mean(ag_iptm))
        if ag_ptm:
            out["antigen_chain_ptm_mean"] = float(np.mean(ag_ptm))

    # ---- heavy / light per-chain confidence --------------------------------
    heavy = cfg.heavy_chain
    light = cfg.light_chain
    if heavy and heavy in chains:
        out["heavy_chain_plddt_cif"] = float(chains[heavy]["plddt"].mean())
        if heavy in ci:
            out["heavy_chain_iptm"] = ci[heavy]
        if heavy in cpt:
            out["heavy_chain_ptm"] = cpt[heavy]
    if light and light in chains:
        out["light_chain_plddt_cif"] = float(chains[light]["plddt"].mean())
        if light in ci:
            out["light_chain_iptm"] = ci[light]
        if light in cpt:
            out["light_chain_ptm"] = cpt[light]
    if heavy and light and cp:
        v = cp.get(heavy, {}).get(light) or cp.get(light, {}).get(heavy)
        if v is not None:
            out["vh_vl_pair_iptm"] = float(v)

    # ---- naming-alias for plddt features (training column names) ------------
    # Training (b1_extract_jsons.py:139-156) read these from summary.json
    # `chain_plddt` which CASP17 pipeline doesn't emit; CIF B-factor is the
    # only source. Expose the CIF-derived values under the training column
    # names so xgb_v8 picks up the actual signal instead of train_means.
    if "heavy_chain_plddt_cif" in out:
        out["heavy_chain_plddt"] = out["heavy_chain_plddt_cif"]
    if "light_chain_plddt_cif" in out:
        out["light_chain_plddt"] = out["light_chain_plddt_cif"]
    if "ab_only_chain_plddt_cif" in out:
        out["ab_only_chain_plddt"] = out["ab_only_chain_plddt_cif"]
    if "antigen_chain_plddt_cif_mean" in out:
        out["antigen_chain_plddt_mean"] = out["antigen_chain_plddt_cif_mean"]

    # ---- misc ---------------------------------------------------------------
    out["n_chains_pred"] = int(len(used))
    out["n_iface"] = (out.get("iface_n_residues_ab", 0) or 0) + (
        out.get("iface_n_residues_ag", 0) or 0
    )
    out["n_model_contacts"] = out.get("iface_n_contacts_total", 0) or 0
    nc = summary.get("num_clashes", summary.get("n_clashes"))
    out["n_clashes"] = int(nc) if nc is not None else np.nan
    return out


# ----- multiprocessing driver ------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(chain_config: ChainConfig, sanitize_cutoff: float) -> None:
    warnings.filterwarnings("ignore")
    _WORKER_STATE["cfg"] = chain_config
    _WORKER_STATE["sanitize_cutoff"] = sanitize_cutoff


def _process_one(row: dict) -> dict:
    cfg: ChainConfig = _WORKER_STATE["cfg"]
    sanitize_cutoff: float = _WORKER_STATE["sanitize_cutoff"]
    out = {"model_id": row["model_id"]}
    try:
        feats = extract_base_features(
            row["src_cif"], row.get("summary_path", ""), cfg,
            sanitize_cutoff=sanitize_cutoff,
        )
        out.update(feats)
    except Exception as e:  # noqa: BLE001
        out["extract_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def run_base_features_parallel(
    index_df: pd.DataFrame,
    chain_config: ChainConfig,
    workers: int = 16,
    sanitize_cutoff: float = DEFAULT_SANITIZE_CUTOFF,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Run :func:`extract_base_features` in a worker pool over ``index_df``.

    ``index_df`` must contain at least ``model_id``, ``src_cif``,
    ``summary_path`` columns (i.e. what :func:`residudes_qa.interface.core.discover.discover_cifs`
    returns).  Returns a DataFrame keyed on ``model_id`` (not set as the
    pandas index — kept as a normal column for easy joins).
    """
    records = index_df[["model_id", "src_cif", "summary_path"]].to_dict("records")
    n = len(records)
    print(f"  base features: {n} cifs  workers={workers}", flush=True)
    t0 = time.time()
    rows: list[dict] = []
    n_err = 0
    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(chain_config, sanitize_cutoff),
    ) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_one, records, chunksize=10), 1
        ):
            rows.append(res)
            if "extract_err" in res:
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
        f"  base features: shape={df.shape}  errors={n_err}  "
        f"({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return df


def save_base_features(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write ``<output_dir>/stage_outputs/base_features.parquet``."""
    out_dir = Path(output_dir) / "stage_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "base_features.parquet"
    df.to_parquet(out, index=False)
    return out
