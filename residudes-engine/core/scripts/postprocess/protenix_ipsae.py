"""
Protenix → ipSAE adapter.

Reads Protenix's per-sample dump files (from runner/dumper.py) and computes
ipSAE scores without going through a file-format conversion round-trip.

Protenix dump layout (when --need_atom_confidence true):
  dump_dir/{dataset}/{pdb_id}/seed_{seed}/predictions/
    {pdb_id}_sample_{rank}.cif
    {pdb_id}_summary_confidence_sample_{rank}.json
    {pdb_id}_full_data_sample_{rank}.json     # contains token_pair_pae, atom_plddt

The `full_data_*.json` key-name → AF3-format mapping:
  token_pair_pae → pae
  atom_plddt     → atom_plddts
  (summary_confidence)['chain_pair_iptm'] → per-chain-pair iptm matrix

We parse Protenix's CIF via ipsae's upstream parser, build the pae_data dict
in memory, and call compute_ipsae directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable, Optional

import numpy as np

# Vendor ipsae
_HERE = os.path.dirname(os.path.abspath(__file__))
_IPSAE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "thirdparty", "ipsae"))
if _IPSAE_DIR not in sys.path:
    sys.path.insert(0, _IPSAE_DIR)

from ipsae import (  # noqa: E402
    parse_structure,
    compute_ipsae,
    _write_byres_tsv,
    _write_summary_txt_and_pml,
    _format_cutoff_string,
)


_SAMPLE_CIF_RE = re.compile(r"^(?P<stem>.+)_sample_(?P<rank>\d+)\.cif$")


def find_sample_triples(predictions_dir: str) -> list[dict]:
    """Scan a Protenix `predictions/` directory and return matching file triples.

    Returns list of {cif, full_data, summary, stem, rank}. Missing members are None.
    """
    triples = []
    if not os.path.isdir(predictions_dir):
        return triples
    for fname in sorted(os.listdir(predictions_dir)):
        m = _SAMPLE_CIF_RE.match(fname)
        if not m:
            continue
        stem = m.group("stem")
        rank = m.group("rank")
        cif = os.path.join(predictions_dir, fname)
        full_data = os.path.join(
            predictions_dir, f"{stem}_full_data_sample_{rank}.json"
        )
        summary = os.path.join(
            predictions_dir, f"{stem}_summary_confidence_sample_{rank}.json"
        )
        triples.append({
            "cif": cif,
            "full_data": full_data if os.path.exists(full_data) else None,
            "summary": summary if os.path.exists(summary) else None,
            "stem": stem,
            "rank": rank,
        })
    return triples


def _load_protenix_pae_data(
    full_data_path: str,
    summary_path: Optional[str],
    structure: dict,
) -> dict:
    """Build pae_data dict (same shape as ipsae.load_pae returns) from Protenix JSONs."""
    token_array = structure["token_mask"]
    CA_atom_num = structure["CA_atom_num"]
    CB_atom_num = structure["CB_atom_num"]
    unique_chains = structure["unique_chains"]
    numres = structure["numres"]

    with open(full_data_path, "r") as f:
        full = json.load(f)

    if "token_pair_pae" not in full:
        raise KeyError(
            f"'token_pair_pae' not found in {full_data_path}. "
            "Was Protenix run with --need_atom_confidence true?"
        )
    pae_full = np.array(full["token_pair_pae"], dtype=float)  # [N_token, N_token]

    # Slice to CA/C1 tokens only (same logic ipsae uses for AF3/Boltz).
    tmask = token_array.astype(bool)
    if pae_full.shape[0] == tmask.size:
        pae_matrix = pae_full[np.ix_(tmask, tmask)]
    elif pae_full.shape[0] == numres:
        # Already residue-level (no ligand tokens) — no slicing needed.
        pae_matrix = pae_full
    else:
        raise ValueError(
            f"token_pair_pae shape {pae_full.shape} mismatches structure "
            f"(N_token={tmask.size}, N_res={numres}). CIF/JSON ordering may differ."
        )

    # plddt
    if "atom_plddt" in full and CA_atom_num.size and CB_atom_num.size:
        atom_plddts = np.array(full["atom_plddt"], dtype=float)
        # Protenix stores 0–1; ipsae expects 0–100 for pDockQ
        if atom_plddts.size and atom_plddts.max() <= 1.0:
            atom_plddts = atom_plddts * 100.0
        max_atom_num = int(max(CA_atom_num.max(), CB_atom_num.max()))
        if atom_plddts.size > max_atom_num:
            plddt = atom_plddts[CA_atom_num]
            cb_plddt = atom_plddts[CB_atom_num]
        else:
            plddt = np.zeros(numres)
            cb_plddt = np.zeros(numres)
    else:
        # Either atom_plddt absent, or CIF parsed with zero CA/CB atoms
        # (malformed / empty input) — fall back to zeros rather than crashing
        plddt = np.zeros(numres)
        cb_plddt = np.zeros(numres)

    # Per-chain-pair iptm from summary (same shape as AF3 iptm_af3 dict)
    iptm_af = {c1: {c2: 0.0 for c2 in unique_chains if c1 != c2} for c1 in unique_chains}
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if "chain_pair_iptm" in summary:
            cpi = np.array(summary["chain_pair_iptm"], dtype=float)
            n_chain = len(unique_chains)
            if cpi.shape == (n_chain, n_chain):
                for n1, c1 in enumerate(unique_chains):
                    for n2, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        iptm_af[c1][c2] = float(cpi[n1, n2])
            # else: chain count mismatch — leave zeros

    return {
        "pae_matrix": pae_matrix,
        "plddt": plddt,
        "cb_plddt": cb_plddt,
        "iptm_af": iptm_af,
        "iptm_af2_scalar": -1.0,
        "ptm_af2_scalar": -1.0,
        "fmt": "af3",  # use af3 branch in writers (dict-form iptm_af)
    }


def run_ipsae_on_sample(
    cif_path: str,
    full_data_path: str,
    summary_path: Optional[str] = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    write_files: bool = True,
    output_dir: Optional[str] = None,
) -> dict:
    """Compute ipSAE for one Protenix sample. Returns scores dict."""
    structure = parse_structure(cif_path)
    pae_data = _load_protenix_pae_data(full_data_path, summary_path, structure)
    scores = compute_ipsae(structure, pae_data, pae_cutoff, dist_cutoff)

    output_paths = {}
    if write_files:
        base = cif_path[:-4]  # strip .cif
        pae_s = _format_cutoff_string(pae_cutoff)
        dist_s = _format_cutoff_string(dist_cutoff)
        stem = f"{base}_{pae_s}_{dist_s}"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.join(output_dir, os.path.basename(stem))
        txt_path = f"{stem}.txt"
        byres_path = f"{stem}_byres.txt"
        pml_path = f"{stem}.pml"
        _write_byres_tsv(byres_path, structure, scores, pae_data)
        _write_summary_txt_and_pml(
            txt_path, pml_path, structure, scores, pae_data,
            pae_cutoff, dist_cutoff, os.path.basename(base),
        )
        output_paths = {"txt": txt_path, "byres": byres_path, "pml": pml_path}

    return {
        "structure": structure,
        "pae_data": pae_data,
        "scores": scores,
        "output_paths": output_paths,
    }


def run_ipsae_on_dump(
    dump_dir: str,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    write_files: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Walk Protenix dump_dir and run ipSAE on every sample found.

    Expected layout: dump_dir/{dataset}/{pdb_id}/seed_*/predictions/*.cif
    """
    results = []
    for root, _, files in os.walk(dump_dir):
        if os.path.basename(root) != "predictions":
            continue
        if not any(f.endswith(".cif") for f in files):
            continue
        for triple in find_sample_triples(root):
            if triple["full_data"] is None:
                if verbose:
                    print(
                        f"[ipsae] skip {triple['cif']}: no full_data JSON "
                        "(set --need_atom_confidence true to enable)"
                    )
                continue
            try:
                res = run_ipsae_on_sample(
                    cif_path=triple["cif"],
                    full_data_path=triple["full_data"],
                    summary_path=triple["summary"],
                    pae_cutoff=pae_cutoff,
                    dist_cutoff=dist_cutoff,
                    write_files=write_files,
                )
                results.append({"triple": triple, "result": res})
                if verbose:
                    out = res["output_paths"].get("txt", "<in-memory>")
                    print(f"[ipsae] {os.path.basename(triple['cif'])} → {out}")
            except Exception as e:
                if verbose:
                    print(f"[ipsae] FAIL {triple['cif']}: {type(e).__name__}: {e}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ipSAE on a Protenix dump_dir.")
    p.add_argument("dump_dir", help="Path to Protenix --dump_dir root")
    p.add_argument("--pae-cutoff", type=float, default=10.0)
    p.add_argument("--dist-cutoff", type=float, default=15.0)
    p.add_argument("--no-write-files", action="store_true")
    args = p.parse_args()
    run_ipsae_on_dump(
        dump_dir=args.dump_dir,
        pae_cutoff=args.pae_cutoff,
        dist_cutoff=args.dist_cutoff,
        write_files=not args.no_write_files,
    )
