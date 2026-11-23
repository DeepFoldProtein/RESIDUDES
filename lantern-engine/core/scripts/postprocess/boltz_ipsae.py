"""
Boltz → ipSAE adapter.

Boltz natively writes `pae_*.npz` + `plddt_*.npz` + `confidence_*.json` +
`*.cif`/`*.pdb`. The upstream ipsae.py Boltz branch reads this layout
directly — so we just walk a Boltz output dir and call run_ipsae for each
matched triple.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

# Vendor ipsae
_HERE = os.path.dirname(os.path.abspath(__file__))
_IPSAE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "thirdparty", "ipsae"))
if _IPSAE_DIR not in sys.path:
    sys.path.insert(0, _IPSAE_DIR)

from ipsae import run_ipsae  # noqa: E402


# Boltz filename: pae_{stem}.npz pairs with {stem}.cif (or .pdb)
_PAE_RE = re.compile(r"^pae_(?P<stem>.+)\.npz$")


def find_boltz_pairs(boltz_dir: str) -> list[dict]:
    """Find (pae_npz, structure) pairs in a Boltz output directory.

    Boltz writes everything for one prediction side-by-side, e.g.:
        AURKA_TPX2_model_0.cif
        pae_AURKA_TPX2_model_0.npz
        plddt_AURKA_TPX2_model_0.npz
        confidence_AURKA_TPX2_model_0.json
    """
    pairs = []
    if not os.path.isdir(boltz_dir):
        return pairs
    for fname in sorted(os.listdir(boltz_dir)):
        m = _PAE_RE.match(fname)
        if not m:
            continue
        stem = m.group("stem")
        pae_path = os.path.join(boltz_dir, fname)
        cif = os.path.join(boltz_dir, f"{stem}.cif")
        pdb = os.path.join(boltz_dir, f"{stem}.pdb")
        if os.path.exists(cif):
            structure = cif
        elif os.path.exists(pdb):
            structure = pdb
        else:
            structure = None
        pairs.append({
            "pae": pae_path,
            "structure": structure,
            "stem": stem,
        })
    return pairs


def run_ipsae_on_boltz_dir(
    boltz_dir: str,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    write_files: bool = True,
    recursive: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Scan a Boltz output tree and run ipSAE on every pae/structure pair found."""
    results = []
    dirs = []
    if recursive:
        for root, _, files in os.walk(boltz_dir):
            if any(f.startswith("pae_") and f.endswith(".npz") for f in files):
                dirs.append(root)
    else:
        dirs = [boltz_dir]

    for d in dirs:
        for pair in find_boltz_pairs(d):
            if pair["structure"] is None:
                if verbose:
                    print(f"[ipsae] skip {pair['pae']}: no matching .cif/.pdb")
                continue
            try:
                res = run_ipsae(
                    structure_path=pair["structure"],
                    pae_path=pair["pae"],
                    pae_cutoff=pae_cutoff,
                    dist_cutoff=dist_cutoff,
                    write_files=write_files,
                    fmt="boltz",
                )
                results.append({"pair": pair, "result": res})
                if verbose:
                    out = res["output_paths"].get("txt", "<in-memory>")
                    print(f"[ipsae] {os.path.basename(pair['structure'])} → {out}")
            except Exception as e:
                if verbose:
                    print(
                        f"[ipsae] FAIL {pair['structure']}: "
                        f"{type(e).__name__}: {e}"
                    )
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ipSAE on a Boltz output dir.")
    p.add_argument("boltz_dir", help="Path to Boltz output root")
    p.add_argument("--pae-cutoff", type=float, default=10.0)
    p.add_argument("--dist-cutoff", type=float, default=15.0)
    p.add_argument("--no-write-files", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    args = p.parse_args()
    run_ipsae_on_boltz_dir(
        boltz_dir=args.boltz_dir,
        pae_cutoff=args.pae_cutoff,
        dist_cutoff=args.dist_cutoff,
        write_files=not args.no_write_files,
        recursive=not args.no_recursive,
    )
