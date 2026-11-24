"""ANARCI-based CDR loop annotation for Ab chains.

Given a CIF model + heavy/light chain identifiers, runs ANARCI on the chain
sequences and returns a residue → CDR-loop mapping that maps **the model's
actual residue numbers** (not IMGT) to ``H1/H2/H3/L1/L2/L3/framework``.

This handles the common pitfall where Ab models are renumbered to anchor
sequence (1..N) rather than IMGT. ANARCI gives the IMGT alignment so we
can label each model residue with its IMGT-defined CDR.

Requires the ANARCI binary on PATH (typically ``.venv-cdr/bin/ANARCI``).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

# IMGT default CDR ranges (inclusive)
IMGT_CDR = {
    "H1": (27, 38),
    "H2": (56, 65),
    "H3": (105, 117),
    "L1": (27, 38),
    "L2": (56, 65),
    "L3": (105, 117),
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_chain_sequence(
    cif_path: str | Path,
    chain_id: str,
) -> list[tuple[int, str]]:
    """Return list of (resnum, one_letter_AA) for given chain in model."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    out: list[tuple[int, str]] = []
    for model in st:
        for ch in model:
            if ch.name != chain_id:
                continue
            for r in ch:
                aa = AA_3TO1.get(r.name)
                if aa is None:
                    continue
                if r.find_atom("CA", "*") is None:
                    continue
                out.append((int(r.seqid.num), aa))
        break
    return out


def run_anarci_on_sequence(
    sequence: str,
    *,
    chain_type: str = "H",   # 'H' or 'L'  (heavy / light Ig)
    scheme: str = "imgt",
    anarci_bin: str = "ANARCI",
) -> Optional[pd.DataFrame]:
    """Run ANARCI on a single sequence and return per-residue IMGT numbering.

    Returns DataFrame with columns: ``imgt_num, imgt_ins, aa`` — one row per
    aligned residue (gaps excluded). Returns None on ANARCI failure.

    ANARCI spawns ``hmmscan`` internally and needs it on PATH. If ``anarci_bin``
    is given as an absolute path inside a venv (typical: ``.venv-cdr/bin/ANARCI``),
    that bin directory is prepended to PATH automatically so the bundled
    ``hmmscan`` is discoverable.
    """
    env = os.environ.copy()
    bin_dir = str(Path(anarci_bin).parent) if os.path.isabs(anarci_bin) else None
    if bin_dir and bin_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "in.fa"
        fa.write_text(f">query\n{sequence}\n")
        out = Path(td) / "out.anarci"
        cmd = [anarci_bin, "-i", str(fa), "-o", str(out),
               "--scheme", scheme, "-r", "ig"]
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           timeout=60, env=env)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

        # ANARCI output: hits in <out>.txt or <out>_H.csv depending on version
        # Newer versions write <out>_<chain>.csv with imgt numbering
        # Fall back: parse the multi-section text output
        candidates = list(Path(td).glob("out*"))
        rows = []
        for cand in candidates:
            txt = cand.read_text()
            # Parse blocks of form: H 27 . R
            for line in txt.splitlines():
                if not line or line.startswith("#") or line.startswith("HMM"):
                    continue
                parts = line.split()
                if len(parts) >= 3 and parts[0] in ("H", "L", "K"):
                    chain_letter, num = parts[0], parts[1]
                    ins = parts[2] if len(parts) >= 4 else ""
                    aa = parts[-1]
                    if aa == "-":
                        continue
                    try:
                        rows.append({
                            "imgt_num": int(num),
                            "imgt_ins": ins if ins != "." else "",
                            "aa": aa,
                        })
                    except ValueError:
                        continue
        if not rows:
            return None
        return pd.DataFrame(rows)


def cdr_loop_for_imgt_num(num: int, chain_type: str) -> str:
    """Return CDR loop label for an IMGT residue number, by chain type."""
    if chain_type.upper() == "H":
        for label in ("H1", "H2", "H3"):
            lo, hi = IMGT_CDR[label]
            if lo <= num <= hi:
                return label
        return "framework_H"
    else:
        for label in ("L1", "L2", "L3"):
            lo, hi = IMGT_CDR[label]
            if lo <= num <= hi:
                return label
        return "framework_L"


def build_residue_cdr_map(
    cif_path: str | Path,
    heavy_chain: Optional[str] = None,
    light_chain: Optional[str] = None,
    *,
    anarci_bin: str = "ANARCI",
) -> dict[tuple[str, int], str]:
    """Map ``(chain_id, model_resnum)`` → CDR-loop label for one model.

    For each Ab chain in the model:
      1. Extract sequence (gemmi, only residues with CA)
      2. Run ANARCI → IMGT numbering
      3. Align ANARCI output residue-by-residue back to model residues
      4. Label each model residue with its IMGT-defined CDR loop

    Non-Ab residues (other chains) are not in the returned dict.

    Notes
    -----
    ANARCI output sequence should match input sequence; we align positionally.
    If ANARCI fails (no Ab detected) we fall back to ``framework`` for all.
    """
    out: dict[tuple[str, int], str] = {}
    for chain_id, chain_type in [(heavy_chain, "H"), (light_chain, "L")]:
        if chain_id is None:
            continue
        seq_records = get_chain_sequence(cif_path, chain_id)
        if not seq_records:
            continue
        sequence = "".join(aa for _, aa in seq_records)
        anarci_df = run_anarci_on_sequence(
            sequence, chain_type=chain_type, anarci_bin=anarci_bin,
        )
        if anarci_df is None or len(anarci_df) == 0:
            # Fall back: mark all framework
            for resnum, _ in seq_records:
                out[(chain_id, resnum)] = f"framework_{chain_type}"
            continue
        # ANARCI aligns to sequence positionally (skips gaps in output)
        # Map ith anarci residue to ith seq_record (model residue)
        n = min(len(anarci_df), len(seq_records))
        for i in range(n):
            imgt_num = int(anarci_df.iloc[i]["imgt_num"])
            model_resnum = seq_records[i][0]
            out[(chain_id, model_resnum)] = cdr_loop_for_imgt_num(imgt_num, chain_type)
        # leftover residues (if seq longer than anarci output)
        for j in range(n, len(seq_records)):
            out[(chain_id, seq_records[j][0])] = f"framework_{chain_type}"
    return out


def annotate_pairs_with_cdr(
    pair_contacts: pd.DataFrame,
    cdr_map: dict[tuple[str, int], str],
) -> pd.DataFrame:
    """Add ``cdr_loop`` column to a pair_contacts DataFrame.

    Parameters
    ----------
    pair_contacts
        From :func:`pair_contacts.extract_pair_contacts` — must have
        ``paratope_chain, paratope_resnum`` columns.
    cdr_map
        From :func:`build_residue_cdr_map`.
    """
    def lookup(row):
        key = (row["paratope_chain"], int(row["paratope_resnum"]))
        return cdr_map.get(key, "framework")
    df = pair_contacts.copy()
    df["cdr_loop"] = df.apply(lookup, axis=1)
    return df


__all__ = [
    "IMGT_CDR",
    "get_chain_sequence",
    "run_anarci_on_sequence",
    "cdr_loop_for_imgt_num",
    "build_residue_cdr_map",
    "annotate_pairs_with_cdr",
]
