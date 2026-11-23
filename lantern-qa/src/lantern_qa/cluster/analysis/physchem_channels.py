"""Residue physicochemical channel classification for Ab-Ag contacts.

Each contact pair (CDR_aa, Ag_aa) is classified into 0+ interaction
"channels" based on residue type heuristics:

* ``hydrophobic``   — both residues from {A V L I M F W P G}
* ``salt_bridge``   — one is from {K R H} and the other from {D E}, or vice versa
* ``polar_hbond``   — at least one residue is in {S T N Q Y C}
* ``aromatic``      — both from {F Y W H} (π-stacking candidate)
* ``cation_pi``     — one from {K R} and the other from {F Y W} (or vice versa)

This is a coarse-grained classification; for full geometric H-bond /
salt-bridge detection use PLIP or Arpeggio. These channels work well as
a fingerprint *feature* per cluster without geometry-dependent caveats.

Output formats
--------------
For each contact row, we return a list of channels. Aggregated per cluster
we get the channel-frequency distribution.

References
----------
Burley & Petsko 1985 (aromatic π-stacking), Gallivan & Dougherty 1999 (cation-π)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# Standard 3-letter → 1-letter
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

HYDROPHOBIC = set("AVLIMFWPG")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQYC")
AROMATIC = set("FYWH")
CATION_PI_POS = set("KR")        # H excluded (weak)
CATION_PI_RING = set("FYW")

CHANNELS = ("hydrophobic", "salt_bridge", "polar_hbond", "aromatic", "cation_pi")


def to_one_letter(resname: str) -> str:
    """3-letter → 1-letter. Returns 'X' for unknown."""
    if not isinstance(resname, str):
        return "X"
    return AA_3TO1.get(resname.upper(), "X")


def classify_pair(aa_a: str, aa_b: str) -> list[str]:
    """Return list of channels for residue pair (order-insensitive).

    ``aa_a``, ``aa_b`` are 1-letter codes (use :func:`to_one_letter` first).
    Multiple channels can apply (e.g. Lys-Tyr could be both polar and cation-π).
    """
    channels: list[str] = []
    if aa_a in HYDROPHOBIC and aa_b in HYDROPHOBIC:
        channels.append("hydrophobic")
    if (aa_a in POSITIVE and aa_b in NEGATIVE) or (aa_a in NEGATIVE and aa_b in POSITIVE):
        channels.append("salt_bridge")
    if aa_a in POLAR or aa_b in POLAR:
        channels.append("polar_hbond")
    if aa_a in AROMATIC and aa_b in AROMATIC:
        channels.append("aromatic")
    if ((aa_a in CATION_PI_POS and aa_b in CATION_PI_RING)
        or (aa_b in CATION_PI_POS and aa_a in CATION_PI_RING)):
        channels.append("cation_pi")
    return channels


def annotate_contacts(
    contacts_pair: pd.DataFrame,
    *,
    paratope_aa_col: str = "ab_resname",
    epitope_aa_col: str = "abc_resname",
) -> pd.DataFrame:
    """Add channel boolean columns to a pair-level contacts DataFrame.

    Parameters
    ----------
    contacts_pair
        Must include paratope+epitope residue identities (use stage-13 style
        ``pair_contacts.csv`` or compute on-the-fly from cif).
    paratope_aa_col, epitope_aa_col
        Column names for residue names (3-letter format expected, e.g. ``ALA``).

    Returns
    -------
    pd.DataFrame
        Original + 5 boolean columns ``ch_hydrophobic``, ``ch_salt_bridge``, ...
    """
    df = contacts_pair.copy()
    aa1 = df[paratope_aa_col].map(to_one_letter)
    aa2 = df[epitope_aa_col].map(to_one_letter)
    channels_list = [classify_pair(a, b) for a, b in zip(aa1, aa2)]
    for ch in CHANNELS:
        df[f"ch_{ch}"] = [ch in cl for cl in channels_list]
    df["n_channels"] = df[[f"ch_{c}" for c in CHANNELS]].sum(axis=1)
    return df


def cluster_channel_distribution(
    contacts_pair_annotated: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
) -> pd.DataFrame:
    """Fraction of contacts in each channel, per cluster.

    Returns
    -------
    pd.DataFrame
        index = cluster_id, columns = 5 channels + 'total_contacts' + 'n_models'
    """
    merged = contacts_pair_annotated.merge(
        cluster_assignments[[model_col, cluster_col]].rename(columns={model_col: "model_id"}),
        on="model_id",
        how="inner",
    )
    rows = []
    for cid, grp in merged.groupby(cluster_col):
        row = {"cluster": int(cid),
               "n_models": grp["model_id"].nunique(),
               "n_contacts": len(grp)}
        for ch in CHANNELS:
            col = f"ch_{ch}"
            row[ch] = float(grp[col].sum() / max(len(grp), 1))   # fraction of contacts
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster").sort_index()


def epitope_residue_type_profile(
    contacts_long: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    model_col: str = "model_id",
    epitope_aa_col: str = "abc_resname",
) -> pd.DataFrame:
    """Fallback for **epitope-only** contacts: residue-type composition.

    When pair data is unavailable (only Ag side residue identity known),
    we can still report the cluster epitope residue-type distribution.
    """
    merged = contacts_long.merge(
        cluster_assignments[[model_col, cluster_col]].rename(columns={model_col: "model_id"}),
        on="model_id",
        how="inner",
    )
    rows = []
    for cid, grp in merged.groupby(cluster_col):
        aa1 = grp[epitope_aa_col].map(to_one_letter)
        n_total = len(aa1)
        rows.append({
            "cluster": int(cid),
            "n_models": grp["model_id"].nunique(),
            "n_contacts": n_total,
            "hydrophobic": float((aa1.isin(HYDROPHOBIC)).sum() / max(n_total, 1)),
            "positive_charge": float((aa1.isin(POSITIVE)).sum() / max(n_total, 1)),
            "negative_charge": float((aa1.isin(NEGATIVE)).sum() / max(n_total, 1)),
            "polar": float((aa1.isin(POLAR)).sum() / max(n_total, 1)),
            "aromatic": float((aa1.isin(AROMATIC)).sum() / max(n_total, 1)),
        })
    return pd.DataFrame(rows).set_index("cluster").sort_index()


__all__ = [
    "classify_pair",
    "annotate_contacts",
    "cluster_channel_distribution",
    "epitope_residue_type_profile",
    "to_one_letter",
    "CHANNELS",
]
