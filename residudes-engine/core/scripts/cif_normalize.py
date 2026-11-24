#!/usr/bin/env python3
"""Normalize a model.cif via gemmi so all four runners (af3 / pt2 / of3 / bt2)
emit byte-equivalent output:

  1. Strip provenance / legal metadata (`_audit_*`, `_citation*`,
     `_pdbx_data_usage`).
  2. Rewrite the `_atom_site` loop in AF3-canonical 18-column order so
     B_iso_or_equiv = pLDDT lands where MolStar / PyMOL / ChimeraX expect it.
  3. Add ModelArchive QA metadata (`_ma_qa_metric*`, `_ma_data`,
     `_ma_model_list`) tying B_iso to pLDDT — MolStar then auto-detects and
     colors by pLDDT.

Idempotent. Safe no-op if gemmi is missing or the file has no atom_site loop.
"""
from __future__ import annotations

from pathlib import Path


_AF3_ATOM_SITE_COLS = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
    "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x", "Cartn_y", "Cartn_z",
    "occupancy", "B_iso_or_equiv",
    "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
)

_DROP_CIF_CATEGORIES = (
    "_audit_author",
    "_audit_conform",
    "_citation",
    "_citation_author",
    "_pdbx_data_usage",
    "_struct",
    "_struct_keywords",
)


def normalize_model_cif(cif_path) -> bool:
    """Normalize the model.cif at `cif_path` in place. Returns True on success."""
    cif_path = Path(cif_path)
    try:
        import gemmi
    except Exception:
        return False
    try:
        doc = gemmi.cif.read_file(str(cif_path))
    except Exception:
        return False
    if not len(doc):
        return False
    b = doc[0]

    for cat in _DROP_CIF_CATEGORIES:
        if b.get_mmcif_category(cat + "."):
            b.set_mmcif_category(cat + ".", {}, raw=True)

    site = b.get_mmcif_category("_atom_site.")
    if not site:
        return False
    n = len(next(iter(site.values())))
    canon = {}
    for col in _AF3_ATOM_SITE_COLS:
        if col in site:
            canon[col] = site[col]
        elif col == "auth_seq_id" and "label_seq_id" in site:
            canon[col] = site["label_seq_id"]
        elif col == "auth_asym_id" and "label_asym_id" in site:
            canon[col] = site["label_asym_id"]
        elif col == "id":
            canon[col] = [str(i + 1) for i in range(n)]
        elif col == "occupancy":
            canon[col] = ["1.00"] * n
        elif col in ("label_alt_id", "pdbx_PDB_ins_code"):
            canon[col] = ["?"] * n
        elif col in ("pdbx_PDB_model_num", "label_entity_id"):
            canon[col] = ["1"] * n
        elif col == "group_PDB":
            canon[col] = ["ATOM"] * n
        else:
            canon[col] = ["?"] * n
    b.set_mmcif_category("_atom_site.", canon, raw=False)

    if not b.get_mmcif_category("_ma_qa_metric."):
        groups: dict = {}
        order: list = []
        b_iso = canon["B_iso_or_equiv"]
        for i in range(n):
            try:
                v = float(b_iso[i])
            except (TypeError, ValueError):
                continue
            key = (canon["label_asym_id"][i], canon["label_seq_id"][i],
                   canon["label_comp_id"][i])
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(v)
        if order:
            b.set_mmcif_category("_ma_qa_metric.", {
                "id": ["1", "2"],
                "mode": ["global", "local"],
                "name": ["pLDDT", "pLDDT"],
                "software_group_id": ["1", "1"],
                "type": ["pLDDT", "pLDDT"],
            }, raw=False)
            global_pl = sum(sum(v) / len(v) for v in groups.values()) / len(order)
            b.set_mmcif_category("_ma_qa_metric_global.", {
                "metric_id": ["1"],
                "metric_value": [f"{global_pl:.2f}"],
                "model_id": ["1"],
                "ordinal_id": ["1"],
            }, raw=False)
            b.set_mmcif_category("_ma_qa_metric_local.", {
                "ordinal_id":    [str(o + 1) for o in range(len(order))],
                "model_id":      ["1"] * len(order),
                "label_asym_id": [k[0] for k in order],
                "label_seq_id":  [k[1] for k in order],
                "label_comp_id": [k[2] for k in order],
                "metric_id":     ["2"] * len(order),
                "metric_value":  [f"{sum(groups[k]) / len(groups[k]):.2f}" for k in order],
            }, raw=False)
            if not b.get_mmcif_category("_ma_data."):
                b.set_mmcif_category("_ma_data.", {
                    "content_type": ["model coordinates"],
                    "id": ["1"],
                    "name": ["Model"],
                }, raw=False)
            if not b.get_mmcif_category("_ma_model_list."):
                b.set_mmcif_category("_ma_model_list.", {
                    "data_id": ["1"],
                    "model_group_id": ["1"],
                    "model_group_name": ["Predicted models"],
                    "model_id": ["1"],
                    "model_name": ["Model"],
                    "model_type": ["Ab initio model"],
                    "ordinal_id": ["1"],
                }, raw=False)

    doc.write_file(str(cif_path))
    return True


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        ok = normalize_model_cif(p)
        print(f"{'ok ' if ok else 'FAIL'} {p}")
