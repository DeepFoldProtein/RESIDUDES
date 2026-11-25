"""Per-family runners. Each exposes ``plan(req) -> Plan`` and ``post(req, plan)``."""
from __future__ import annotations

from . import alphafold3, boltz, esmfold2, openfold3, protenix

FAMILIES = {
    "protenix": protenix,
    "openfold3": openfold3,
    "boltz": boltz,
    "alphafold3": alphafold3,
    "esmfold2": esmfold2,
}


def for_family(family: str):
    try:
        return FAMILIES[family]
    except KeyError:
        raise SystemExit(f"no runner for family {family!r}") from None
