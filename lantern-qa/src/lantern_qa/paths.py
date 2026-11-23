"""Where lantern-qa finds its data, in one place.

The vendored EMA scorers already resolve every data file from an environment variable first
(``EMA_DFIRE_V2_TABLE``, ``EMA_BBDEP_LIB``, ...). ``configure_env()`` fills those variables from
the lantern-qa data root unless the caller has set them, so the scorer code stays as it was in
CASP17 and only this module knows the layout.

Data root, in order: ``$LANTERN_QA_DATA`` -> ``<lantern-qa>/data/files``.
Layout under the root (fetched and checked by ``lantern-qa data fetch`` / ``data verify``):

    xgb_models/                     XGBoost models + meta + normalization json
    ema/dfire/dfire_v2_energy_table_with_types.pt
    ema/dunbrack/ALL.bbdep.rotamers.lib, NDRD_TCBIG.txt.gz, ndrd_tcbig_table.pkl
    ema/potentials/aadars_v3_full.pt
    proteinmpnn/model_weights/*.pt
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent           # .../lantern-qa/src/lantern_qa
SRC_DIR = PACKAGE_DIR.parent                            # .../lantern-qa/src
PROJECT_DIR = SRC_DIR.parent                            # .../lantern-qa
MANIFEST = PROJECT_DIR / "data" / "manifest.toml"


def data_root() -> Path:
    env = os.environ.get("LANTERN_QA_DATA")
    return Path(env).expanduser().resolve() if env else PROJECT_DIR / "data" / "files"


def xgb_models_dir() -> Path:
    return Path(os.environ.get("LANTERN_QA_XGB_MODELS", data_root() / "xgb_models"))


def proteinmpnn_code_dir() -> Path:
    """ProteinMPNN code vendored in the package (protein_mpnn/, mpnn_utils.py, ...)."""
    return PACKAGE_DIR / "mpnn" / "run"


def proteinmpnn_weights_dir() -> Path:
    return data_root() / "proteinmpnn" / "model_weights"


# env var read by the vendored code  ->  path under the data root (or package)
ENV_DEFAULTS = {
    "EMA_DFIRE_V2_TABLE": lambda: data_root() / "ema" / "dfire" / "dfire_v2_energy_table_with_types.pt",
    "EMA_BBDEP_LIB": lambda: data_root() / "ema" / "dunbrack" / "ALL.bbdep.rotamers.lib",
    "EMA_NDRD_PATH": lambda: data_root() / "ema" / "dunbrack" / "NDRD_TCBIG.txt.gz",
    "EMA_AADARS_V3_POT": lambda: data_root() / "ema" / "potentials" / "aadars_v3_full.pt",
    "EMA_PROTEINMPNN_DIR": proteinmpnn_code_dir,
    "EMA_PROTEINMPNN_WEIGHTS": proteinmpnn_weights_dir,
    "LANTERN_QA_XGB_MODELS": xgb_models_dir,
}


def configure_env() -> dict[str, str]:
    """Set every variable in ENV_DEFAULTS that the caller has not set. Returns the effective values."""
    out = {}
    for key, fn in ENV_DEFAULTS.items():
        os.environ.setdefault(key, str(fn()))
        out[key] = os.environ[key]
    return out
