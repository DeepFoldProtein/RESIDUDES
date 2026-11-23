"""lantern-qa: model quality assessment for structure-prediction pools.

Packages the CASP17 QA stack of the DeepFold team into one reproducible unit:
  lantern_qa.interface interface scorer: chain-pair features + XGBoost DockQ/ICS/IPS (ex-XGBDimerScorer)
  lantern_qa.cluster  MassiveClustering (GPU TM/iTM/interface clustering, medoids, networks)
  lantern_qa.ema      EMA scorers the features need (DFIRE v2, aADARS v3, Dunbrack bbdep/NDRD,
                      H-bond network, ProteinMPNN likelihood)
  lantern_qa.mpnn     vendored ProteinMPNN code (weights live in the data root)

Importing the package points the scorers at the data root (see lantern_qa.paths).
"""
from lantern_qa.paths import configure_env as _configure_env

__version__ = "0.1.0"
_configure_env()
