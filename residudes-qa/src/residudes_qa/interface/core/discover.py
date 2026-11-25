"""Index discovery — walk results/<sto>/<variant>/<method>/seed_*/sample_*/.

Builds the canonical ``index.parquet`` that downstream feature modules iterate
over.  One row per discovered ``model.cif``.  Generalized from H1335's
``19b_rebuild_fab_unit_index.py`` so any CASP target laid out as

    <target_dir>/results/<sto>/<variant>/<method>/seed_<S>/sample_<I>/model.cif

can be indexed with a single call.

The two id columns match the H1335 convention exactly:

* ``model_id``       = ``<method>_seed<S>_sample<I>__<variant>.pdb``
* ``candidate_uid``  = ``<target>__<method>__seed<S>_sample<I>__<variant>``

so downstream joins (xgb input table, picks, viz) stay drop-in compatible
with the H1335 artefacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

INDEX_COLUMNS: list[str] = [
    "target",
    "pdb",
    "sto",
    "variant",
    "method",
    "seed",
    "sample",
    "candidate_uid",
    "model_id",
    "src_cif",
    "summary_path",
]


def _iter_sample_dirs(
    results_root: Path,
    sto: str,
    variants: Optional[Iterable[str]],
    methods: Optional[Iterable[str]],
):
    """Yield ``(variant, method, seed, sample, sample_dir)`` tuples."""
    sto_root = results_root / sto
    if not sto_root.is_dir():
        return
    vfilter = set(variants) if variants else None
    mfilter = set(methods) if methods else None
    for vdir in sorted(p for p in sto_root.iterdir() if p.is_dir()):
        if vfilter is not None and vdir.name not in vfilter:
            continue
        for mdir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            if mfilter is not None and mdir.name not in mfilter:
                continue
            for seed_dir in sorted(mdir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                for sample_dir in sorted(seed_dir.glob("sample_*")):
                    if not sample_dir.is_dir():
                        continue
                    try:
                        sample = int(sample_dir.name.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    yield vdir.name, mdir.name, seed, sample, sample_dir


def discover_cifs(
    target_dir: str | Path,
    sto: str,
    variants: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    target: Optional[str] = None,
    pdb: Optional[str] = None,
) -> pd.DataFrame:
    """Walk ``<target_dir>/results/<sto>`` and index every ``model.cif``.

    Parameters
    ----------
    target_dir
        Root directory for the CASP target, e.g.
        ``/data/casp17/targets/H1335``.
    sto
        Stoichiometry tag (the second path level under ``results/``), e.g.
        ``"A1B1C1D1E1"`` for H1335.
    variants, methods
        Optional whitelists; ``None`` means all on disk.
    target, pdb
        Override the values used for the ``target`` / ``pdb`` columns and the
        ``candidate_uid`` prefix.  Default: ``target = target_dir.name``,
        ``pdb = target``.

    Returns
    -------
    pd.DataFrame
        Columns: see :data:`INDEX_COLUMNS`.  Rows are sorted by
        ``(variant, method, seed, sample)`` (via the iterator).  Rows whose
        ``model.cif`` is missing are silently skipped.
    """
    target_dir = Path(target_dir)
    results_root = target_dir / "results"
    tgt = target or target_dir.name
    pdb_id = pdb or tgt

    rows: list[dict] = []
    for variant, method, seed, sample, sample_dir in _iter_sample_dirs(
        results_root, sto, variants, methods
    ):
        cif = sample_dir / "model.cif"
        if not cif.exists():
            continue
        summ = sample_dir / "summary.json"
        model_id = f"{method}_seed{seed}_sample{sample}__{variant}.pdb"
        candidate_uid = f"{tgt}__{method}__seed{seed}_sample{sample}__{variant}"
        rows.append(
            {
                "target": tgt,
                "pdb": pdb_id,
                "sto": sto,
                "variant": variant,
                "method": method,
                "seed": seed,
                "sample": sample,
                "candidate_uid": candidate_uid,
                "model_id": model_id,
                "src_cif": str(cif),
                "summary_path": str(summ) if summ.exists() else "",
            }
        )

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return df


def save_index(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write the index to ``<output_dir>/index.parquet`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "index.parquet"
    df.to_parquet(out, index=False)
    return out


def discover_and_save(
    target_dir: str | Path,
    sto: str,
    output_dir: str | Path,
    variants: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    target: Optional[str] = None,
    pdb: Optional[str] = None,
) -> tuple[pd.DataFrame, Path]:
    """Convenience: :func:`discover_cifs` then :func:`save_index`.

    Prints a short summary (row count, per-variant / per-method counts) to
    stdout, matching the H1335 stage-19b style.
    """
    df = discover_cifs(target_dir, sto, variants, methods, target=target, pdb=pdb)
    out_path = save_index(df, output_dir)
    print(f"  rows: {len(df)}")
    if not df.empty:
        print(f"  per variant: {df.variant.value_counts().to_dict()}")
        print(f"  per method:  {df.method.value_counts().to_dict()}")
    print(f"  saved: {out_path}")
    return df, out_path
