#!/usr/bin/env python
"""Merge per-shard PPI outputs into the final combined table.

Each `run-all --mode ppi --shard k --nshards N` job writes its slice's combined
table to ``<out>/shards/shard{k}of{N}/xgb_dockq.csv`` (disjoint model rows). This
concatenates them into ``<out>/xgb_dockq.csv``, re-sorted by the mean score.

Usage:  ppi_merge.py <out_dir>
"""
import sys
import glob
from pathlib import Path

import pandas as pd


def main() -> None:
    out = Path(sys.argv[1])
    fs = sorted(glob.glob(str(out / "shards" / "shard*" / "xgb_dockq.csv")))
    if not fs:
        sys.exit(f"[ppimerge] no shard xgb_dockq.csv under {out}/shards/")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "model_id" in df.columns:
        df = df.drop_duplicates(subset="model_id", keep="first")
    sort_col = "xgb_mean" if "xgb_mean" in df.columns else (
        next((c for c in df.columns if c.endswith("_mean")), None))
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    df.to_csv(out / "xgb_dockq.csv", index=False)
    print(f"[ppimerge] merged {len(df)} rows from {len(fs)} shards "
          f"-> {out / 'xgb_dockq.csv'}", flush=True)


if __name__ == "__main__":
    main()
