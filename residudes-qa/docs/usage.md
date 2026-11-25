# Installing, the data files, and running it

Alternatively, the published images may be used instead of a local installation: `residudes-qa`
for the CPU stack and `residudes-qa-cu130` for the GPU scorers
([`containers/README.md`](../../containers/README.md)). The data files are in either case fetched
to a host directory and bind-mounted, since they are versioned data rather than part of the
runtime.

## Install (uv)

```sh
cd RESIDUDES/residudes-qa
uv sync --extra cu130          # the exact CASP17 build: torch 2.11.0 from PyPI with its CUDA 13.0 libraries (driver >= 580: RTX 4090 / RTX 6000 Ada, B200)
uv sync --extra cu124          # torch 2.6.0+cu124 for nodes whose driver stops at CUDA 12.4
uv run residudes-qa data fetch   # 60 files, 281 MB, into data/files, each checked against its sha256 (--link to symlink)
uv run residudes-qa doctor       # python, pinned versions, torch/GPU, data, optional tools
```

Python is pinned to 3.12.12 (`.python-version`); uv downloads it if needed.

Dependencies match the CASP17 production venv (`analysis/.venv`), listed in full in
`envs/casp17-analysis-venv.freeze.txt` and locked in `uv.lock`. There is one exception:
- That venv carried networkx 2.2, which conflicts with its own torch 2.11 (torch needs ≥ 2.5.1).
- networkx is only used by the Louvain fallback, which runs only when leidenalg is missing.
- So it is 3.4.2 here, with no effect on default results.

`bin/residudes-qa` runs the CLI from `.venv` without installing.

## Data

`data/manifest.toml` lists every data file with its path, size, sha256, licence and
sources. The data root is `$RESIDUDES_QA_DATA`, otherwise `data/files` (git-ignored).

| Group | Files | Origin | Note |
|---|---|---|---|
| `xgb_models/` | 46 (25 MB) | trained by the team | defaults: PPI `xgb_ppi_v6_*`; antibody `xgb_v9_full` (DockQ) and `xgb_agab_v10_1_*` (ICS/IPS); older versions kept as fallbacks |
| `ema/dfire/` | 1 | team re-implementation of DFIRE2 | |
| `ema/dunbrack/` | 3 | Dunbrack bbdep2010 and NDRD | **academic licence: not redistributed**; fetched from the team copy only |
| `ema/potentials/` | 1 | team-trained aADARS v3 | |
| `proteinmpnn/model_weights/` | 12 (71 MB) | ProteinMPNN (MIT) | 11 files match the upstream repository byte for byte, so `--source url` works off-cluster |

`residudes-qa data list|fetch|verify`:
- Fetches from the team copy (`local`) first, then from the public URL.
- Refuses any file whose sha256 differs from the manifest.

Importing the package fills the environment variables the original scorers read,
pointing them at the data root. Values set beforehand win. The variables are:
- `EMA_DFIRE_V2_TABLE`, `EMA_BBDEP_LIB`, `EMA_NDRD_PATH`, `EMA_AADARS_V3_POT`
- `EMA_PROTEINMPNN_DIR`, `EMA_PROTEINMPNN_WEIGHTS`
- `RESIDUDES_QA_XGB_MODELS`

## Use

Arguments pass through unchanged. Only the command in front changes:

| CASP17 | residudes-qa |
|---|---|
| `xgbdimer <sub> …` / `XGBDimerScorer_tool/run_scoring.py <sub> …` | `residudes-qa interface <sub> …` |
| `XGBDimerScorer_tool/run_scoring.sh <sub> …` (Slurm fan-out, sharding, self-heal) | `scripts/interface/run_scoring.sh <sub> …` |
| `XGBDimerScorer_tool/shard_ppi.sh …` | `scripts/interface/shard_ppi.sh …` |
| `MassiveClustering_tool/run_clustering.py <target> …` | `residudes-qa cluster <target> …` |
| `MassiveClustering_tool/run_clustering.sh <target> …` (Slurm) | `scripts/cluster/run_clustering.sh <target> …` |
| `massivecluster <sub> …` | `residudes-qa massivecluster <sub> …` |
| `visualization/analyze_target.sh <target>` | `residudes-qa visualize <target>` (or `scripts/visualize/analyze_target.sh`) |
| `visualization/analyze_msa_target.sh <target>` | `residudes-qa visualize-msa <target>` |

```sh
# interface scores (DockQ + ICS + IPS) for one target
residudes-qa interface run-ppi-v4 --target-dir targets/T1417 --sto A3 --targets dockq,ics,ips -o out/T1417_iface
# the same on Slurm (GPU, automatic sharding)
scripts/interface/run_scoring.sh run-all --target-dir targets/T1417 --sto A3 --mode ppi -o out/T1417_iface
# cluster the pool (GPU TM matrix, adaptive K, medoids)
residudes-qa cluster targets/T1417 --sto A3 --method tm --gpu 0
# confidence plots and ranking CSVs into targets/T1417/visualization/
residudes-qa visualize targets/T1417 --no-msa
```

Output file and column names are kept from CASP17, because the clustering join, foldmine
and the ground-truth analyses read them. They are `xgb_pred`, `xgb_ics`, `xgb_ips`,
`master_with_xgb.parquet` and `xgb_scores.parquet`, and model files such as
`xgb_ppi_v6_full.json`.

Tool manuals: `docs/interface/`, `docs/cluster/`, `docs/ema/`, `docs/visualize/`.

## Optional parts

- **Antibody CDR masking and aADARS** need ANARCI with HMMER.
  - CASP17 used a separate venv: `analysis/.venv-cdr`, Python 3.10, anarci 2026.2.13.2.
  - Point `RESIDUDES_QA_CDR_BIN` at its `bin/`.
- **TCR docking angles** (`--method tcr-angle`) need the external Pierce binary (built against GSL) and `tcr3d_pierce.py`.
  - Set `RESIDUDES_QA_TCR_TOOLS` and `RESIDUDES_QA_TCR_RENUMBER`. They default to the team paths.
