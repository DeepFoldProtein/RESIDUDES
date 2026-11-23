# XGBDimerScorer

End-to-end **feature extraction + xgb scoring** for CASP17 dimer/multimer
targets. Output: one `master.parquet` per target with all features +
`xgb_v8_pred` (AbAg) or `xgb_pred` (PPI, predicts DockQ).

**Filtering / clustering / picking is delegated to `MassiveClustering_tool`** —
XGBDimer's job is to produce the feature table; MassiveClustering picks
submission candidates.

| Mode | Use case | xgb model | entry point |
|---|---|---|---|
| **`ab`** | Antibody-antigen complexes (e.g. H1335 Fab + gHgLgO) | `xgb_v9_full` | `run-all --mode ab` |
| **`ppi`** | Generic protein-protein dimers / multimer interfaces | `xgb_ppi_v6_full` | `run-all --mode ppi` (= `run-ppi-v4`) |

> **One entry point, two engines.** `run_scoring.sh run-all --mode {ab,ppi}` is
> the single command for both. `--mode ppi` auto-dispatches to the ppi_v4
> interface engine (interface auto-detect → proper `master_ppi_v4` features →
> `xgb_ppi` v6); **no ab/ag chains needed**. `run-ppi-v4` is kept as an alias.
> Both modes get **GPU + parallel fan-out + auto-sharding + self-heal** for free
> (see [§ Parallel execution](#parallel-execution-gpu--sharding--self-heal)).
> Pick the engine/model with `--ab-model v9` / `--ppi-model v6` (auto-fallback if
> a version is absent). `--help` groups flags by mode (`ab-mode` / `ppi-mode`).

> **OpenFold3 (`of3`) added + labels re-measured against FoldBench official** (2026-06-09).
> `xgb_ppi_v6_full` = v5 + OpenFold3 (**7 methods**: af3/b1/b2/ptx/ptx_v2/ef2/of3), trained on
> **FoldBench official DockQ** (OST 2.11.1, frozen GT — verified MAE 0.0018 vs reference eval).
> LOPO Pearson dockq 0.793 (vs pdockq 0.53). **ICS/IPS** also predicted:
> `xgb_ppi_v6_ics_full` / `xgb_ppi_v6_ips_full` (LOPO Pearson 0.79 / 0.78) — opt in with
> **`--targets dockq,ics,ips`** on `run-ppi-v4`/`score` (default `dockq` only; see [§ ICS/IPS](#icsips-scoring---targets)).
> Auto-falls back to v5 if the new file is absent; old models kept for rollback.
> ESMFold2 (`ef2`, single-seq) and earlier methods unchanged. AgAb = `xgb_v9_full` (12 methods).
> Full audit/reproduction: `/store/gunwoong/FoldBench/casp17_eval/` (FoldBench-faithful, shareable).

> **AgAb `xgb_agab_v10_1_*_full` added alongside (2026-06-10)** — deepfold3 7-method
> (af3/b1/b2/v1/v2/ef2/of3, Protenix=v1/v2), **FoldBench official labels**, xgb_ab.py-compatible
> (feat_cols_breakdown + method-z; verified scores through the tool). Adds **ICS/IPS** prediction
> (`xgb_agab_v10_1_{ics,ips}_full`) — not in v9; opt in with **`--targets dockq,ics,ips`** on
> `score --mode ab` (ics/ips come from v10.1, dockq from v9). **Default stays `xgb_v9_full`** for DockQ:
> v9 LOPO 0.82 (12 methods, ~3× data) > v10.1 0.766 (clean single-config). Use v10.1 for
> FoldBench-consistent labels + ICS/IPS. (abepitope dropped — verified importance 0.003; PAE
> deferred — partial coverage, importance 0.045.)

> **PPI scoring uses the ppi_v4 interface engine** (`run-all --mode ppi`, or its
> alias `run-ppi-v4`). `xgb_ppi_v6_full` was trained on the benchmark feature
> pipeline (`p2…p5` → `master_ppi_v4`, 224 features); the legacy
> `base/cpu/dfire/mpnn` groups reproduce only ~32/224, so feeding them to v6
> collapses scores into a narrow band. The interface engine reproduces the
> **full `master_ppi_v4` schema** (97.8 % of model gain) and scores per
> interface pair. **`run-all --mode ppi` now dispatches to this engine
> automatically** — the old in-tool `score --mode ppi` (mostly-imputed) path is
> no longer used by run-all. See [§ PPI v4 pipeline](#ppi-v4-pipeline-run-ppi-v4).

---

## How well it predicts, and how that was measured

The scorer predicts a structure-comparison metric from the model alone, so the question is how close its
prediction lands to the metric computed against the released structure. Reported performance, as Pearson *r*
between predicted and measured, under leave-one-protein-out cross-validation (LOPO: every model of a held-out
target is kept out of training):

| model | target metric | methods in training | LOPO Pearson *r* |
|---|---|---|---|
| `xgb_ppi_v6_full` | DockQ | 7 | 0.793 |
| `xgb_ppi_v6_ics_full` / `..._ips_full` | ICS / IPS | 7 | 0.79 / 0.78 |
| `xgb_v9_full` (antibody–antigen, default) | DockQ | 12 | 0.82 |
| `xgb_agab_v10_1_*_full` | DockQ, ICS, IPS | 7 | 0.766 |
| pDockQ, as a baseline | DockQ | – | 0.53 |

Labels are FoldBench official DockQ recomputed with OpenStructure 2.11.1 (MAE 0.0018 against the reference
evaluation), and the audit lives at `/store/gunwoong/FoldBench/casp17_eval/`.

**What those numbers do not establish.** The training code is not in this repository, so the split cannot be
checked here, and "LOPO" is not expanded anywhere else in it; the figures above are as reported by whoever trained
the models. The correlation is over a benchmark of models from the listed methods, which is not the same
distribution as one target's pool of thousands of samples from one or two predictors — and that is where it is
used. CASP17 gave a direct counter-example: on the antibody–antigen target H2442 the antibody-mode score ranked
the pool at Spearman ρ = −0.28, i.e. backwards, which is why lantern leaves these scores out of the slate unless
`LANTERN_USE_AB_INTERFACE=1` is set. Treat the scorer as a ranking aid within a pool whose behaviour you have
checked on that target, not as a measurement.

## Quick start (5 commands)

```bash
# 0. From any directory — the tool resolves its own paths
TOOL=/gpfs/deepfold/casp/casp17/analysis/XGBDimerScorer_tool

# 1. Discover all model.cif under a target (5 sec for 20,000 models)
$TOOL/run_scoring.sh discover \
    --target-dir /gpfs/deepfold/casp/casp17/targets/H1335 \
    --sto A1B1C1D1E1 \
    -o /path/to/out

# 2. Extract features (Slurm; ~1 hour for 20,000 cifs)
$TOOL/run_scoring.sh features --all \
    --index /path/to/out/index.parquet \
    --ab-chains DE --ag-chains ABC --heavy E --light D \
    -o /path/to/out

# 3. Join into master parquet
$TOOL/run_scoring.sh master \
    --index /path/to/out/index.parquet \
    --stage-dir /path/to/out/stage_outputs \
    -o /path/to/out

# 4. xgb score (5 sec)
$TOOL/run_scoring.sh score --master /path/to/out/master.parquet --mode ab -o /path/to/out

# 5. (Optional) One-shot wrapper that chains all 4 steps
$TOOL/run_scoring.sh run-all \
    --target-dir /gpfs/deepfold/casp/casp17/targets/H1335 \
    --sto A1B1C1D1E1 --mode ab \
    --ab-chains DE --ag-chains ABC --heavy E --light D \
    -o /path/to/out
```

Output (after step 4 or 5):

```
/path/to/out/
├── index.parquet                # 19,356 × 11 (target/variant/method/seed/sample/...)
├── stage_outputs/               # per-stage feature parquets
├── master.parquet               # 19,356 × ~130 (all features joined)
├── master_with_xgb.parquet      # 19,356 × ~131 (master + xgb_pred)
└── xgb_scores.parquet           # 19,356 × ~17 (xgb_pred + key meta)
```

Now hand `master_with_xgb.parquet` to MassiveClustering for filtering/clustering.

---

## Parallel execution (GPU + sharding + self-heal)

`run_scoring.sh run-all` is **GPU + parallel + auto-sharded by default** — the
same command for both modes, no extra flags:

```bash
# ab (antibody–antigen)
run_scoring.sh run-all --target-dir targets/H1382 --sto A2B2C2 --mode ab \
    --ag-chains AB --ab-chains CDEF --heavy C --light E -o OUT

# ppi (no ab/ag chains — interfaces auto-detected)
run_scoring.sh run-all --target-dir targets/H1382 --sto A2B2C2 --mode ppi -o OUT
```

What it does:

- **GPU stages on GPU.** ab: `dfire`/`mpnn` run on the GPU (~13–19× vs CPU). ppi:
  the whole ppi_v4 scorer runs on GPU.
- **Parallel fan-out (ab).** `discover → { base · cpu · dfire · mpnn } in
  parallel → merge`, instead of one sequential job. Heavy stages
  (`base/cpu/dfire/mpnn`) are **sharded** into N disjoint slices; `master` globs
  the shard parquets and concats.
- **Parallel fan-out (ppi).** the index is split into N slices, each a GPU job
  running the interface engine; a merge job concats the per-shard `xgb_dockq.csv`.
- **Auto shard count.** N from model count: `1` below ~300 models, else `≥2`,
  up to `XGB_SHARD_MAX` (3) for ~14k pools. Override with `--shards N`.
- **Self-heal.** every GPU job routes through `run_all_worker.sh`: if the GPU
  isn't visible at startup, or a CUDA/OOM error hits mid-run, it **re-allocates
  to a different node** (excluding the bad one) up to `XGB_MAXTRY` (3) times,
  then falls back to CPU. Each job logs its node, GPU, and any re-placement
  (exact problem + action) under `OUT/logs/`.
- **device recorded.** `OUT/run_meta.json` + a `device` column on every output
  parquet record whether each stage ran on `cuda` or `cpu`.

Speed (measured per-item rates → projected wall): 1.4k models ~74 min → ~4 min;
14k (MF pool) ~12 h → ~25 min.

| flag / env | effect |
|---|---|
| `--shards N` | force N shards (`1` = off; default = auto from model count) |
| `--serial` | one sequential job, no fan-out |
| `--no-gpu` / `--gres NONE` | CPU-only |
| `--gres gpu:2` | GPUs per GPU stage |
| `--ab-model v9` / `--ppi-model v6` | pick the xgb model version (auto-fallback) |
| `XGB_MAXTRY` | self-heal re-placement cap (default 3) |
| `XGB_SHARD_MAX` / `XGB_SHARD_PER` / `XGB_SHARD_MIN` | auto-shard tuning |

> Reproducibility note: GPU `mpnn_score` is not bit-deterministic, so re-running
> the same target on GPU can shift `xgb_pred` by ~0.02–0.04. CPU stages and
> `dfire` are bit-identical.

---

## PPI v4 pipeline (`run-all --mode ppi`, alias `run-ppi-v4`)

The ppi engine scores **DockQ** per protein-protein interface `(chain_a,
chain_b)` using the full `master_ppi_v4` schema and `xgb_ppi` v6. For a
multi-chain target it scores each interface (auto-detected, or selected with the
flags below). `'_v4'` in the old name is the **feature schema**, not the model
version (model is **v6**; override with `--ppi-model`).

### One command

```bash
TOOL=/gpfs/deepfold/casp/casp17/analysis/XGBDimerScorer_tool
# preferred: unified entry (gets GPU + sharding + self-heal)
$TOOL/run_scoring.sh run-all --mode ppi \
    --target-dir /gpfs/deepfold/casp/casp17/targets/H1360 \
    --sto A1B1C1D1E1 --variants af3_msa --methods af3 --target H1360 \
    -o /path/to/out
# equivalent alias:
$TOOL/run_scoring.sh run-ppi-v4 --target-dir ... --sto ... -o /path/to/out
```

(`run_scoring.sh` auto-submits to Slurm — run it from the login node. `--mode
ppi` auto-shards large pools across GPU jobs and merges; reuse an existing index
with `--index index.parquet` instead of re-discovering.)

**Which interfaces get scored** (pick one; default is auto):

| flag | behaviour |
|---|---|
| *(none)* | **auto-detect** every chain pair with ≥ `--min-contacts` (default 10) heavy-atom contacts, strongest first |
| `--hub A` | chain A vs every other chain |
| `--pairs A-B,A-C` | exactly these interfaces |
| `--chain-a A --chain-b B` | a single interface |

Output layout:

```
out/
├── xgb_dockq.csv          # the combined summary, one row per model (see below)
├── parquet/               # all parquet artifacts gathered here
│   ├── index.parquet
│   ├── master_<A-B>.parquet        # full feature table per interface (~170 cols → model uses 224 after z/onehot)
│   └── xgb_scores_<A-B>.parquet    # model_id + meta + xgb_pred (predicted DockQ)
└── csv/                   # human-readable mirrors
    ├── master_<A-B>.csv
    └── xgb_scores_<A-B>.csv
```

`xgb_dockq.csv` columns (sorted by `xgb_mean` desc):

```
model_id, target, model (method), msa (variant), seed, sample,
iptm, ptm, ranking_score, plddt_mean,          # whole-complex confidence
xgb_<A-B>, xgb_<A-C>, …,                        # predicted DockQ per interface
xgb_mean                                        # mean across interfaces
```

### ICS/IPS scoring (`--targets`)

By default scoring predicts **DockQ only** (column `xgb_pred`). To also predict the
interface-quality metrics **ICS** and **IPS**, pass `--targets`:

```bash
# PPI — all three, one pass (run-ppi-v4 or score)
$TOOL/run_scoring.sh run-ppi-v4 ... --targets dockq,ics,ips -o OUT
$TOOL/run_scoring.sh score --master master.parquet --mode ppi --targets dockq,ics,ips -o OUT

# AgAb — dockq from v9 (default), ics/ips from v10.1
$TOOL/run_scoring.sh score --master master_with_xgb.parquet --mode ab --targets dockq,ics,ips -o OUT

# AgAb — use v10.1 for dockq too (FoldBench-consistent labels across all 3)
$TOOL/run_scoring.sh score --master master_with_xgb.parquet --mode ab \
    --targets dockq,ics,ips --ab-version v10_1 -o OUT
```

`--ab-version {v9,v10_1}` (AgAb only) picks the **dockq** model family: `v9` (default,
12-method, LOPO 0.82) or `v10_1` (deepfold3 7-method, FoldBench labels, LOPO 0.766).
ICS/IPS always come from v10.1 regardless. PPI dockq is always v6.

Columns produced: `xgb_pred` (dockq, unchanged), `xgb_ics`, `xgb_ips`. In `run-ppi-v4`'s
combined `xgb_dockq.csv` each interface gets `xgb_<tag>` / `xgb_ics_<tag>` / `xgb_ips_<tag>`
plus per-target means `xgb_mean` / `xgb_ics_mean` / `xgb_ips_mean`. Model mapping is automatic
(PPI ics/ips → `xgb_ppi_v6_{ics,ips}_full`; AgAb ics/ips → `xgb_agab_v10_1_{ics,ips}_full`).
Default (no flag) is byte-identical to the old single-DockQ behaviour.

### What it computes (the `master_ppi_v4` schema — 224 features)

Ported verbatim from the benchmark training pipeline
(`/store/gunwoong/Benchmark/PPI/scripts`) into
[`xgbdimer/features/ppi_v4.py`](../../src/lantern_qa/interface/features/ppi_v4.py):

| Source script | Feature group | Examples |
|---|---|---|
| `p2_extract_features` | plDDT dist, per-chain plDDT, iface (CB 8 Å / heavy 5 Å), pdockq, JSON confidence | `cif_plddt_*`, `iface_n_contacts`, `iface_pdockq`, `json_iptm`, `chain_pair_iptm_*`, `chain_iptm_*` (dict **and** list parsed), `chain_ptm_*` |
| `p2b_extract_extended` | iface chemistry, topology, SS bonds, + EMA bbdep/ndrd/hbond | `iface_hydropathy_*`, `iface_n_salt_bridges`, `iface_n_aromatic_contacts`, `radius_of_gyration`, `asphericity`, `n_cys_total`, `bbdep_E_bbdep_chi_v2_*`, `ndrd_E_ndrd_*`, `hbond_network_score` |
| `p2c_dfire_gbind` | DFIRE on complex / a / b / pair + ΔG_bind | `complex_E_dfire`, `gbind_dfire_global`, `gbind_dfire_pair`, `gbind_dfire_pair_per_atom` |
| `p2e_dihedrals` | φ/ψ/ω + Ramachandran (global + iface) | `phi_mean`, `omega_n_cis`, `rama_outlier_frac`, `rama_iface_outlier_frac` |
| `p3_mpnn` | ProteinMPNN NLL (global + iface) | `mpnn_score`, `mpnn_iface_all_mean` |
| `p5_feature_engineering` | size logs, `iptm_minus_ptm`, target-rank | `log_N_atom`, `iptm_minus_ptm`, `*_rank_target_high/low` |

One-hot (`method_*`), per-method z (`*_z_method`, from
`normalization.json`) and per-target z (`*_z_target`, in-batch over the scored
pool) are produced at score time by `score.xgb_ppi.prep_xgb_ppi_input`.

**Efficiency:** scoring several interfaces of one complex shares all
whole-complex work (global plDDT/JSON/`chain_pair_iptm`/`chain_pair_pae` decode/
topology/Ramachandran/bbdep/ndrd/hbond/complex-DFIRE/whole-MPNN) — computed once
per cif via `run_ppi_v4_multi`, with per-chain DFIRE cached. Only interface-
specific features recompute per pair. Verified bit-identical to the naive
per-pair path and ~3× faster for 4 interfaces (more interfaces → larger
speedup). MPNN is seeded (`torch.manual_seed(0)`) so its NLL is reproducible and
consistent across a complex's interfaces.

### Coverage / known gaps

Verified on H1360 (af3_msa): **199/224 features populated = 97.8 % of model
gain** (vs ~41 % via the legacy path). Imputed-to-train-mean remainder:

- **MPNN sequence recovery** (`overall_recovery`, `mpnn_iface_recovery_mean`, +
  z/rank) — ~1.4 % gain, **intentionally not computed** (needs MPNN sampling;
  `MPNNScorer(skip_sampling=True)`).
- **`chain_pair_pae_*`** — ~0.8 % gain. The CASP `summary.json` does not carry a
  chain-pair PAE matrix; it can be reconstructed from `arrays/pae_logits.npz`
  (token PAE → chain×chain) — not yet implemented.
- `json_complex_*` (af3 lacks these Boltz keys) and merge-artifact columns —
  0 % gain.

Feature **values** were checked against the training af3 distribution: all in
range, no scale/sign bugs. Large-complex features (`gbind_dfire_global`,
`radius_of_gyration`, `asphericity`) sit at the high end of the training range
because the trainer saw mostly dimers — expected, not a bug.

---

## Install (one-time)

Uses the shared analysis venvs — same as MassiveClustering_tool. If
those exist nothing to do:

```bash
ls /gpfs/deepfold/casp/casp17/analysis/.venv/bin/python      # main venv
ls /gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin/ANARCI  # ANARCI venv (Ab mode only)
```

If missing:
```bash
cd /gpfs/deepfold/casp/casp17/analysis
bash setup.sh                    # base venv
bash setup.sh --with-cdr         # + ANARCI venv (for --mode ab)
```

The wrapper script (`run_scoring.sh`) auto-detects venvs and sets
`PYTHONPATH`, `EMA_PATH`, and HMMER `PATH` — no manual env setup needed.

### Portability — sharing with teammates

All scoring **code and data live inside the analysis tree**, so the tool is
self-contained on any node where `/gpfs/deepfold` is mounted:

- CPU/GPU scorers are imported from `EMA/scorers/`. The two parser modules they
  depend on — `hbond_energy.py` (hbond) and `bbdep_chi_energy.py` (bbdep) — are
  **vendored there**, so there is **no `/runs/users/...` personal dependency**.
- Reference data is bundled under `EMA/data/` (Dunbrack rotamer lib, NDRD,
  DFIRE table, aADARS potentials). Each resolver checks `$EMA_*` env → bundled
  file, in that order.

**To hand off:** give a `deepfold`-group teammate the path
`/gpfs/deepfold/casp/casp17/analysis/XGBDimerScorer_tool` — they can run
`run_scoring.sh` directly, no copying or env setup.

> **Caveat (venv interpreter):** `.venv`/`.venv-cdr` symlink their Python into
> the creator's `~/.local/share/uv/python/...`. This works for `deepfold`-group
> members (`/home` is shared across nodes and the dirs grant group-traverse),
> but a **non-group user is blocked** by `~/.local` permissions. For a fully
> user-independent install, recreate the venv against a shared `/gpfs`
> interpreter (e.g. `uv venv --python /gpfs/deepfold/anaconda3/envs/AlphaFold/bin/python`)
> rather than re-running `setup.sh` (which lets `uv` install Python under the
> caller's home).

---

## Sample command (verified on H1335)

```bash
TOOL=/gpfs/deepfold/casp/casp17/analysis/XGBDimerScorer_tool
$TOOL/run_scoring.sh run-all \
    --target-dir /gpfs/deepfold/casp/casp17/targets/H1335 \
    --sto A1B1C1D1E1 \
    --mode ab \
    --ab-chains DE --ag-chains ABC --heavy E --light D \
    -o /gpfs/deepfold/casp/casp17/playground/$USER/H1335_xgb_test
```

Result: 19,356 × 131 master_with_xgb.parquet with xgb_v8 scores
**byte-identical** to original H1335 scores (verified 19,356/19,356
within 1e-5).

---

## Tool layout

```
XGBDimerScorer_tool/
├── README.md                        (this)
├── docs/
│   └── FILTERS.md                   filter concepts (CDR-clean, STRICT, clash, gO regional, atom integrity)
├── pyproject.toml
├── run_scoring.py                   thin entry → cli.main
├── run_scoring.sh                   one-command wrapper (auto-resolves venvs)
├── xgbdimer/                        package
│   ├── cli.py                       argparse: discover / features / master / score / filter / cluster / pick / run-all
│   ├── core/
│   │   ├── discover.py              cif glob → index.parquet
│   │   └── master.py                stage outputs → master.parquet (left-join on model_id)
│   ├── features/
│   │   ├── base.py                  plddt + json + iface + chain_pair_iptm (~68 cols)
│   │   ├── cpu.py                   hbond + bbdep + ndrd (8-CPU parallel) — 11 cols
│   │   ├── dfire.py                 GPU DFIREv2 — 3 cols
│   │   ├── mpnn.py                  ProteinMPNN GPU — 5 cols
│   │   └── aadars.py                ANARCI CDR (.venv-cdr) — 22 cols, ab mode only
│   ├── filter/
│   │   ├── clash.py                 1.7 Å heavy atom pair count (disulfide-aware)
│   │   ├── ab_clean.py              CDR-clean + STRICT cascade (ab mode)
│   │   ├── ppi_clean.py             generic PPI sanity
│   │   └── template_rmsd.py         gH-aligned regional RMSD (Kabsch)
│   ├── features/
│   │   └── ppi_v4.py                full master_ppi_v4 schema extractor (p2+p2b+p2c+p2e+p3+p5 ports) — used by run-ppi-v4
│   ├── score/
│   │   ├── xgb_ab.py                xgb_v9_full (AgAb; +ESMFold2 ef2; falls back to v8)
│   │   └── xgb_ppi.py               xgb_ppi_v6_full (PPI; 7 methods incl OpenFold3; FoldBench labels; falls back to v5)
│   ├── cluster/
│   │   └── com_kmeans.py            COM in anchor-frame, KMeans k-sweep + silhouette
│   ├── pick/
│   │   ├── per_cluster.py           medoid + xgb_top per cluster
│   │   └── collect.py               final_picks/ folder + MANIFEST + README
│   └── viz/
│       └── _3d.py                   3D scatter / xgb gradient / filter funnel
├── config/
│   ├── ab.yaml                      sample config for Ab-Ag targets
│   └── ppi.yaml                     sample config for PPI targets
├── sbatch_templates/                cpu/dfire/mpnn/aadars Slurm templates
└── tests/                           (placeholder — see verification below)
```

---

## Verification status (against H1335 reference, 2026-05-17)

| Module | Test | Result |
|---|---|---|
| `core/discover.py` | H1335 cif glob | 19,356 cifs (exact match) |
| `score/xgb_ab.py` | H1335 master → predict | **byte-identical** to original (19,356/19,356 within 1e-5) |
| `filter/clash.py` | 12 H1335 picks vs known counts | 4/5 atom-clean exact match (pt2 minor diff due to terminal-O outlier handling) |
| `filter/ab_clean.py` | H1335 master → CDR-clean | **923 / 19,356 (4.8%) exact** |
| `cluster/com_kmeans.py` | CDR-clean subset KMeans | k=6 silhouette 0.71 (STRICT subset reaches 0.757 per H1335 REPORT) |
| `features/base.py` | 1-cif smoke test (sc2 bt2) | 68 features, json_iptm=0.839 matches |
| `features/aadars.py` | 1-cif (`.venv-cdr` ANARCI) | 23 features, e_cdr_h3=-837, frac_FR=0.247 |
| `features/cpu.py` | 1-cif (`_init_worker` + hbond/bbdep/ndrd) | 11 features, hbond_n=1018, ndrd_per_res=6.39 |
| `features/dfire.py` | 1-cif (GPU) | 3 features, dfire_v2_per_atom=14.01 |
| `features/mpnn.py` | (needs full sbatch test) | ⏳ pending |
| `viz/_3d.py` | 3-panel scatter | ⏳ pending |
| `score/xgb_ppi.py` (legacy ppi path) | PPI target | feeds only ~32/224 v4 feats — **superseded by `run-ppi-v4`** |
| `features/ppi_v4.py` + `run-ppi-v4` | H1360 af3_msa (A vs B/C/D/E) | 199/224 feats = 97.8 % model gain; values in-range vs training; CLI == demo driver |

---

## Portability hardening (re-verified 2026-05-21)

The EMA scorers used to hardcode `/runs/users/gunwoong/...`. On compute nodes
(where that path isn't mounted) this broke the pipeline: `hbond_network` raised
`ModuleNotFoundError: hbond_energy` — which killed the entire `cpu` stage
(54k errors in one H1340 job) — and `bbdep_chi_spline_v2` could not load its
rotamer-library parser (`TypeError` / silent NaN). Fix:

- `hbond_energy.py` and `bbdep_chi_energy.py` **vendored into `EMA/scorers/`**,
  imported via `Path(__file__).parent` (works both as a package and as a script).
- Every personal-path fallback removed from the `ndrd` / `dfire` / `aadars_v3` /
  `proteinmpnn` / `aadars_per_cdr` resolvers — now `$EMA_*` env → bundled
  `EMA/data/` only.

Re-verified end-to-end on **H1340 (ab)** with the personal path no longer
referenced anywhere in the import path:

| Stage | Result |
|---|---|
| `cpu` (hbond + bbdep + ndrd) | real values, 0 err, 0 NaN — bbdep loads the vendored parser |
| `dfire` | resolves bundled table → scores a real structure |
| `aadars` + ANARCI (`.venv-cdr`) | VH→type H, VL→type K numbering; full per-CDR energies |

---

## CLI subcommand reference

```
xgbdimer discover     # cif glob → index.parquet
xgbdimer features     # extract one feature group (--feature base|cpu|dfire|mpnn|aadars|clash)
xgbdimer master       # join stage outputs → master.parquet
xgbdimer score        # xgb predict → xgb_scores.parquet + master_with_xgb.parquet
xgbdimer filter       # optional in-tool filter (default: prefer MassiveClustering)
xgbdimer cluster      # optional in-tool COM KMeans (default: prefer MassiveClustering)
xgbdimer pick         # optional in-tool per-cluster pick
xgbdimer run-all      # --mode ab: discover → features → master → score
                      # --mode ppi: dispatches to the ppi_v4 interface engine
xgbdimer run-ppi-v4   # alias of `run-all --mode ppi` (PPI interface engine; '_v4' = feature schema, model is v6)
```

Via `run_scoring.sh`, `run-all` adds GPU + parallel fan-out + auto-sharding +
self-heal (see [§ Parallel execution](#parallel-execution-gpu--sharding--self-heal)).
`xgbdimer <sub> --help` lists options (run-all groups them by `ab-mode` / `ppi-mode`).

Each subcommand: `xgbdimer <sub> --help` for full options.

---

## What goes into `master.parquet`

After `run-all` (without `--apply-filter`), master has these column groups:

| Group | n cols | Source |
|---|---|---|
| meta (target, variant, method, seed, sample, model_id, ...) | 11 | `core.discover` |
| **base** (plddt percentiles + json + iface + chain_pair_iptm) | 68 | `features.base` |
| cpu (hbond + bbdep + ndrd) | 11 | `features.cpu` |
| dfire | 3 | `features.dfire` |
| mpnn (overall + iface mean/recovery) | 5 | `features.mpnn` |
| aadars (CDR-H1..L3 + framework, ab mode only) | 22 | `features.aadars` |
| clash (heavy atom @ 1.7 Å + inter-chain) | 3 | `filter.clash` |
| method_master (af3/b2/of3/v2 for xgb one-hot) | 1 | `core.master` |
| **xgb_pred** | 1 | `score.xgb_ab` or `score.xgb_ppi` |
| **total** | **~127** | |

---

## Hand-off to MassiveClustering

```bash
# After run-all produced master_with_xgb.parquet:
MC=/gpfs/deepfold/casp/casp17/analysis/MassiveClustering_tool
PY=/gpfs/deepfold/casp/casp17/analysis/.venv/bin/python

# Option A: heavy clustering (TM/iTM/dRMSD with adaptive K + Leiden)
$MC/run_clustering.sh /gpfs/.../targets/H1335 \
    --sto A1B1C1D1E1 --method all \
    --output-base /path/to/out_clustered

# Option B: filter / join xgb scores (same CLI, non-destructive, compose freely)
$PY -m massiveclustering.cli filter-strict \
    --master /path/to/master_with_xgb.parquet --mode ab \
    -o /path/to/filter_strict.parquet
$PY -m massiveclustering.cli join-xgb \
    --clustering /path/to/clustering.parquet \
    --xgb-scores /path/to/xgb_scores.parquet \
    -o /path/to/clustering_with_xgb.parquet

# Option C: interpret each cluster's binding mode — CDR × epitope contact map
$PY -m massiveclustering.cli cluster-analysis \
    --clusters /path/to/clusters.parquet \
    --contacts /path/to/contacts_long.parquet \
    --master   /path/to/master_with_xgb.parquet \
    -o /path/to/contact_analysis -t H1335 \
    --paratope-chains D,E --epitope-chains A,B,C --heavy E --light D \
    --anarci-bin /gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin/ANARCI
```

See `docs/FILTERS.md` for every QC filter, and
`MassiveClustering_tool/docs/{quickstart,advanced}.md` for the contact-map
analysis (frequency/enrichment, CDR×epitope 2D map, geometric stability).

---

## See also

- `docs/FILTERS.md` — every QC/filter concept (CDR-clean, STRICT, clash 1.7 Å, gO regional RMSD, atom integrity)
- `/gpfs/deepfold/casp/casp17/playground/gunwoong_analysis/H1335/analysis/REPORT.md §13` — full H1335 5-pick decision (reference case study this tool generalizes)
- `MassiveClustering_tool/README.md` — heavy clustering on pairwise distance matrix
