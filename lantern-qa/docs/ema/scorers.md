# Scorers

Each module exposes a single scoring function (or class) consuming a PDB path
and returning a scalar / per-residue array. See
../docs/METRICS.md for formulas and validation, and
../docs/METRIC_LIST.md for the complete metric
inventory.

## Module index

| File | Score | Dir | Validated on |
|---|---|:-:|---|
| [`proteinmpnn_score.py`](../../src/lantern_qa/ema/scorers/proteinmpnn_score.py) | ProteinMPNN per-residue NLL (general QA) | ↓ | CASP14/15/16, MF |
| [`mpnn_iface.py`](../../src/lantern_qa/ema/scorers/mpnn_iface.py) | MPNN global + interface mask + ensemble | ↓ | CASP15/16, MF, Ab-Ag |
| [`aadars_v3_scorer.py`](../../src/lantern_qa/ema/scorers/aadars_v3_scorer.py) | CDR×AA_ag CB-CB potential | ↓ | 11 abag |
| [`aadars_per_cdr.py`](../../src/lantern_qa/ema/scorers/aadars_per_cdr.py) | per-CDR breakdown (H1/H2/H3/L*/FW) | ↓ | 11 abag |
| [`dfire_v2_scorer.py`](../../src/lantern_qa/ema/scorers/dfire_v2_scorer.py) | DFIRE pair potential v2 | ↓ | MF mono, cluster medoid |
| [`bbdep_chi_spline_v2.py`](../../src/lantern_qa/ema/scorers/bbdep_chi_spline_v2.py) | Backbone-dep χ rotamer | ↑ | (use DEVIATION transform) |
| [`ndrd_scorer.py`](../../src/lantern_qa/ema/scorers/ndrd_scorer.py) | Neighbor-dep Ramachandran | ↑ | MF mono, cross-pred |
| [`hbond_network.py`](../../src/lantern_qa/ema/scorers/hbond_network.py) | Mayo H-bond functional | ↓ | CASP16 (12/12 sign) |
| `rosetta_score.py` | Rosetta REF15 / scorefxn (3-backend wrapper) | ↓ | comparison baseline (optional) |

## Conventions

- Each `score_*` function is **pure** (no global state, no side effects).
- Per-residue arrays are aligned with **MPNN's iteration order** (chain-then-residue, AA20 only, full backbone present).
- Direction `↓` (lower better) → wrap as `-x` before z-scoring.

## Path resolution (data files)

All scorers that need data files use a **3-tier fallback** chain:

1. **Environment variable** (e.g. `EMA_BBDEP_LIB`, `EMA_NDRD_PATH`,
   `EMA_AADARS_V3_POT`, `EMA_PROTEINMPNN_DIR`) — for users with custom paths.
2. **Bundled in this repo** under `EMA/data/dunbrack/` or `EMA/data/potentials/`.
3. **Legacy dev path** on the original cluster (only as last resort).

Failure raises `FileNotFoundError` with an explanatory message and a fix
suggestion.

## Direction-flip warnings

| Metric | Caveat |
|---|---|
| `bbdep_chi_spline_v2` | **89% sign-flip on cross-predictor pools.** Apply DEVIATION transform: `|x - target_median|`. See STATISTICS.md §4 / CC5. |
| `hbond_network.score_hbond_per_res` | 56% sign on MF — use `score_hbond_total` instead. |

## ProteinMPNN — now top-level

ProteinMPNN was previously located inside `MassiveClustering_tool/`. It has
been moved to **[`proteinmpnn_score.py`](../../src/lantern_qa/ema/scorers/proteinmpnn_score.py)** here because
it is a **general QA tool**, not a clustering primitive. The old import path
still works via a re-export stub:

```python
# Preferred
from EMA.scorers.proteinmpnn_score import MPNNScorer

# Backwards-compatible (re-exports the same class)
from massiveclustering.confidence.mpnn_scoring import MPNNScorer
```

`MPNNScorer()` now auto-finds the bundled
`MassiveClustering_tool/proteinmpnn/run/` directory and weights — no path
needed:

```python
from EMA.scorers.mpnn_iface import score_pdb
result = score_pdb("model.pdb")
# → {'mpnn_global': ..., 'mpnn_iface': ..., 'n_iface': ..., 'n_total': ...}
```

Run `bash MassiveClustering_tool/proteinmpnn/setup_weights.sh`
once to fetch the ~70 MB of model weights (MIT, from
[`dauparas/ProteinMPNN`](https://github.com/dauparas/ProteinMPNN)).

## Rosetta wrapper — optional baseline

`rosetta_score.py` is a 3-backend wrapper:
- PyRosetta (preferred — Python API)
- Rosetta CLI binary (`score_jd2.linuxgccrelease`)
- Graceful error with install instructions if neither

```python
from EMA.scorers.rosetta_score import RosettaScorer, is_available
if is_available():
    s = RosettaScorer(scorefxn='ref15')
    result = s.score('model.pdb')
    # → {'total_score': -123.4, 'per_term': {...}, 'backend': 'pyrosetta', ...}
```

**Known issue (2026-04):** the Rosetta wheel for Python 3.11+ has an ABI
mismatch (`undefined symbol: PyCode_GetVarnames`). Use Python 3.10 for
PyRosetta, or fall back to the CLI binary.

## Smoke test

```bash
python EMA/scripts/all_metrics_smoke.py
```

Runs every scorer on bundled test PDBs (1UBQ + 2OOB) and reports pass, fail or soft failure for each.
