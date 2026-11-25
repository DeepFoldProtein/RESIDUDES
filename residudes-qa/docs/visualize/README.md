# AF3 Prediction Result Visualization

Analyzes AlphaFold3 prediction results for a target directory and produces
PNG plots and CSV files summarizing structure quality metrics.

Runs entirely inside an Apptainer container (`af3-viz.sif`) — no Python
environment setup required. Any account on the system can run it as-is.

---

## Usage

```bash
/data/casp17/visualization/analyze_af3_target.sh [options] <target_dir>
```

`<target_dir>` is the root directory of a CASP target (e.g.
`/data/casp17/targets_test/H1106`). It must contain AF3
inference results under the standard unified output schema layout:

```
<target_dir>/
  A/results/<sto>/<msa_variant>/<model>/seed_<N>/**/sample_<i>/   ← monomer chain A
  B/results/<sto>/<msa_variant>/<model>/seed_<N>/**/sample_<i>/   ← monomer chain B
  results/<sto>/<msa_variant>/<model>/seed_<N>/**/sample_<i>/     ← complex (A1B1, …)
```

`<msa_variant>` is the MSA pipeline (e.g. `af3_msa`, `mmseqs2_gpu`,
`plmmsa_otalign`) and `<model>` is the prediction tool (e.g. `af3`, `bt2`,
`of3`, `pt2`). Every variant × model combination found is analyzed
independently, and rankings are keyed by
`(stoichiometry, msa_variant, model, seed, sample)`.

Each `sample_<i>/` directory must contain `model.cif`, `summary.json`, and
an `arrays/` subdirectory with `.npy` confidence files.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dpi N` | `150` | PNG resolution in DPI |
| `--no-plots` | — | Skip PNG generation, write CSV only |
| `--no-csv` | — | Skip CSV generation, write PNGs only |
| `-v`, `--verbose` | — | Enable debug logging |
| `-h`, `--help` | — | Show help |

### Examples

```bash
# Run on a target (output goes to <target_dir>/visualization/)
analyze_af3_target.sh /data/casp17/targets_test/H1106

# CSV only (fast, no matplotlib rendering)
analyze_af3_target.sh --no-plots \
    /data/casp17/targets_test/H1106
```

---

## Output Files

Output is written to `<target_dir>/visualization/` mirroring the input path:

```
<target_dir>/visualization/
  <sto>/<msa_variant>/<model>/   ← per-run plots and CSV (one dir per prediction type)
  per_seed_summary.csv           ← best metrics per seed across all prediction types
  summary.csv                    ← top-5 models per prediction type
```

For example, `results/A1B1/af3_msa/af3/` → `visualization/A1B1/af3_msa/af3/`,
and monomer chain B → `visualization/B1/af3_msa/af3/`.

### PNG plots (per prediction type)

| File | Content |
|------|---------|
| `contact_prob.png` | Contact probability heatmap (best model) |
| `distogram.png` | Distogram — 2 rows (expected / argmax) × N seeds |
| `msa.png` | MSA coverage heatmap with sequence identity coloring |
| `plddt_seed<N>.png` | Per-residue pLDDT — all samples overlaid (one per seed) |
| `pae_seed<N>.png` | PAE heatmap — one panel per sample (one per seed) |
| `pde_seed<N>.png` | PDE heatmap — one panel per sample (one per seed) |
| `ramachandran_seed<N>.png` | Ramachandran plot — one panel per sample (one per seed) |

### CSV files

| File | Content |
|------|---------|
| `<run_dir>/all_rankings.csv` | All (seed, sample) pairs with quality metrics, sorted by `ranking_score` desc |
| `visualization/per_seed_summary.csv` | Best metrics per seed for each prediction type |
| `visualization/summary.csv` | Top-5 models per prediction type |

#### Columns in `all_rankings.csv`

| Column | Description |
|--------|-------------|
| `stoich` | Stoichiometry (e.g. `A1`, `A1B1C1D1E1`) |
| `msa_variant` | MSA pipeline variant (e.g. `af3_msa`, `mmseqs2_gpu`, `plmmsa_otalign`) |
| `model` | Prediction model/tool (e.g. `af3`, `bt2`, `of3`, `pt2`) |
| `seed` | Inference seed |
| `sample` | Sample index within the seed |
| `ranking_score` | AF3 composite score (pTM for monomers, 0.8×iPTM + 0.2×pTM for complexes) |
| `ptm` | Predicted TM-score |
| `iptm` | Interface pTM (complexes only) |
| `fraction_disordered` | Fraction of tokens predicted disordered |
| `has_clash` | Whether the model has steric clashes |
| `inter_pde_mean` | Mean Predicted Distance Error in Å (gpde from summary.json) |

---

## Directory Contents

```
visualization/
  analyze_af3_target.sh   Entry point (run this)
  analyze.py              Main Python script
  analysis/               Python package (reader, metrics, plots, ranking)
  af3-viz.sif             Apptainer container image (Python 3.12 + dependencies)
  af3-viz.def             Container build recipe
```

---

## Rebuilding the Container

The container only needs to be rebuilt if Python package dependencies change.
**Code changes** (`analyze.py`, `analysis/`) take effect immediately — the
`visualization/` directory is bind-mounted into the container at runtime.

To rebuild (requires an admin account with Apptainer build privileges):

```bash
cd /data/casp17/visualization
apptainer build af3-viz.sif af3-viz.def
```
