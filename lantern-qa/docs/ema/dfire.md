# DFIRE v2 — energy table

`dfire_v2_energy_table_with_types.pt` (8.5 MB)

## What is this

A **custom-trained DFIRE-style pair potential** built in-house at DeepFold:

- 167 atom types (heavy atoms, fine-grained)
- Trained on **PDB natives** + decoys from **CASP14/15** submissions
- Functional form: `E_total = E_pos − α · E_neg` with `α = 1.57` (learned from training)
- Bin width: 0.5 Å, cut-off via stored `cut_bin`

This is **not** the original Zhou & Zhou 2002 DFIRE — it's a re-derivation
trained against a more recent decoy set.

## File schema (pickle / torch.load)

```python
import torch
d = torch.load('dfire_v2_energy_table_with_types.pt', weights_only=False)
# d.keys(): E_pos, E_neg, alpha, types, type_index, valid_pairs,
#           bin_centers, num_bins, cut_bin
```

| Key | Shape / type | Meaning |
|---|---|---|
| `E_pos` | `[n_types, n_types, n_bins]` | favorable-side bin energies |
| `E_neg` | `[n_types, n_types, n_bins]` | unfavorable-side bin energies |
| `alpha` | float | training-learned weight for `E_neg` |
| `types` | list[str] (167 entries) | atom-type names (e.g. `'ASP_OD'`) |
| `type_index` | dict[str→int] | reverse lookup |
| `valid_pairs` | set[(i,j)] | pair indices with sufficient training data |
| `bin_centers` | float array | distance bin centers (Å) |
| `num_bins` | int | number of distance bins |
| `cut_bin` | int | bin index of the energy cutoff |

## How the scorer finds this

`EMA/scorers/dfire_v2_scorer.py` uses 2-tier path resolution (env → bundled;
no hardcoded personal/dev-cluster fallback):

1. `$EMA_DFIRE_V2_TABLE` env var (custom path)
2. Bundled at `EMA/data/dfire/dfire_v2_energy_table_with_types.pt`

```python
from EMA.scorers.dfire_v2_scorer import DFIREv2Scorer
s = DFIREv2Scorer()  # auto-finds bundled file
r = s.score('model.pdb')
# {'E_dfire_v2': ..., 'E_dfire_v2_per_atom': ..., 'n_atoms': 601, 'n_pairs': 87317}
```

## Attribution

- **Original DFIRE concept**: Zhou & Zhou, *Distance-scaled, finite ideal-gas
  reference state*, *Protein Sci.* 2002.
- **This v2 retraining**: DeepFold internal work (Gunwoong Park, 2024).
  Uses the same DFIRE functional form but new atom types and a re-derived
  reference state from CASP14/15 decoys.

## Companion files (optional, in `EMA/data/dfire/`)

These older variants are **not** bundled in this repo but exist on the dev
cluster — if you need them, copy from
`/runs/users/gunwoong/work/DFIRE/DFIRE_torch/library/`:

- `2A10dfire40_pair.lib` (8 MB) — earlier pair-only library
- `2A10dfire40_gaussian.pt` (16 MB) — Gaussian-smoothed variant
