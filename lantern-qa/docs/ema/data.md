# EMA Data

This directory holds:

| Subdir / file | Source | Tracked in git? |
|---|---|---|
| `potentials/aadars_v*.pt` | trained by `EMA/builders/build_aadars_v*.py` | bundled (small, <2 MB) |
| `dunbrack/*.lib`, `dunbrack/*.txt.gz` | Dunbrack Lab | bundled in **team-internal** builds (academic/non-commercial use); see `dunbrack/README.md` |
| `setup_dunbrack.sh` | this repo (verifies bundled files OR points to upstream) | bundled |
| Large training caches (`role_pairs_*.pt`, `sabdab_cdr_rotamer_stats.pt`) | builders | not bundled, regenerate |

## Setup

```bash
# 1. Dunbrack libraries (BBDep, NDRD) — required by bbdep_chi_spline_v2 / ndrd_scorer
bash setup_dunbrack.sh

# 2. (Optional) re-train aADARS potentials from scratch
sbatch ../slurm/sbatch_aadars_v8.sh
```

## Potentials

| File | Cells | Decoy gen | Use |
|---|---|---|---|
| `aadars_v4.pt` | 73,000 | rotation, AA×AA pair | legacy |
| `aadars_v5_ab.pt` | 2,240 | rotation, closest-heavy | legacy |
| `aadars_v5_nb.pt` | 1,280 | nanobody-only rotation | legacy |
| `aadars_v7_pilot.pt` | 2,240 | FFT random surface, 80 decoys/native | recommended for general Ab-Ag |

Production `v7` and `v8` (CDR-proximal FFT) are still being computed — see `slurm/sbatch_aadars_v8.sh`. The `_pilot` was the 80-decoy proof-of-concept; full versions use 100/native and 1000-complex pool.

## Tensor Format

```python
import torch
pot = torch.load("potentials/aadars_v7_pilot.pt")
# pot keys: 'energy', 'metadata'
# pot['energy'].shape == [n_cdr_types=8, n_aa=20, n_distance_bins=14]
# pot['metadata'] == {'r_min': 0.0, 'r_max': 8.0, 'cdr_types': [...], ...}
```

## Storage policy

- Files **< 5 MB** can live in git directly.
- Files **5 MB – 100 MB**: use [Git LFS](https://git-lfs.github.com/) or attach to GitHub release.
- Files **> 100 MB** (e.g. `role_pairs_dataset_v2.pt` at 190 MB): never commit — regenerate from builders.
