# Dunbrack Lab libraries (BBDep + NDRD + CDL)

Side-chain rotamer and Ramachandran libraries from the [Dunbrack Lab](http://dunbrack.fccc.edu/),
Fox Chase Cancer Center.

## License

> **Free for academic / non-commercial use** with proper citation.
> Redistribution restricted; commercial use requires a separate license from the Dunbrack Lab.

The full license text is in `bbdep2010_license.txt` and
`handy_license_reference.txt`.

These files are bundled here for **internal team / research use only**. If this
repository is ever made public, the libraries should be removed and users should
download them directly from http://dunbrack.fccc.edu/.

## Citations (required when using these data)

- **BBDep 2010** — Shapovalov & Dunbrack. *A smoothed backbone-dependent rotamer
  library for proteins derived from adaptive kernel density estimates and
  regressions.* Structure **19**, 844–858 (2011).

- **NDRD TCBIG** — Ting et al. *Neighbor-dependent Ramachandran probability
  distributions of amino acids developed from a hierarchical Dirichlet process
  model.* PLOS Comp Bio **6**(4), e1000763 (2010).

- **CDL (Conformation Dependent Library)** — Berkholz et al. *Conformation
  dependence of backbone geometry in proteins.* Structure **17**, 1316–1325 (2009).

## Files

| File | Size | Used by | Description |
|---|---:|---|---|
| `ALL.bbdep.rotamers.lib` | 80 MB | `bbdep_chi_spline_v2.py` | Backbone-dependent rotamer probabilities (φ, ψ → 81 rotamer classes per AA) |
| `NDRD_TCBIG.txt.gz` | 69 MB | `ndrd_scorer.py` | Neighbor-dependent Ramachandran prob (TCBIG = Turn+Coil+Bridge+PIHelix+310Helix). **Renamed** from upstream `NDRD_TCBIG (1).txt.gz` to remove the space |
| `KernRegr_CDL_v1.2.1_Oct14-2009.txt` | 2.2 MB | `cdl_scorer.py` (optional) | Conformation-dependent ideal bond geometries via kernel regression |
| `omegaCDL_OmegaBetweenAsPhi1Psi0_KernRegr_v1.3.1_Aug12-2011.txt` | 786 KB | `cdl_scorer.py` (optional) | Per-residue ideal omega angles |
| `SimpleOpt1-25.zip` | 8.2 MB | (reference) | Top500 optimal-rotamer benchmark data |
| `bbdep2010_license.txt` | 20 KB | — | License text (Dunbrack Lab) |
| `handy_license_reference.txt` | 2.1 KB | — | Quick license summary |
| `Description.txt` | 1.3 KB | — | Library description |

## How the scorers find these files

The path resolution chain (highest priority first):

1. **Environment variable** — `EMA_BBDEP_LIB` for BBDep, `EMA_NDRD_PATH` for NDRD.
   Useful when you want to point at an alternative local copy.
2. **Bundled** — `EMA/data/dunbrack/ALL.bbdep.rotamers.lib` and
   `EMA/data/dunbrack/NDRD_TCBIG.txt.gz` (the files in this directory).

If neither resolves, the scorer raises `FileNotFoundError` with an explanatory
message. (There is no longer any hardcoded personal/dev-cluster fallback — the
resolver is env → bundled only, so it works for any user on any node where this
tree is mounted.)

## Cache

`ndrd_scorer.py` parses NDRD_TCBIG once and caches the parsed table as
`ndrd_tcbig_table.pkl` in this directory. First call takes ~30 s; subsequent
calls are instant. The cache is auto-regenerated if missing.

## Removing the libraries (if you go public)

```bash
# Drop the heavy data files but keep license + README
rm EMA/data/dunbrack/ALL.bbdep.rotamers.lib
rm EMA/data/dunbrack/NDRD_TCBIG.txt.gz
rm EMA/data/dunbrack/KernRegr_CDL_v1.2.1_Oct14-2009.txt
rm EMA/data/dunbrack/omegaCDL_*.txt
rm EMA/data/dunbrack/SimpleOpt1-25.zip
# Then re-add `EMA/data/dunbrack/*.lib` and `*.txt.gz` to .gitignore
# and tell users to run setup_dunbrack.sh
```
