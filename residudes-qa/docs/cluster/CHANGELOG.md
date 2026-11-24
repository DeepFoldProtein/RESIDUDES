# Changelog

## 2026-06-17 — CASP17 CAPRI MassiveFold run (14 targets, ~14k models each)

Performance, scale, and correctness work driven by large / multi-copy targets
(T2342 spike A3, H2340/H2354/H2355 A3B3C3 antibody trimers, H2356 A8B4, etc.).
All changes verified against the pre-change behaviour (see "verified" notes).

### Performance
- **Parallel structure loading** for all GPU metric kernels (drmsd / tm / itm /
  com / anchored). Shared loader in `core/gpu/_loader.py`
  (`parse_structure_ca_batch`, `parse_structure_with_chains_batch`,
  `_parallel_map`): fork `Pool` + `imap` (input order preserved) + automatic
  serial fallback. Worker count from `$SLURM_CPUS_PER_TASK` (else cpu_count).
  *Verified parallel == serial (atol 1e-6) for all 5 metrics; ~56× faster
  loading (2000 files 46.6 s → 0.8 s @16 workers).*
- **Vectorized iface-jaccard matrices** (`core/iface_metrics.py`,
  `iface_jaccard_residue_matrix` + `iface_jaccard_contact_matrix` via new
  `_jaccard_distance_from_sets`). Replaced the O(N²) pure-Python frozenset
  double-loop (~99 M set ops at N=14070 → hours) with a sparse incidence
  matrix + one matmul: `I = B·Bᵀ`, `union = |Qi|+|Qj|−I`.
  *Verified identical to the loop (incl. empty-set edge cases); hours → seconds.*

### Scale / feasibility (previously crashed, now complete)
- **dRMSD host-RAM guard** (`core/gpu/drmsd.py`): the O(L²) per-model distance
  vectors were materialised at once (T2342 L=4365 → 347 GB; H2340 → 119 GB) and
  OOM-killed the process before the GPU was used. Added a host-RAM budget +
  uniform CA-stride subsampling (`DRMSD_HOST_BUDGET_GB`, default 32 GB). Disk
  output unchanged (~756 MB N×N matrix). *Verified: T2342 stride 4, H2340
  stride 2, both complete.*
- **`--flat-dir` no longer requires `results/<sto>/`** (`run_clustering.py`):
  falls back to `base=target_dir` (label "complex") when the in-house results
  tree is absent — needed for minimal external-bundle targets.
- Fixed `compute_drmsd_matrix_multi_gpu` `NameError` (referenced an
  `exclude_indices` it never took). NB: that fn still lacks the gram-trick /
  chunking of the single-GPU path — see KNOWN_ISSUES.

### Correctness
- **Consensus medoid/RMSF fix** (`core/consensus.py`): the stable-residue
  refinement substituted the reference into slot 0, desyncing coords from
  `valid_indices` whenever `reference_idx != cluster_indices[0]` (the normal
  medoid-passing path) → wrong medoid attribution, dropped member 0,
  double-counted reference. Now re-aligns every member in lockstep.
- **`--list` returns before the consistency guard** (`run_clustering.py`): it
  used to parse all ~14k structures on the login node despite being advertised
  as "no work, no Slurm".
- **comprehensive PDB lookup resolves extensions** (`cli.py`): structure names
  may be bare basenames while the pool holds `<name>.pdb`/`.cif` (iface flat-dir
  mode) — now tries both, so jaccard medoid/consensus export works.

### Tooling
- **Generic role-based chain relabel** for multi-copy pools where backends
  disagree on chain order (af3 per-protomer interleaved vs afm/cf per-role
  grouped → same AₙBₙCₙ assembly, different per-chain length vector → TM/dRMSD
  "shape MISMATCH" reject). `playground/gunwoong_analysis/_mf_relabel_generic.py`
  buckets chains by length and renames to a canonical order. Use this instead of
  `canonicalize-chains` for ≥7-chain assemblies (that brute-forces n_chains! and
  hangs).

See **KNOWN_ISSUES.md** for latent issues not exercised by this run (chamfer/
wasserstein N² fill, network stability/seed builders O(N²), multi-GPU dRMSD OOM,
homo-oligomer TM permutation sensitivity, `--skip-existing` CDR-mode reuse).
