# MassiveClustering — known issues & audit log

Audit date: 2026-06-16 (CASP17 CAPRI MassiveFold run, 14 targets, ~14k models each).
Covers correctness/perf/scale across `core/gpu/`, `core/`, driver/CLI.

## FIXED (verified)

| Area | Issue | Fix |
|------|-------|-----|
| `core/gpu/drmsd.py` | Host-RAM OOM: all O(L²) per-model distance vectors materialised at once (T2342 L=4365 → 347 GB; H2340 → 119 GB) → OOM-killed. GPU side was chunked, host side was not. | Host-RAM guard + uniform CA stride subsampling (`DRMSD_HOST_BUDGET_GB`, default 32). Validated: T2342 stride 4, H2340 stride 2, matrices complete. Disk output unchanged (~756 MB N×N). |
| `core/gpu/` drmsd, tmscore, itm, com, anchored | Structure loading was single-threaded gemmi parse (~10 min for 14k). | Shared parallel loader in `_loader.py` (`parse_structure_ca_batch` / `parse_structure_with_chains_batch`, fork Pool + `imap` order-preserving + serial fallback). com/anchored: parse parallel, Kabsch align stays serial. **Validated parallel==serial for all 5 metrics (atol 1e-6); ~56× faster loading.** |
| `core/gpu/drmsd.py` | `compute_drmsd_matrix_multi_gpu` referenced undefined `exclude_indices`. | Routed through batch loader (NameError gone). NB: this fn still has an OOM (see latent below) and is dead code. |
| `run_clustering.py` | `--flat-dir` still required `results/<sto>/` → bailed on minimal targets (H2348/H2349). | Fall back to `base=target_dir` when absent in flat-dir mode. |
| `core/consensus.py` | Refinement substituted `ref_coords` into slot 0, desyncing coords from `valid_indices` when `reference_idx != cluster_indices[0]` (the normal medoid-passing path) → wrong medoid attribution, dropped member 0, double-counted reference in mean/RMSF. | Re-align every member in lockstep; don't substitute ref into slot 0. Validated. |
| `run_clustering.py` | `--list` ran `verify_sequence_consistency` (14k gemmi parses) before returning → hammered login node. | Return before the consistency guard. |

## LATENT — real but not hit by the current pipeline (NOT fixed)

Only the code paths we exercise (tm/drmsd/itm/com/iface-jaccard-residue, default Leiden network, fuzzy clustering) were in scope to fix. These remain:

- **`core/network.py` — O(N²) float64 co-association + Python double-loops.** `bootstrap_community_stability`, `multiseed_communities`, `consensus_communities` allocate N×N float64 (1.6 GB each) and fill via nested Python loops over all pairs, per bootstrap/seed/resolution. On ~14k nodes → OOM / effectively hangs. **Gated behind `--stability` / `--n-seeds` / `--consensus-communities` / `--resolution-scan`** (we don't pass these). Fix before using those flags at pool scale: gate on N / subgraph only, float32, vectorize co-assoc (one-hot outer product / `np.add.at`).
- **`core/iface_metrics.py` — chamfer/wasserstein N² serial pair-fill** rebuilding cKDTrees per pair (Sinkhorn per pair). Unusable at ~14k. We use `iface-jaccard-residue` only. Fix before using chamfer/wasserstein: precompute per-model KDTrees once, parallelize the upper-triangle fill.
- **`core/gpu/drmsd.py:compute_drmsd_matrix_multi_gpu` — OOM.** Builds `(batch, N, n_dist)` broadcast diff (no gram trick, no chunking, no stride guard). Dead code (driver uses single-GPU). Fix or delete; until then route callers to single-GPU.
- **`run_clustering.py` — `--skip-existing` ignores CDR mode.** The `cached` cross-check (cdr_residues.json presence vs `--cdr`) is computed then discarded; the live reuse condition omits it → could reuse a non-CDR matrix for a `--cdr` run. We don't use `--cdr`/`--skip-existing`.
- **`core/consensus.py` — `medoid_indices[i]` positional vs label.** `i` enumerates `unique_labels` incl. `-1` (the `-1` `continue` advances `i`), so with noise labels the medoid index desyncs. No noise under fuzzy c-means (our default); bites DBSCAN/HDBSCAN.
- **`core/gpu/com_distance.py:185` — `np.stack([])` crash** if every model fails to parse (degenerate all-bad pool). Add `if not com_buf: raise`.
- **`core/network.py` — full graph built twice** in the network analysis (alternate community method rebuilds from the matrix). Wasted O(N²) pass, non-fatal.
- **CLI/driver — no all-NaN/degenerate matrix guard** before clustering; `--flat-dir` `.cif` bundles miss ranking rows (join key is exact `.pdb` string). Low/edge.

## Notes
- pdist over models in dRMSD stays serial **by design** — parallelising via a process pool would ship O(L²) distance vectors over IPC (hundreds of GB), slower than the C-level serial pdist. The loading parallelisation already removed the dominant bottleneck.
- Padded TM/iTM/dRMSD kernels assume equal-length, positionally-aligned models — guaranteed upstream by `verify_sequence_consistency`, but the kernels don't self-assert it (latent if called standalone).
