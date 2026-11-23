# MassiveClustering

GPU-accelerated protein structure clustering for **CASP17 immune complex pose selection** — pick 5 submission models from pools of 10,000+ predictions (AF3, Boltz2, OpenFold3, Protenix v2, MassiveFold).

---

## Install

```bash
bash analysis/setup.sh                  # base venv (.venv/)
bash analysis/setup.sh --with-cdr       # + ANARCI venv (for CDR detection)
```

Verify:
```bash
analysis/.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## One-command clustering

```bash
bash run_clustering.sh /path/to/targets/H1311 \
     --sto A1B1C1D1E1 --variant af3_msa_8yiv \
     --method tm \
     --output-base /path/to/playground/H1311_clustering/af3_msa_8yiv \
     --log-dir /path/to/playground/H1311_clustering/_slurm_logs
```

Pipeline: discover CIFs → GPU pairwise distance matrix → adaptive K clustering → network analysis (Leiden + Louvain) → consensus + medoid PDB export. Slurm-managed, parallel across metrics.

---

## What's supported

| Capability | Notes |
|---|---|
| **Distance metrics** | `tm` (TM-score), `itm` (interface TM), `drmsd` (CA-pair RMSE), `anchored-rmsd`/`anchored-tm` (probe within anchor frame), `com` (probe COM in anchor frame), `tcr-angle` (Pierce crossing/incident); **interface** metrics via `iface-matrix`: `iface-jaccard-residue/-contact` (contact pattern, set-based) + anchor-aligned `iface-chamfer`/`iface-wasserstein` (binding-mode geometry — best cluster recovery on H1335, silhouette 0.74) |
| **CDR-aware** | ANARCI IMGT scheme; `--probe-spec AB:cdr`, `--anchor-mode framework`, etc. |
| **Cluster algorithms** | Fuzzy C-means (default), K-means, hierarchical, Leiden |
| **Adaptive K** | 4-signal composite (silhouette + BIC + balance + plateau) |
| **Network analysis** | KNN graph (k = 2·log₂N), Leiden + Louvain dual, hubs, transition matrix; **robustness**: `--resolution-scan`, `--stability` (bootstrap ARI + co-association), `--consensus-communities` (fuse partitions across resolutions) |
| **Multi-metric / consensus** | Cluster on complementary metrics (global TM/dRMSD + interface jaccard/chamfer) and check agreement — a single metric/run is not enough for robust structure clustering |
| **Medoid output** | Per-cluster representative CIF/PDB + 2-view (xz, xy) face PNGs |
| **Consensus + RMSF** | Iterative Kabsch, per-residue conserved-segment detection |
| **Confidence aggregation** | iPTM, pTM, pLDDT, ranking_score per cluster from `summary.json` |
| **Provenance** | Which generator (AF3/Boltz/Protenix/OpenFold/MF) dominates each cluster |
| **Contact-map analysis** | `cluster-analysis`: per-cluster epitope frequency + enrichment, CDR×epitope 2D map (ANARCI loops), physicochemical channels, geometric stability (mean±std min-dist) — see below |
| **XGB bridge** | `join-xgb` + `filter-*` (clash, cdr-clean, atom-integrity, regional-rmsd, strict) operate on XGBDimerScorer master parquets — see below |
| **Terminal repair** | `repair-terminal`: snaps mis-placed terminal carboxyl O/OXT (Protenix bug) back onto sane geometry — **keeps** the model, records which were fixed in a manifest |
| **Submission picks** | `submission-picks`: RRF-fuse heterogeneous scores + diversity-aware selection across clusters (reference starting point) |

---

## Documentation

- **[docs/quickstart.md](notes/quickstart.md)** — install + first cluster + reading outputs (5 min tutorial)
- **[docs/advanced.md](notes/advanced.md)** — distance metric choice, anchor/probe design, K selection, sub-clustering, submission picker
- **[docs/troubleshooting.md](notes/troubleshooting.md)** — common errors (Slurm log dir permission, dRMSD OOM, Pierce/ANARCI issues, biopython 1.81 collision)
- **[CLUSTERING.md](CLUSTERING.md)** — original deep-dive on CASP17 idioms (verbose, kept for completeness)
- **[CHANGELOG.md](CHANGELOG.md)** — recent perf/scale/correctness changes (parallel loading ~56× on 2,000 files at 16 workers, vectorized iface-jaccard, dRMSD host-RAM guard, consensus fix, …)
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** — latent issues + which code paths are/aren't pool-scale-safe (read before using chamfer/wasserstein, network `--stability`/`--n-seeds`, or multi-GPU dRMSD)

---

## CASP17 case studies

Real submission decisions documented in:
- H1311 (TCR-pMHC class-I) report
- H1312 (Anti-EXT1/EXT2 nanobody) report

Each compares server submission (2 weeks ago) vs human submission (this round) with clustering setup, picks, scores, and risk notes.

---

## Output layout

```
<output-base>/<sto>/<variant>/<method>/
├── matrices/        <method>_matrix.npy + structure_names.npy + anchor_probe.json
├── ranking.csv      per-prediction confidence joined from summary.json
├── _pdb_view/       CIF→PDB cache (medoid/consensus use this) — removed at end of run; pass `--keep-pdb-view` to retain
└── results/
    ├── <method>_clustering.csv   (cluster, structure, membership, UMAP coords)
    ├── <method>_cluster_stats.csv
    ├── <method>_comprehensive.png
    ├── <method>_k_diagnostic.png
    ├── network/
    │   ├── summary.json          (Leiden Q, n_communities, hubs)
    │   ├── summary_louvain.json  (Louvain cross-check)
    │   ├── transition.csv
    │   ├── report.png
    │   └── pretty_community_graph.png
    ├── representatives/
    │   ├── <target>_representatives.csv
    │   └── cluster*_medoid.pdb
    └── consensus/                per-cluster mean coords + RMSF
```

Post-processing (medoid_faces.png, medoid_iface.csv, per-rep face PNGs, submission picker) lives in `playground/gunwoong_analysis/<target>_clustering/`.

---

## Python CLI (low-level)

For cases where you already have a distance matrix:

```bash
massivecluster comprehensive \
    --matrix tm:tm_matrix.npy drmsd:drmsd_matrix.npy \
    --names structure_names.npy \
    --pdbs _pdb_view/ \
    --output results/ \
    --target H1311 \
    --method fuzzy

massivecluster network --matrix dist.npy --labels labels.npy -o network/
```

Subcommands: `comprehensive`, `run`, `network`, `analyze`, `report`, `heatmap`, `consensus`, `mpnn-score`, `superposition-core`, `inpaint`, `iface-matrix`, `cluster-analysis`, `submission-picks`, `repair-terminal`, `canonicalize-chains`, `join-clusters` (see below), plus the XGB bridge `filter-*` / `join-xgb`. Run `massivecluster <cmd> --help` for options.

Quick map of the newer / less-obvious ones:

| Subcommand | What it does | When to reach for it |
|---|---|---|
| `iface-matrix` | builds an interface distance matrix (jaccard/chamfer/wasserstein); feeds back into `comprehensive`/`run`/`network` as a `.npy` | binding-mode clustering — global TM/dRMSD miss "where on the antigen does it sit" |
| `cluster-analysis` | per-cluster epitope frequency + enrichment, CDR×epitope 2D map, chemistry channels | Ab–Ag / generic PPI interpretation after clustering |
| `submission-picks` | RRF-fuse heterogeneous scores + diversity-aware selection across clusters | choose final N submission models |
| `repair-terminal` | snaps mis-placed terminal carboxyl O/OXT (pt2 bug) back onto sane geometry; writes manifest | pt2 models flagged by `filter-atom-integrity` — keeps the model |
| `canonicalize-chains` | relabels homo-oligomer chains into a common frame | run BEFORE matrix/clustering if predictors transposed chain labels (TM/dRMSD will blow up otherwise) |
| `consensus` | per-cluster mean coords + per-residue RMSF (iterative Kabsch) | identify conserved core vs flexible regions across a cluster |
| `join-clusters` | merge cluster assignments from independent metric runs | cross-metric consensus / multi-method agreement table |

---

## CDR × epitope contact-map analysis (`cluster-analysis`)

Once you have cluster assignments for Ab–Ag complexes, this runs the full
Phase 1–3b interpretation pipeline **in one command** and drops everything
into a standard folder layout. It answers: *which epitope does each cluster
target, with which CDR loops, and via what chemistry?*

```bash
massivecluster cluster-analysis \
    --clusters    clusters.parquet \        # needs model_id, cluster, src_cif
    --master      master.parquet \          # per-cluster quality summary (+ score col for picks)
    -o            out/contact_analysis \
    -t            H1335 \
    --paratope-chains D,E --epitope-chains A,B,C \
    --heavy E --light D \
    --anarci-bin  /path/to/.venv-cdr/bin/ANARCI \
    --max-models  25      # optional: subsample per cluster to speed up 2D
```

`--contacts contacts_long.parquet` is **optional**: if omitted, the epitope
contact table is derived from the cluster CIFs (the same pair extraction is
reused for the 2D map, so no double work). So **cluster results + master + CIFs
are enough** — no pre-built contacts table needed.

Output layout:

```
out/contact_analysis/
├── manifest.json                  # entry point: phases, timings, dirs
├── phase1_contact_map/            # epitope-side frequency + enrichment
│   ├── global/epitope_frequency.csv
│   ├── per_cluster/cluster_*/data/{epitope_frequency,enrichment}.csv
│   └── cross_cluster/F_binary_matrix.csv
├── phase2_signatures/             # interpretation
│   ├── overview.md                # cluster → primary epitope / method / pick table
│   ├── cluster_*_fingerprint.md
│   └── signatures.json
├── phase3a_1d_viz/                # epitope-side plots
│   ├── cluster_*_top_epitope_bars.png
│   ├── cluster_vs_epitope_{frequency,enrichment}_heatmap.png
│   ├── enrichment_grid.png
│   ├── physchem_distribution.png
│   └── by_method/cluster_*_freq_by_method.png   (af3=blue, bt2=gold, of3=green, pt2=red)
└── phase3b_2d_viz/                # the "real" CDR × epitope contact map
    ├── data/
    │   ├── pair_contacts.parquet            # every (model, paratope-res, epitope-res) pair
    │   ├── anarci_cdr_map.csv               # residue → IMGT CDR loop
    │   └── cluster_*_top_pairs.csv          # ranked contacts + chemistry + geometry
    └── figures/
        ├── cluster_*_cdr_epitope_2d.png         # y=CDR residue (ANARCI loop strip), x=epitope
        └── channel_distribution_per_cluster.png  # pair-level physchem channels
```

Phase toggles: `--no-1d`, `--no-2d`, `--no-freq-by-method`. Phase 3b (pair-level
2D map) needs `--paratope-chains`/`--epitope-chains` and a `src_cif` column;
`--heavy`/`--light` enable ANARCI IMGT CDR-loop labelling on the y-axis.

> **model_id key**: `--clusters`, `--contacts`, and `--master` must share the
> same `model_id` convention (the join key). If your `contacts_long` was built
> with a path-style id (`variant/method/seed_N/sample_M`) but clustering uses a
> filename id, harmonize them first or the join yields zero overlap.

H1335 validation (EC3 cluster): top pair `E.105 (CDR-H3) → A.405 (gO core)` at
0.77 frequency — matches the known CS2it1p2 hotspot; 6/6 clusters map to the
correct epitope chain.

### How to read the outputs

Recommended reading order for a colleague new to a target:

1. **`phase2_signatures/overview.md`** — start here. One table: each cluster →
   primary epitope chain, dominant generator, top score, recommended pick.
   Gives the "what binds where, by which method" map in 30 seconds.
2. **`phase3a_1d_viz/cluster_vs_epitope_enrichment_heatmap.png`** — which epitope
   residues are *specific* to each cluster (red = enriched vs the global baseline,
   not just frequently contacted).
3. **`phase3b_2d_viz/figures/cluster_*_cdr_epitope_2d.png`** — the real contact
   map for the cluster you care about: y = paratope residue grouped by CDR loop
   (colored strip: H1/H2/H3 orange-red, L1/L2/L3 blues), x = epitope residue,
   color = contact frequency. Bright cells = consensus contacts.
4. **`phase3b_2d_viz/data/cluster_*_top_pairs.csv`** — the quantitative backup:
   per contact, the CDR loop, chemistry channel, frequency, and
   `mean_min_dist ± std`. Sort by freq, then look for tight (low mean, low std)
   chemically-specific pairs (salt bridge / cation-π / aromatic) — those are the
   hotspots worth trusting.

**Frequency vs enrichment** (the common confusion): *frequency* = how often the
cluster touches a residue; *enrichment* = how much **more** than the all-model
baseline (= specificity). A residue can be high-frequency everywhere (big sticky
surface) yet not specific. See [docs/advanced.md §10](notes/advanced.md) for the
full metric reference and what this can / cannot claim about affinity.

### Per-mode usage

| Mode | paratope / epitope | CDR strip | Typical call |
|---|---|---|---|
| **Ab–Ag** (e.g. H1335 Fab + gHgLgO) | `--paratope-chains` = Fab H+L, `--epitope-chains` = antigen | yes — pass `--heavy`/`--light` + `--anarci-bin` | `--paratope-chains D,E --epitope-chains A,B,C --heavy E --light D` |
| **Generic PPI** (chain1 ↔ chain2) | `--paratope-chains` = one partner, `--epitope-chains` = the other | no — omit `--heavy`/`--light` (rows fall back to chain grouping, no ANARCI) | `--paratope-chains A --epitope-chains B` |

In PPI mode the 2D map still works — y-axis is just "partner-1 residue by chain"
instead of CDR loops, and the physicochemical channel decomposition is unchanged.

---

## XGBDimerScorer bridge (filter + xgb scoring)

If `analysis/XGBDimerScorer_tool/` is on `PYTHONPATH` (or sits as a sibling
under `analysis/`), 6 extra subcommands are auto-registered. They operate
on the `master.parquet` / `master_with_xgb.parquet` output of XGBDimerScorer
and let you toggle individual sanity filters from the same CLI.

| Subcommand | Purpose | Default |
|---|---|---|
| `filter-atom-integrity` | flag models with any atom > 10 Å from its residue CA (pt2 terminal-O bug etc.) | OFF — opt-in |
| `filter-clash` | 1.7 Å heavy-atom clash audit (disulfide-aware); writes `n_clash_17` + `n_inter_clash_17`; `--max-inter N` writes filtered subset | OFF — opt-in |
| `filter-cdr-clean` | Ab CDR-clean cascade (`frac_FR ≤ 30%`, `frac_H3 ≥ 20%`, iface ≥ 100, ...) | OFF — opt-in |
| `filter-strict` | full STRICT cascade (CDR-clean + atom + no-of3 + gO core ≤ 6 Å) | OFF — opt-in |
| `filter-regional-rmsd` | anchor-aligned probe RMSD vs reference (e.g. 7LBE.A→gH, gO regional) | OFF — opt-in |
| `join-xgb` | join `xgb_v8` or `xgb_ppi_v3` scores onto an existing clustering parquet | — |

Each is non-destructive: input is a parquet, output is a new parquet
(`-o <path>`). Original clustering output is never modified — chain them in
any order.

```bash
# Example: take an XGBDimer master + run STRICT + then join xgb scores
massivecluster filter-strict \
    --master /path/to/out/master_with_xgb.parquet \
    --mode ab \
    -o /path/to/out/filter_strict.parquet

massivecluster filter-clash \
    --master /path/to/out/filter_strict.parquet \
    --max-inter 2 \
    -o /path/to/out/filter_strict_clash.parquet

# Or join xgb scores onto existing clustering output (no filter).
# Clustering now writes a .parquet sibling next to every *_clustering.csv, so
# point at it directly. The join key auto-resolves (clustering `structure`/`model`
# ← xgb `model_id`) — no --on needed; pass a .csv and it's read + converted too.
massivecluster join-xgb \
    --clustering results/<sto>/<variant>/tm/results/<target>_tm_clustering.parquet \
    --xgb-scores /path/to/out/parquet/xgb_scores_A-B.parquet \
    -o results/<sto>/<variant>/tm/results/clustering_with_xgb.parquet
```

Filter concept reference: see [`XGBDimerScorer_tool/docs/FILTERS.md`](../interface/FILTERS.md) for what each
filter does, why it exists, and the H1335 case study that motivated each.

**On/off model**: every filter is **off by default**. You compose what you
need step-by-step. The XGBDimer side keeps the full feature table; the
MassiveClustering side adds filter columns / writes filtered subsets without
ever dropping rows from upstream files.

---

## License & citations

CASP17 internal tool. If used externally please cite:
- TM-align (Zhang & Skolnick)
- Pierce TCR docking angle (Pierce & Weng)
- Leiden algorithm (Traag, Waltman, van Eck)
- ANARCI (Dunbar & Deane)
