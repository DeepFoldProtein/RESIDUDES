# Advanced usage

For deeper CASP17 idioms, distance-metric tuning, anchor/probe design, sub-clustering, submission picking.

---

## 1. Distance metric choice — when to use what

| Metric | What it measures | When to prefer |
|---|---|---|
| **`tm`** (TM-score) | Global fold similarity. CA-only, length-normalized. | Default. Fold-level differences. |
| **`itm`** (interface TM-score) | TM-score on interface residues only (auto-detected) | Multimer interface focus. Same fold but different docking. |
| **`drmsd`** | RMSE of all CA-CA distance pairs. No alignment. | Local conformation differences (loop motion, hinge). |
| **`anchored-rmsd`** | RMSD on probe chain(s), Kabsch on anchor chain(s) | "Where does X sit relative to Y?" — TCR on pMHC, Ab on antigen. |
| **`anchored-tm`** | TM-score on probe chain(s), Kabsch on anchor | Like anchored-rmsd, but TM-normalized (length-tolerant). |
| **`com`** | Pairwise L2 of probe-chain centers-of-mass | Rigid-body position only. Coarsest, cleanest "binding face". |
| **`tcr-angle`** | Pierce crossing/incident angle (sin/cos circular) | TCR-pMHC specific. Canonical docking angle space. |
| **`iface-jaccard-residue`** | 1 − Jaccard of epitope (or paratope) residue *sets* | Contact pattern, alignment-free. "Same residues touched?" |
| **`iface-jaccard-contact`** | 1 − Jaccard of (paratope,epitope) residue *pairs* | Stricter — same specific pairings, not just same patch. |
| **`iface-chamfer`** | Symmetric Chamfer distance of interface atom clouds | Interface *shape*, coordinate-based (anchor-align first). |
| **`iface-wasserstein`** | Sinkhorn-regularised W₂ between interface clouds | Like Chamfer but optimal-transport (smoother). |

The four `iface-*` metrics are built with `massivecluster iface-matrix` (writes a
distance `.npy` you feed to `comprehensive`/`run`/`network` with `--metric drmsd`):

```bash
# contact pattern (set-based, from a contacts table — fast)
massivecluster iface-matrix --metric iface-jaccard-residue \
    --contacts contacts_long.parquet -o out/iface
# interface shape (coord-based, from CIFs — submit heavy runs via sbatch)
massivecluster iface-matrix --metric iface-chamfer \
    --clusters clusters.parquet --ab-chains DE --ag-chains ABC -o out/iface
```

They are **complementary** to TM/dRMSD: global metrics see fold/pose, interface
metrics see *binding mode*. A single metric can be misleading — cluster on a few
and compare (see §4 robustness).

For the jaccard metrics you don't even need a contacts table — point at the
CIFs and they're extracted on the fly:

```bash
massivecluster iface-matrix --metric iface-jaccard-residue \
    --clusters clusters.parquet --ab-chains DE --ag-chains ABC -o out/iface
```

**Which interface metric?** (H1335, 6 known binding-mode clusters; silhouette
computed against the known labels — higher = better recovery)

| Metric | within | between | norm. separation | silhouette vs known |
|---|---|---|---|---|
| `iface-chamfer` (anchor-aligned) | 553 | 10519 | **+0.95** | **+0.74** |
| `iface-jaccard-residue` | 0.74 | 0.99 | +0.25 | +0.23 |
| `iface-jaccard-contact` | 0.92 | 1.00 | +0.08 | +0.08 |
| `iface-chamfer` (raw, no align) | 4077 | 4047 | −0.01 | −0.26 |

→ **Anchor-aligned `iface-chamfer` is the strongest binding-mode metric.** After
Kabsch-aligning every model on the antigen (`ag_chains`, the default anchor), the
paratope point-cloud *position* directly encodes where the Ab binds — distinct
epitopes land in spatially separate clouds. `iface-jaccard-residue` is a solid,
cheap (set-based, no CIF parse) alternative; `-contact` (exact pairs) is too
strict for clustering — use it only to confirm two modes are genuinely distinct.

> **Critical — coord metrics MUST be anchor-aligned.** `iface-chamfer`/
> `-wasserstein` now Kabsch-align on `--anchor-chains` (default = `--ag-chains`)
> **automatically**. The raw (unaligned) distance is meaningless for binding mode
> — H1335 raw chamfer silhouette was **−0.26** (worse than random) vs **+0.74**
> aligned, because raw distance is dominated by global pose, not interface shape.
> Only pass `--anchor-chains ""` (disable) if your structures are already in a
> common frame.

### CDR / framework subsetting

```bash
--anchor-chains AB --anchor-mode framework  # Kabsch on framework only
--probe-chains AB --probe-mode cdr          # measure CDR only
# OR equivalently:
--anchor-spec 'AB:framework' --probe-spec 'AB:cdr'

# Mixed: anchor CDE all + extras
--anchor-spec 'C:framework+D:framework+E'
```

Requires `setup.sh --with-cdr`.

---

## 2. Anchor / probe design patterns

### TCR-pMHC (5 chain: A=TCRα B=TCRβ C=MHC D=β2m E=peptide)

```bash
# Standard: how does TCR sit on pMHC?
--anchor-chains CDE --probe-chains AB

# CDR-focused: rigid body fixed via MHC+β2m+peptide, only CDR loops vary
--anchor-chains CDE --probe-spec AB:cdr

# Peptide pose given pMHC scaffold:
--anchor-chains CD --probe-chains E
```

### Nanobody/Ab–antigen (3 chain: A,B antigen, C nanobody)

```bash
# Where is the nanobody sitting?
--anchor-chains AB --probe-chains C

# Nanobody CDR focused (framework should be identical across models):
--anchor-chains AB --probe-spec C:cdr
```

### Generic dimer

```bash
--anchor-chains A --probe-chains B  # B's pose relative to A
```

---

## 3. K (number of clusters) selection

Default `--cluster-method fuzzy` uses **adaptive K** with 4 signals:
1. Penalized silhouette (favors well-separated clusters)
2. BIC (penalizes overfitting)
3. Cluster size balance (avoids singleton dumps)
4. Plateau detection (stops when no improvement)

Outputs:
- `{target}_{metric}_k_diagnostic.png` — visualizes all 4 signals
- `{target}_{metric}_k_scores.csv` — full sweep table

If you see "WARNING: Silhouette still ascending at k_max" in the log:

```bash
# Default k_max is 50; raise it
bash run_clustering.sh ... --k-max 80
```

For `leiden` clustering, K is auto-determined by community detection — no K selection step.

---

## 4. Network analysis output interpretation

Per-method `results/network/`:

| File | What |
|---|---|
| `summary.json` | n_nodes/edges, density, modularity, top-10 community sizes, hub_indices |
| `summary_<other>.json` | Same metrics from the other community algorithm (Leiden ↔ Louvain) — cross-validation |
| `transition.csv` | (K, K) inter-cluster mean distance — block matrix shows which clusters are nearest |
| `report.png` | 4-panel: degree dist, community assignment heatmap, hubs, modularity |
| `pretty_community_graph.png` | community-as-node graph, color by dominant predictor (af3/bt2/of3/pt2) |

**Modularity (Q) interpretation:**
- Q > 0.7 : strong community signal — clustering reliable
- 0.3 < Q < 0.7 : moderate — usable but ambiguous
- Q < 0.3 : weak — clusters likely arbitrary slicing

**KNN-k choice:** auto = `2·log₂(N)`. Override with `--knn-k N` if you want to override (rarely needed).

### Robustness: don't trust a single run

One Leiden run at one resolution can be unstable. Three flags quantify/repair this:

| Flag | What it does | Output |
|---|---|---|
| `--resolution-scan` | Leiden CPM over a log-spaced resolution range | `*_resolution_scan.csv` (resolution → n_communities, modularity) + suggested resolution |
| `--stability` | bootstrap: subsample `--sample-frac` of nodes `--n-boot` times, re-detect, compare to the full partition by **Adjusted Rand Index**, accumulate a **co-association** matrix | `*_bootstrap_ari.csv`, `*_node_stability.csv`, `*_coassociation.npy` |
| `--consensus-communities` | run Leiden at many resolutions, fuse the partitions via co-association (pairs together in ≥50% stay together) | `*_consensus_communities.csv`, `*_consensus_coassoc.npy` |

**Reading it:** mean bootstrap ARI → 1.0 means the communities are reproducible
(trustworthy); low or high-variance ARI means the single-run partition is fragile
— prefer the consensus partition or a coarser resolution. H1335 interface-jaccard
network: mean ARI **0.95 ± 0.04** (robust). Pair this with clustering on a second
metric (e.g. dRMSD) and compare — agreement across metrics + bootstraps is the
real confidence signal.

---

## 5. Per-cluster face PNGs (medoid inspection)

`representatives/cluster_NN_face.png` shows the medoid as 2-panel (side xz + top xy) Cα trace:
- Anchor chains drawn **faded gray** (background scaffold)
- Probe chains drawn **bold colored** (focus)
- Peptide (chain E, if present) overlaid as red dots

Generated by the playground `per_rep_faces.py` script after main pipeline completes. Per-method grid PNG (`<method>_medoid_faces.png`) shows all medoids in one image.

---

## 6. Sub-clustering (Stage 2)

After Stage 1 COM clustering identifies binding faces, **Stage 2** drmsd inside each COM cluster reveals local conformation variants.

Run:
```bash
python playground/gunwoong_analysis/H1312_clustering/stage2_com_to_drmsd.py
python playground/gunwoong_analysis/H1312_clustering/stage2_visualize.py
```

Output `com_anchor-AB_probe-C/results/stage2_drmsd/`:
- `cluster_NN/members.csv` — (structure, com_cluster, sub_cluster)
- `cluster_NN/sub_medoids.csv` — per-sub-cluster medoid + size + iface
- `cluster_NN/sub_faces.png` — grid of sub-medoid traces
- `cluster_NN/sub_3d.png` — 3D scatter of sub-clusters within parent
- `stage2_summary.csv` — per-COM-cluster K_sub, Q, mean_intra_drmsd
- `stage2_overview.png` — N×M panel grid

When **Q_sub < 0.1** for a COM cluster → that cluster is mostly an outlier basket (no internal structure). Avoid using it as a submission representative.

---

## 7. Submission picker — strategy

Combine three axes:
1. **(A) Diversity** — different clusters / binding faces. Picks all 5 from one cluster = wasted slot.
2. **(B) Tightness** — small intra-cluster distance, high Q. Tight cluster = consistent prediction.
3. **(C) Quality** — high xgb_v8_pred (DockQ regression), high ipTM, high ranking_score.

Score:
```
combined = 0.45·xgb_v8 + 0.25·xgb_ppi_v3 + 0.15·ranking_score + 0.10·iptm    # H1312
```

For H1311 (with TCR-angle):
```
combined = 0.40·xgb_v8 + 0.25·xgb_v8_rank + 0.20·ranking_score + 0.15·iptm
+ zone filter (canonical 30-80° / 8YIV 15-32° / drop flipped > 110°)
```

Use scripts:
- `H1311_clustering/submission_picker.py`
- `H1312_clustering/submission_picker.py`

Output `submission/`:
- `picks.csv` — 5 rows with rationale
- `model_1.cif`..`model_5.cif` — CIF copies (rename for submission)
- `picks_overlay.png` — 5 medoids side-by-side
- `picks_3d.png` (H1312) / `picks_angle.png` (H1311) — pose space visualization

---

## 8. AF3-only restricted analysis

If you want to rank candidates **using only AF3 predictions** (e.g. trusted predictor only):

```bash
python playground/gunwoong_analysis/H1311_clustering/  # AF3-only candidate table
```

Outputs `af3_only/`:
- `af3_candidates_ranked.csv` — 5120 af3 models, sorted by xgb_v8_pred
- `af3_best_per_cluster.csv` — best AF3 representative per tcr-angle cluster
- `cluster_features_af3.csv` — AF3-only per-cluster {size_frac, ipTM mean/max/center mz, xgb mean/max/center mz}

Some clusters may have **zero AF3 members** (e.g. H1311 C8, C20, C22, C28). Means non-AF3 generators produced those poses but AF3 doesn't.

---

## 9. cluster_features tables

Per-cluster aggregate statistics — used by submission picker and downstream analysis.

Columns (H1312 example):
```
cluster_index, cluster_size, cluster_size_normalized (= size / total, fraction),
xgb_combined_{mean, max, center}    # mean of xgb_v8 + xgb_ppi_v3
xgb_abag_{mean, max, center}        # xgb_v8 alone
xgb_ppi_{mean, max, center}         # xgb_ppi_v3 alone
iptm_{mean, max, center}
ptm_{mean, max, center}
center_pdb, center_predictor        # the xgb-combined-best model in this cluster
```

Columns for H1311 (AF3-only):
```
cluster_index, cluster_size_af3, cluster_size_af3_frac,
AF3_ipTM_{mean, max, center}, AF3_ipTM_*_mz,    # z-score (matches summary_table _mz)
xgb_pdockq, xgb_pdockq_{mean, max}, xgb_pdockq_*_mz,
AF3_center_pdb, AF3_center_crossing, AF3_center_incident
```

`_mz` = z-score: `(x - mean) / std`. `_frac` = fraction of total. `_normalized` is min-max [0,1] (avoid — bias toward extremes).

---

## 10. Contact-map analysis metrics (`cluster-analysis`)

What the per-cluster CDR×epitope analysis actually computes, and what it can
and cannot claim.

### Frequency vs enrichment

| Metric | Definition | Reads as |
|---|---|---|
| **frequency** `F_c[res]` | fraction of cluster-`c` models where the epitope residue is in contact (≤ cutoff Å) | "how often this cluster touches the residue" |
| **enrichment ΔF** | `F_cluster − F_global` (also log-odds) | "how much *more* than the all-model baseline" → cluster **specificity** |

A residue can be high-frequency everywhere (large/sticky surface) — high
frequency alone is not specific. Enrichment removes the baseline, isolating each
cluster's signature contacts. Saved in `per_cluster/cluster_*/data/enrichment.csv`
(`enrichment_delta`, `enrichment_log_odds`).

### Pair-level geometry + chemistry (Phase 3b)

`cluster_*_top_pairs.csv` columns:

| Column | Meaning |
|---|---|
| `paratope`, `cdr_loop` | Ab residue + ANARCI IMGT loop (H1/H2/H3/L1/L2/L3/framework) |
| `epitope` | antigen residue |
| `channels` | physicochemical class(es): hydrophobic / salt_bridge / polar_hbond / aromatic / cation_pi |
| `freq`, `n_models` | contact reproducibility across the cluster ensemble |
| `mean_min_dist`, `std_min_dist` | mean ± std of per-model min heavy-atom distance (Å) |

**Geometric stability:** low `mean_min_dist` + low `std` + high `freq` across
independent predictors (af3/bt2/of3/pt2) = a tight, reproducible contact. This
is a *consensus / stability* signal, **not** a binding free energy.

### What this can and cannot tell you

- **Can:** the binding mode (which CDR ↔ which epitope), its chemistry, and how
  reproducible/tight it is across predictors — an interface-confidence signal.
- **Cannot:** a quantitative affinity (ΔG / Kd). Contact frequency is prediction
  reproducibility, not energetics.
- **Affinity-like ranking** is the scorer's job (xgb_v8 / pDockQ / ipTM /
  dfire). Use them together: contact map = *what / where / chemistry*; scorers =
  *how good / how strong*.

---

## 11. Reproducing H1311 / H1312 CASP17 submissions

See:
- H1311 submission report
- H1312 submission report

Each report has the exact `run_clustering.sh` invocation + post-processing scripts + the final 5 picks.
