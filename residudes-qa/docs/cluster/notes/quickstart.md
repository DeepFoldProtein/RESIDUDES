# Quickstart

5-minute tutorial. Cluster a CASP target pool, look at outputs, pick submission models.

---

## 0. Install (once)

```bash
cd /path/to/CASP17_analysis
bash analysis/setup.sh                  # base venv (.venv/)
bash analysis/setup.sh --with-cdr       # + ANARCI venv (.venv-cdr/) for CDR detection

# verify
analysis/.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## 1. Cluster a target — one command

```bash
bash analysis/MassiveClustering_tool/run_clustering.sh \
     /path/to/targets/H1311 \
     --sto A1B1C1D1E1 --variant af3_msa_8yiv \
     --method tm \
     --output-base /path/to/playground/H1311_clustering/af3_msa_8yiv \
     --log-dir /path/to/playground/H1311_clustering/_slurm_logs
```

What happens:
1. Discovers `H1311/results/A1B1C1D1E1/af3_msa_8yiv/{af3,bt2,of3,pt2}/seed_*/sample_*/model.cif`
2. Verifies every model has the same chain composition
3. Submits a Slurm job: GPU pairwise TM-score matrix → comprehensive clustering → network analysis → consensus + medoid PDB export
4. Output under `<output-base>/A1B1C1D1E1/af3_msa_8yiv/tm/`

> **First time?** Pick `--method tm` to start. It's the safest baseline. Switch to `dramsd / itm / anchored-rmsd` later.

---

## 2. Inspect outputs

After ~15-45 minutes per Slurm job (depending on N):

```
<output-base>/A1B1C1D1E1/af3_msa_8yiv/tm/
├── matrices/
│   ├── tm_matrix.npy                    # (N, N) similarity in [0,1]
│   └── structure_names.npy              # row/col label per model
├── ranking.csv                          # per-model confidence join from summary.json
├── _pdb_view/                           # CIF → PDB copies — removed at end of run; pass `--keep-pdb-view` to keep
└── results/
    ├── tm_clustering.csv                # cluster, structure, max_membership, umap1/2, ...
    ├── tm_cluster_stats.csv             # per-cluster size, medoid, intra-distance
    ├── tm_comprehensive.png             # 6-panel summary
    ├── tm_k_diagnostic.png              # K-selection (silhouette+BIC+plateau)
    ├── network/
    │   ├── summary.json                 # Q_leiden, n_communities, hub_indices
    │   ├── summary_louvain.json         # alt algorithm cross-check
    │   ├── transition.csv               # inter-cluster mean distance
    │   ├── report.png                   # 4-panel network diagnostics
    │   └── pretty_community_graph.png   # communities-as-nodes view
    ├── representatives/
    │   ├── *_representatives.csv        # cluster → medoid mapping
    │   └── cluster*_medoid.pdb          # medoid PDB per cluster
    └── consensus/                       # per-cluster mean coords + RMSF
```

Quick look at clusters:

```bash
cd <output-base>/A1B1C1D1E1/af3_msa_8yiv/tm/results
head -20 tm_cluster_stats.csv            # top clusters by size
open tm_comprehensive.png                 # silhouette / cluster-size distribution
open network/pretty_community_graph.png   # community structure
```

---

## 3. Multiple metrics in one go

```bash
bash run_clustering.sh /path/to/targets/H1311 --sto A1B1C1D1E1 \
     --method tm \
     --output-base $OUT --log-dir $LOG

bash run_clustering.sh /path/to/targets/H1311 --sto A1B1C1D1E1 \
     --method drmsd \
     --output-base $OUT --log-dir $LOG

bash run_clustering.sh /path/to/targets/H1311 --sto A1B1C1D1E1 \
     --method anchored-rmsd \
     --anchor-chains CDE --probe-spec AB:cdr \
     --output-base $OUT --log-dir $LOG
```

Each method writes to its own subfolder under `<output-base>`. Compare modularity across methods:

```bash
for d in $OUT/A1B1C1D1E1/af3_msa_8yiv/*/results/network/summary.json; do
    method=$(basename $(dirname $(dirname $d)))
    Q=$(python -c "import json; print(round(json.load(open('$d'))['modularity'], 3))")
    echo "$method  Q=$Q"
done
```

Higher Q (modularity) → cluster signal stronger for that metric.

---

## 4. Pick submission models

After clustering finishes, the playground submission_picker provides a starting point:

```bash
python playground/gunwoong_analysis/H1311_clustering/submission_picker.py
# → playground/.../submission/picks.csv + model_1..5.cif + picks_3d.png + picks_overlay.png
```

See [advanced.md](advanced.md) for picking strategy details.

---

## 5. CDR × epitope contact-map analysis (Ab–Ag targets)

Once you have **cluster assignments + an epitope contact table**, interpret each
cluster's binding mode in one command — *which epitope, which CDR loops, what
chemistry, how tight*:

```bash
analysis/.venv/bin/python -m massiveclustering.cli cluster-analysis \
    --clusters  clusters.parquet \         # model_id, cluster, src_cif
    --master    master_with_xgb.parquet \  # per-cluster quality (+ score col for picks)
    -o          out/contact_analysis -t H1335 \
    --paratope-chains D,E --epitope-chains A,B,C \
    --heavy E --light D \
    --anarci-bin /gpfs/deepfold/casp/casp17/analysis/.venv-cdr/bin/ANARCI \
    --max-models 25      # optional subsample per cluster (speeds up the 2D pass)
```

> `--contacts contacts_long.parquet` is **optional** — omit it and the epitope
> contact table is built from the cluster CIFs (reused for the 2D map). Cluster
> results + master + CIFs are enough.

Produces, under `out/contact_analysis/` (entry point: `manifest.json`):

| Folder | Key outputs | Answers |
|---|---|---|
| `phase1_contact_map/` | per-cluster `epitope_frequency.csv`, `enrichment.csv` | which epitope residues, how specific |
| `phase2_signatures/` | `overview.md`, `cluster_*_fingerprint.md` | cluster → epitope chain / method / recommended pick |
| `phase3a_1d_viz/` | epitope bars, cluster×epitope heatmaps, `enrichment_grid.png`, `by_method/` (af3=blue, bt2=gold, of3=green, pt2=red) | epitope-side overview |
| `phase3b_2d_viz/` | `cluster_*_cdr_epitope_2d.png`, `channel_distribution_per_cluster.png`, `data/cluster_*_top_pairs.csv` | **the real CDR×epitope contact map** + chemistry + geometric tightness |

The `cluster_*_top_pairs.csv` is the concrete "is the contact real" table:
`paratope, cdr_loop, epitope, channels, freq, n_models, mean_min_dist, std_min_dist`.
Low `mean_min_dist` + low `std` + high `freq` across independent predictors =
a tight, reproducible contact (e.g. H1335 EC3: `E.106 Tyr (H3) → A.425 Arg`,
2.07 ± 0.64 Å, cation-π, in 72% of cluster models).

> Phase toggles: `--no-1d`, `--no-2d`, `--no-freq-by-method`. Phase 3b (the 2D
> map) needs `--paratope-chains`/`--epitope-chains` + a `src_cif` column;
> `--heavy`/`--light` enable ANARCI IMGT CDR-loop labelling.
>
> **Join key**: `--clusters`, `--contacts`, `--master` must share the same
> `model_id` convention or the merge yields zero rows.

---

## 6. Common pitfalls

1. **Slurm log dir must be writable.** If `<target>/cluster_results/_slurm_logs/` is owned by another user, jobs fail with SIGRT19 (Slurm can't create stderr file). Use `--log-dir` to point to your own dir.
2. **GPU memory.** For N > 15,000 with dRMSD, allocator may need free-memory budget detection (already auto-applied as of 2026-05).
3. **Pierce TCR-angle method** needs `.venv-cdr` + biopython-compatible `renumber_tcr.py`. See [troubleshooting.md](troubleshooting.md).
4. **CDR mode** (e.g. `--probe-spec AB:cdr`) requires `setup.sh --with-cdr` — ANARCI must be on PATH for ID assignment.
5. **`cluster-analysis` join key.** `--clusters`, `--contacts`, `--master` must share the same `model_id` convention. If they don't overlap you get `no overlap between contacts_long.model_id and cluster_assignments.model_id` — harmonize the ids first (e.g. rebuild the path-style key from `variant/method/seed/sample`).
6. **`cluster-analysis` ANARCI.** Pass the absolute `--anarci-bin .../.venv-cdr/bin/ANARCI`; the tool auto-prepends its bin dir to PATH so the bundled `hmmscan` is found. Without it ANARCI silently fails and every paratope residue is labelled `framework` (no H1/H2/H3 strip).
