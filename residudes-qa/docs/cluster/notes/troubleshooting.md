# Troubleshooting

Common errors when running MassiveClustering, and fixes.

---

## Slurm jobs fail immediately (~1 sec, exit code `0:53`)

**Symptoms:**
```
$ scontrol show job <jobid>
JobState=FAILED Reason=RaisedSignal:53(Real-time_signal_19)
ExitCode=0:53  RunTime=00:00:01
```

**Cause:** Slurm cannot create the StdOut/StdErr files because the destination directory isn't writable for your user. Default destination is `<target>/cluster_results/_slurm_logs/`, which is often owned by another user (e.g. `deepfold`).

**Fix:** pass `--log-dir <writable-dir>` to `run_clustering.sh`:

```bash
mkdir -p /path/to/playground/H1312_clustering/_slurm_logs
bash run_clustering.sh ... --log-dir /path/to/playground/H1312_clustering/_slurm_logs
```

---

## dRMSD job fails with CUDA OOM

**Symptoms:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.74 GiB.
GPU 0 has a total capacity of 23.52 GiB ...
```

**Cause:** N × n_dist × 4 bytes upload exceeds GPU memory. For N=20,480, n_dist=337k (Cα pairs from 822 residues), the upload is 27.6 GB.

**Fix:** automatic since 2026-05. `drmsd.py` detects free GPU memory via `torch.cuda.mem_get_info()` and switches to chunked-path (per-row tile matmul) when budget is exceeded. No user action needed.

**If OOM still happens:** check whether the GPU is shared with another job (`nvidia-smi` on the node). Slurm may co-schedule via `OverSubscribe=OK`. Resubmit when GPU is free.

### dRMSD process "Killed" (host RAM, not CUDA)
**Symptom:** job dies with `Killed` (OOM-killer), no CUDA traceback, at/before "Computing distance vectors".
**Cause:** the O(L²) condensed distance vector per model is materialised for all N models at once on the **host** (e.g. a 1455-res ×3 spike → L=4365 → ~38 MB/model → ~347 GB for 9k models).
**Fix:** automatic since 2026-06. `drmsd.py` projects host-RAM need and, if it exceeds `DRMSD_HOST_BUDGET_GB` (default 32), uniformly subsamples Cα by a stride (logged). Override via that env var. Subsampling is positional → dRMSD stays comparable (just coarser), fine for global-shape clustering of very large structures.

---

## tcr-angle method: all NaN values returned

**Symptoms:** `tcr-angle_matrix_features_deg.npy` has `(nan, nan)` for all rows, or many specific predictor / seed range.

**Causes & fixes:**

### A. Pierce/ANARCI subprocess fails with `No module named 'anarci'`

Pierce wrapper calls `subprocess.run(["python", renumber_tcr.py, ...])`, which resolves bare `python` via PATH. If PATH leads to a system python without anarci, you get this error.

Fix already applied in `_pierce_worker.py` — it monkey-patches `subprocess.run` to use `.venv-cdr/bin/python` (which has anarci installed). If you're calling Pierce directly (outside the worker), apply the same patch.

### B. `renumber_tcr.py` fails with `Cannot change id from … to …`

Biopython 1.81 added strict residue-ID and chain-ID uniqueness checks. Re-ordering in `renumber_tcr.py` collides.

**Fix:** use the patched copy at `playground/gunwoong_analysis/_tcr_tools_patched/scripts/renumber_tcr.py`. Override `tcr3d_pierce.RENUMBER_SCRIPT` before calling:

```python
import tcr3d_pierce
tcr3d_pierce.RENUMBER_SCRIPT = '/path/to/playground/_tcr_tools_patched/scripts/renumber_tcr.py'
```

The patch is purely an order-of-operations change (renumber via temp IDs first, then final IDs) — same final output as the original code on biopython < 1.81.

### C. Specific seed range silently missing from CSV

The Pierce-csv worker may have been killed before processing all models. Check input CSV row count by group_dir / predictor / seed range:

```python
import pandas as pd
df = pd.read_csv("h1311_angles_pierce.csv")
print(df.groupby(["group_dir","predictor"]).size())
```

If counts don't match expected (e.g. expected 5120 per predictor in af3_msa_8yiv), run the backfill script:

```bash
python playground/gunwoong_analysis/H1311_clustering/pierce_backfill/run_backfill.py
```

This writes a delta CSV + merged full CSV to `pierce_backfill/`.

---

## Empty `network/` directory or only `summary.json`

**Symptom:** Only `network/summary.json` exists; missing `transition.csv`, `summary_louvain.json`, `report.png`, `pretty_community_graph.png`.

**Cause:** Network step partially failed. Older `run_clustering.py` had a `transition undefined` bug — exception swallowed, only `summary.json` written.

**Fix:** already fixed in `run_clustering.py:1249-1305`. If running on an old clone, pull latest.

For already-completed runs, re-run the network step standalone (see `playground/gunwoong_analysis/H1311_clustering/postprocess.py`).

---

## Method output prefix is mis-labeled

**Symptom:** Running `--method anchored-rmsd` produces `drmsd_clustering.csv`, `drmsd_cluster_stats.csv`, etc. inside `anchored-rmsd_anchor-CDE_probe-AB-cdr/results/`.

**Cause:** Downstream `massivecluster comprehensive` CLI uses a label-mapping where anchored-rmsd → drmsd (for distance-vs-similarity conversion routing). Output file prefix accidentally inherits the routing label.

**Status:** known issue. Files are correct, just prefix is wrong. Workaround: rename files post-run, or filter by directory not filename. Proper fix planned.

---

## Pierce CSV: which `group_dir` to use?

Pierce CSV has a `group_dir` column matching the MSA variant (`af3_msa`, `af3_msa_8yiv`, `mmseqs2_gpu`, ...). When clustering on a specific MSA variant, pass `--tcr-angle-group-filter <variant>` to restrict.

If you pool variants, **do not** use `--tcr-angle-group-filter`; the cluster matrix will include all variants present in the CSV.

---

## Homo-oligomer: TM / dRMSD blows up despite identical fold

**Symptom:** Two predictions of the same homo-N-mer look structurally identical
in PyMOL but TM-score is far below 1.0 (e.g. 0.3 on a homo-12-mer) or dRMSD is
huge. Network clusters fragment into N spurious groups.

**Cause:** Predictors emit homo-oligomer chains in different orders / labels —
chain `A` in one model is chain `F` in another. TM-align / dRMSD assume the
chain-label-to-position mapping is fixed, so the metric measures a label
permutation, not a fold difference.

**Fix:** preprocess with `canonicalize-chains` **before** building the matrix.
It greedily relabels chains by minimum-distance match to a reference frame so
the same monomer slot has the same label across the pool.

```bash
massivecluster canonicalize-chains \
    --inputs /path/to/<sto>/<variant>/<runner>/seed_*/sample_*/model.cif \
    --reference <one anchor model> \
    -o /path/to/<sto>/<variant>/<runner>_canon/
```

Then run `run_clustering.sh` against the `_canon` tree. Skip this step for
hetero-multimers (chains already distinguishable by sequence).

---

## Cluster sizes inflated by duplicates

If a model appears twice with different `pdb_key` (e.g. once with and without `__variant` suffix), distance matrix has duplicate rows and clusters look big but redundant.

**Check:**
```python
import numpy as np
names = np.load("structure_names.npy", allow_pickle=True).tolist()
print(len(names), len(set(names)))  # should match
```

If they don't match, deduplicate before clustering.

---

## `gemmi` failure on specific CIF files

Some inference outputs save CIF files with unusual headers (custom data blocks, missing `_entry.id`). `gemmi.read_structure` may emit warnings or fail.

**Detect:**
```bash
analysis/.venv/bin/python -c "import gemmi; s = gemmi.read_structure('/path/to/model.cif'); print(len(s))"
```

If a single model fails, the whole pipeline may abort. Either:
- Exclude that model from the input list, OR
- Save a stripped-down version (extract atoms only)

The shape-check step early in `run_clustering.py` catches mismatched chain counts and reports the bad model.

---

## Slurm partition memory error

```
sbatch: error: Memory specification can not be satisfied
sbatch: error: Batch job submission failed: Requested node configuration is not available
```

The cluster's Slurm config may report `MEMORY=1` per node (placeholder). Adding `#SBATCH --mem=96G` then fails because the requested value exceeds the (fake) reported value.

**Fix:** don't specify `--mem` in the sbatch script. The default unbounded behavior gives you full node memory.

---

## "0 AF3 members in cluster X" when filtering AF3-only

Some COM/TM clusters consist entirely of bt2 / of3 / pt2 models. AF3-only stats are then NaN. This is biologically meaningful (AF3 doesn't predict that pose) — not a bug.

Recommendation: report those clusters separately in submission notes, not drop silently.

---

## `cluster-analysis`: 2D map has no CDR loops (everything is "framework")

**Symptom:** the CDR×epitope heatmap y-axis strip is all one color; every
paratope residue is labelled `framework_H`/`framework_L`; `anarci_cdr_map.csv`
has no H1/H2/H3 rows.

**Cause:** ANARCI ran but its internal `hmmscan` was not on PATH, so it failed
silently and the pipeline fell back to all-framework labels.

**Fix:** pass the **absolute** ANARCI path — the tool then auto-prepends that
bin dir to PATH so the bundled `hmmscan` is found:

```bash
--anarci-bin $RESIDUDES_QA_CDR_BIN/ANARCI
```

---

## `cluster-analysis`: "no overlap between contacts_long.model_id and cluster_assignments.model_id"

**Cause:** `--clusters`, `--contacts`, and `--master` use different `model_id`
conventions (e.g. path-style `variant/method/seed_N/sample_M` vs filename
`method_seedN_sampleM__variant.pdb`). The join key must match.

**Fix:** harmonize the ids first, **or** omit `--contacts` — the contact
table is then derived from the cluster CIFs using one consistent `model_id`
(needs `--paratope-chains`/`--epitope-chains` + a `src_cif` column).

---

## xgb scoring: model file not found on a compute node

**Symptom:** `score` works on the login node but a Slurm job fails to open the
xgb model JSON.

**Cause:** an older default pointed at `/store/gunwoong/...`, which may not be
mounted on every compute node.

**Fix:** the models now ship inside the tool at
`XGBDimerScorer_tool/models/` (on `/data`, mounted everywhere) and are
the default. If you pinned a `/store` path explicitly, drop the override or
point `--model-path` at the bundled copy.

---

## pt2 (Protenix) models flagged by atom-integrity (atoms tens of Å from CA)

**Symptom:** `filter-atom-integrity` flags a large fraction of pt2 models; the
offending atom is a terminal `O`/`OXT`.

**Cause:** known Protenix bug — the C-terminal carboxyl oxygen is placed far from
its residue. The backbone/interface are fine; only that atom is wrong.

**Fix (keep the model):** run `repair-terminal` to snap the oxygen back onto sane
geometry. It writes repaired CIFs + a `repair_manifest.csv` flagging which models
were touched (`effective_cif` points to the repaired copy). Contact-map analysis
already ignores the atom (`sanitize_cutoff`), so only submission-quality CIFs and
geometry-sensitive scorers need the repair.

---

## `_pdb_view/` disappeared — how to keep it

**Behavior:** `<output>/.../method/_pdb_view/` is **auto-removed** at the end of every run, after `results/representatives/` is written. The dir can be many GB on large pools (e.g. 6 GB across 4 runs of 1875/300 models each), so default = delete.

**Keep it:**

```bash
bash run_clustering.sh ... --keep-pdb-view
```

Use when you'll re-run `medoid_faces` / consensus from the same pool without regenerating PDBs. For one-off clusterings, the default cleanup is what you want — `representatives/cluster*_medoid.pdb` already has every medoid you need to inspect.

---

## "Where did my output go?"

`run_clustering.sh` always writes to:
```
<--output-base>/<sto>/<variant>/<method>[_<anchor-tag>]/
```

If you didn't pass `--output-base`, default is `<target_dir>/cluster_results/<sto>/<variant>/<method>/`.

Check with:
```bash
ls $OUTPUT_BASE/<sto>/<variant>/
```
