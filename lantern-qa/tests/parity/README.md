# Parity tests

These scripts check that lantern-qa reproduces the original CASP17 tools. Each one runs the
original tool and lantern-qa on the same inputs, on the same node, in one Slurm job, and
compares every number. They need the DeepFold cluster: the originals live under
`/gpfs/deepfold/casp/casp17/`.

| Script | Compares | Reference |
|---|---|---|
| `interface_t1417.py OUT` | interface scorer: (A) scoring the reference `master.parquet`, all 2,460 models; (B) features base/clash/cpu/dfire/mpnn on a stratified subset | `targets/T1417/xgb_trimer/` plus `XGBDimerScorer_tool` re-run now |
| `cluster_t1417.py OUT` | `lantern-qa cluster` vs `run_clustering.py` (TM method): matrix, structure order, cluster labels; test target links seeds 1–20 of af3/of3/pt2 (300 models) | MassiveClustering_tool re-run now |
| `visualize_t1417.sh WORK` | `lantern-qa visualize` vs the af3-viz container on a test target built from T1417 seed links | the container re-run now |

```sh
sbatch -p normal --gres=gpu:1 -c 8 -t 2:00:00 \
  --wrap ".venv/bin/python tests/parity/interface_t1417.py /store/deepfold/lantern-qa-parity/interface_t1417"
```

Each comparison reports the largest absolute difference per column:
- **`new_vs_orig`**: lantern-qa vs the original tool run in the same job. The port is faithful if this is within tolerance, or no larger than the run-to-run noise below.
- **`new_vs_new2`**: lantern-qa run twice. This is the noise of the GPU stages: DFIRE runs float32 reductions, and ProteinMPNN's forward pass draws a random decoding order.
- **`orig_vs_ref`**: the original tool today vs the file it wrote in CASP17. A difference here means the original tool's models or code changed after the reference was written; it is not a porting error.

Results go to `OUT/parity_summary.json`. `docs/figures/make_figures.py --refresh` turns them into `docs/figures/parity.svg`.

Latest run (2026-11-21, ada, RTX 4090): every check is identical except `base` (2e-5, float32 rounding) and `mpnn`
(0.051, the same as lantern-qa against itself, 0.054). The `orig_vs_ref` drift of the interface score (0.108) comes from
the original tool's PPI model having been replaced after 2026-07-08.
