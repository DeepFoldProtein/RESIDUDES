# Provenance

Everything under `src/lantern_qa/` except `paths.py`, `data.py`, `doctor.py`, `cli.py`,
`__init__.py` and `__main__.py` was copied on 2026-11-21 from the DeepFold CASP17 tree on
the ada cluster. The copies were then changed only as listed below.

| Package path | Copied from | Notes |
|---|---|---|
| `interface/` | `/gpfs/deepfold/casp/casp17/analysis/XGBDimerScorer_tool/xgbdimer/` (last modified 2026-06-15) | renamed from `xgbdimer` |
| `cluster/` | `.../analysis/MassiveClustering_tool/massiveclustering/` (2026-06-16) | renamed from `massiveclustering` |
| `cluster/run_clustering.py` | `.../analysis/MassiveClustering_tool/run_clustering.py` (2026-06-20) | |
| `ema/scorers/` | `.../analysis/EMA/scorers/{aadars_per_cdr, aadars_v3_scorer, bbdep_chi_energy, bbdep_chi_spline_v2, dfire_v2_scorer, hbond_energy, hbond_network, mpnn_iface, ndrd_scorer, proteinmpnn_score}.py` | the subset the interface features import |
| `mpnn/run/` | `.../analysis/MassiveClustering_tool/proteinmpnn/run/` without `model_weights/` | ProteinMPNN code plus the team's `mpnn_utils.py`, `run_mpnn.py`, `generate_json.py` |
| `visualize/` | `/gpfs/deepfold/casp/casp17/visualization/{analysis/*, analyze.py → target.py, analyze_msa.py → msa.py}` | |
| `scripts/interface/` | `XGBDimerScorer_tool/{run_scoring.sh, run_all_worker.sh, shard_ppi.sh, shard_worker.sh, ppi_merge.py, run_scoring.py, sbatch_templates/}` | |
| `scripts/cluster/` | `MassiveClustering_tool/{run_clustering.sh, run_clustering.sbatch}` | |
| `tests/cluster/` | `MassiveClustering_tool/tests/` | |
| `docs/` | the tools' READMEs and notes | |

## Changes to copied code

1. **Package names.** All imports were rewritten mechanically:
   - `xgbdimer` → `lantern_qa.interface`
   - `massiveclustering` → `lantern_qa.cluster`
   - `EMA.scorers` → `lantern_qa.ema.scorers`
   - `analysis` (visualization) → `lantern_qa.visualize`

   User-facing names in help text, banners and Slurm job names were changed the same way. Output file and column names were not changed.
2. **Data paths.** Model and potential paths no longer point into the tool directories. They resolve through `lantern_qa.paths`, in four places:
   - `interface/score/xgb_{ab,ppi}.py`: model directory.
   - `ema/scorers/ndrd_scorer.py`: the NDRD cache sits next to `$EMA_NDRD_PATH`.
   - `ema/scorers/proteinmpnn_score.py`: the weight directory comes from `$EMA_PROTEINMPNN_WEIGHTS`, the code directory from the package.
   - `cluster/cli.py`: ProteinMPNN directory.

   The scorers' existing environment-variable lookups are unchanged; `lantern_qa.paths.configure_env()` fills them.
3. **Hard-coded `/gpfs/...` paths removed.**
   - `EMA_PATH` defaults in `interface/features/{cpu,dfire,mpnn,aadars,ppi_v4}.py` are gone.
   - The `sys.path` insertion of `XGBDimerScorer_tool` in `cluster/xgb_filters.py` is gone.
   - The `/viz` container path in `visualize/` is gone.
   - The optional TCR tools in `cluster/core/gpu/{tcr_angle,_pierce_worker}.py` now read `LANTERN_QA_CDR_BIN`, `LANTERN_QA_TCR_TOOLS` and `LANTERN_QA_TCR_RENUMBER`, falling back to the team paths.
4. **Shell wrappers.** They source `scripts/env.sh`, which resolves the venv python, `LQA_ROOT` and `PYTHONPATH`, instead of `analysis/.venv`. Slurm jobs receive `LQA_ROOT` via `--export`.
5. **Visualization wrappers.** `scripts/visualize/*.sh` call `lantern-qa visualize[-msa]` from the uv venv instead of the af3-viz Apptainer container.

## Environment

- **Pins:** dependencies are pinned to `/gpfs/deepfold/casp/casp17/analysis/.venv` as frozen on 2026-11-21 (`envs/casp17-analysis-venv.freeze.txt`).
- **networkx:** the only deliberate difference, 2.2 → 3.4.2 (see README).
- **Visualization:** the af3-viz container ran numpy 2.4.4. Here it runs on the QA pin, numpy 1.26.4.
