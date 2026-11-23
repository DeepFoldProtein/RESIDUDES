# bench/

Reproducible speed / consistency benchmark of the three models on the three
backends, driven through the normal `lantern-engine submit` path (so the numbers are
what a user gets, not a micro-benchmark).

| file | role |
|---|---|
| `targets.json` | the input matrix: prepared AF3-dialect inputs on the team's shared filesystem, chosen for an even spread of token counts (36 → 2004), plus models, backends, GPU tag and seed |
| `run.py` | submits one Slurm job per model × backend × target into `runs/bench/<target>_<sto>/<model>-<backend>[-<suffix>]/`; skips combinations that already have a `run.json`; records job ids in `runs/bench/jobs.json`. `--suffix v2` = the optimised code series, `--suffix warm` / `v2-warm` = second JAX pass with a hot compile cache, `--site` = alternate site file (e.g. a different `mem_fraction` or the `core/jaxenv` environment) |
| `collect.py` | reads every run (`.lantern-engine/run.json`, `timing.json`, top sample `summary.json`) → `docs/benchmark/results.csv` (one row per run, with `series` v1/v2 and `variant` warm/mem85), calls `plot.py`, writes `docs/benchmark.md`. Figures show the current state (v2 where it exists, warm where it exists); the table lists every run |
| `plot.py` | the figures, matplotlib + seaborn, dependencies declared inline (PEP 723): `uv run bench/plot.py`. Forward time, wall time, speed-up over PyTorch, v1→v2 gain, structural agreement, pLDDT |
| `agreement.py` | TM-score / RMSD of every run's top model against the PyTorch top model (tmtools via `uv run`) → `docs/benchmark/agreement.csv` |
| `micro/` | GPU micro-probes used during optimisation: `triattn_impl.py` (which attention implementation a shape gets, peak memory), `pairformer_hlo.py` (what the compiled JAX pairformer contains: cuDNN calls, largest tensors, temp buffers); run with `lantern-engine run --jax -- bench/micro/<probe>.py` |
| `svgchart.py` | dependency-free SVG fallback used by `collect.py` only when `uv` is not available |

```sh
bench/run.py                                    # v1 matrix (or resubmit what is missing)
bench/run.py --suffix v2                        # the same matrix on the optimised code
bench/run.py --backends jax --suffix v2-warm    # second JAX pass once the cold v2 runs are done
bench/collect.py                                # rebuild docs/benchmark.md + figures from whatever has finished
uv run bench/agreement.py                       # backend-vs-PyTorch TM-scores
LANTERN_ENGINE_PROFILE=1 lantern-engine predict …         # per-stage timing → <out>/profile.json (Boltz-2, OpenFold 3)
```

Settings are the CASP17 production defaults (1 seed, 5 diffusion samples,
10 recycles). `forward_s` is the model writer's `timing.json` (forward pass
only); `wall_s` is the whole process as measured by lantern-engine. For JAX the
first run at a shape compiles inside the forward pass; the figures use the
warm run where one exists and the table lists both. What changed between v1
and v2 is listed in `docs/benchmark_details.md`.
