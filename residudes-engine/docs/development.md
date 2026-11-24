# Development

## Layout

```
residudes-engine/
├── bin/residudes-engine                 launcher: finds Python ≥ 3.11, sets PYTHONPATH=src, runs `python -m residudes_engine`
├── pyproject.toml              name residudes-engine; zero runtime dependencies; console script `residudes-engine`
├── sites/                      machine descriptions (TOML)
├── scripts/slurm/              residudes-engine.sbatch (generic job: predict, engines build, …) · setup_venv.sbatch
├── src/residudes_engine/                the CLI (stdlib only): registry · config · cli · setup · verify · runners/ · schema/
├── tests/test_plan.py          GPU-free unit tests
├── docs/                       user-facing docs (English)
├── runs/                       scratch outputs, uv cache, setup logs (gitignored)
└── core/                       the runtime: model code, TensorRT/JAX backends, adapters, lockfile (core/README.md)
```

## Rules

1. **stdlib only** inside `src/residudes_engine`. Heavy imports happen in the
   subprocess, from `core/.venv`. The CLI must run on login nodes and pods
   without the runtime.
2. **No machine paths in code.** They go in `sites/*.toml`. A grep for
   `/gpfs`, `casp17`, `/data/` over `src/` must return nothing. The only
   absolute paths allowed are the generic libstdc++ search defaults in
   `config.py` (`/opt/conda/lib`, `/usr/lib/gcc/...`, `/usr/lib64`).
3. **Tags are the only names.** `pt2/of3/bt2`, `auto/trt/jax/torch`,
   `rtx4090/rtx6000ada/b200/h100/a100`, `vanilla/cuequiv`. Long names and
   on-disk engine prefixes live in `registry.py` and docs only.
4. **Reproduce, don't reinvent.** A runner's command line stays identical to
   the known-good invocation it was verified against; differences go in
   `post()`.
5. **Never lose a prediction over bookkeeping.** `post()` and the manifest
   run in `try/except`; a flattening bug must not delete or hide outputs.
   Conversely, a model that exits 0 without writing samples is a failure
   (`predict` exits 3).
6. **`core/` is edited deliberately.** It is the runtime, not the place for
   CLI logic. Changes there are small, commented, and tested with a real
   prediction. Internal names in `core/` are historical; the CLI maps the
   user-facing ones (`RESIDUDES_ENGINE_JAX_*`) onto them in `config.runtime_env`.

## Running the tests

```sh
cd residudes-engine
python3 -m unittest discover -s tests -v       # ~0.1 s, no GPU, no venv
residudes-engine doctor                             # needs core/.venv; imports torch/TRT/JAX, loads the plugin
```

`tests/test_plan.py` builds a throwaway site (empty venv skeleton, empty
checkpoints, fake engines) and checks every family × backend plan, output
flattening, the verifier, the CLI parser, `--backend auto`, engine
discovery and the site writer. Add a case whenever a flag, env var or
layout rule changes.

## End-to-end check

Reference target: `T1430` (36 residues, MSA and four templates), with a
reference prediction for every model from the original pipeline.

```sh
residudes-engine submit --model pt2 --input <T1430 input.json> --out runs/T1430/pt2 --seeds 1
residudes-engine verify runs/T1430/pt2/seed_1 --ref <reference seed directory for pt2>
```

Top pLDDT should match the reference within ±0.5 (BF16 / kernel
nondeterminism); see [`verification.md`](verification.md).

## Adding things

- **Model version**: `registry.MODELS[tag] = ModelSpec(...)` with
  `checkpoint`, `engine_prefix`, known `engines`, and `weights` (how
  `setup checkpoints` fetches it: `url` + `size` + `sha256`, or a publisher's
  downloader run inside the venv).
- **Family**: `runners/<family>.py` exposing `FAMILY`, `plan()`, `post()`;
  register in `runners/__init__.py`; add the build script mapping in
  `setup.py` if TensorRT engines apply.
- **Site**: `residudes-engine setup site …`, or copy `sites/local.example.toml`.
- **CLI flag**: add to `_add_predict_options`, `Request`, the runner that
  consumes it, and `_passthrough` so `submit` forwards it.

## Updating the runtime (`core/`)

`core/` is plain files in this repo. Upstream model fixes are applied as
ordinary commits touching `core/…`; keep them small and record the reason
in the commit message. The pinned Python stack is `core/pyproject.toml` +
`core/uv.lock`; after changing either, run `residudes-engine setup venv` and the
end-to-end check before committing. Two known debts: the project name inside
`core/pyproject.toml` is historical (it is tied to the lock and changes at the
next re-lock), and the `jax-cu13` extra needs the CUDA plugin fix folded into
the lock (today `setup venv` patches it after sync).

## Versioning

`residudes_engine.__version__` follows semver. Bump the minor version when a CLI
flag, site key or output convention changes; record the TensorRT version in
the release notes (it decides whether existing engines still load).
`run.json` records the repository commit of every run.

