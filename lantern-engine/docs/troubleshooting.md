# Troubleshooting

Start with `lantern-engine doctor`: it resolves the site, detects the GPU,
finds a usable libstdc++, imports torch / TensorRT / JAX inside the venv,
checks the JAX CUDA plugin and confirms the `IdentitySZ` TensorRT plugin
loads. Every ✗ line maps to a section here. After a failed run, read
`<out>/inference.log` and `<out>/.lantern-engine/run.json` (the exact command and
environment used).

## `setup venv` fails

- `Disk quota exceeded` while building → uv's cache sits on a full or
  quota-limited filesystem. lantern-engine pins `UV_CACHE_DIR` to
  `runs/_setup/uv-cache`; point `LANTERN_ENGINE_UV_CACHE_DIR` somewhere with
  ~25 GB free.
- `Readme file does not exist` or other build-backend errors → the
  checkout is incomplete; `git status` should be clean and `core/README.md`
  present.
- `Failed to query Python interpreter` → no Python 3.11; let uv fetch one
  (`uv python install 3.11`) or set `LANTERN_ENGINE_PYTHON`.
- The script needs network to PyPI; on clusters run `setup venv --slurm` on
  a node that has it. First build downloads ≈ 12 GB; later builds reuse the
  cache and take ~2 minutes.

## `setup checkpoints` fails

- `unknown url type: https` → the Python running the CLI was built without
  the `ssl` module (pyenv builds often are). lantern-engine downloads with `curl`
  when it is present (`doctor` shows `cli py … curl found`); install curl or
  run the CLI with a Python that has ssl (`LANTERN_ENGINE_PYTHON=core/.venv/bin/python`).
- `AccessDenied` from the Protenix publisher URL → expected from some
  networks; the next source in the list is a mirror whose file is verified
  against the pinned size and sha256 before it is accepted.
- OpenFold 3 asks `Download … (yes/no)` and hangs or raises `EOFError` →
  only when its downloader is called by hand; lantern-engine passes
  `skip_confirmation=True`.
- `No module named 'boltz.main'` → a Python started with `core/` as its
  working directory sees the backend stub `core/boltz/` instead of the
  installed package; lantern-engine runs downloaders from the checkpoints
  directory for this reason.

## `GLIBCXX_3.4.32 not found` / `IdentitySZ plugin NOT FOUND`

The prebuilt C++ TensorRT plugin was compiled with GCC 12 and needs a newer
libstdc++ than Ubuntu 22.04 / RHEL 9 ship. lantern-engine searches
`[site].libstdcxx`, then `/opt/conda/lib`, GCC 13/12 paths and the system
lib, and preloads the first one containing the symbol. Fix:
`apt install g++-13` (or point `libstdcxx` at any conda `libstdc++.so.6`
≥ GCC 12) and re-run `doctor`. Protenix v2 at seq_len > 1024 cannot even
build engines without this plugin.

## `cannot pick TRT engines` / `no TRT engines for … on …`

No engine directory for this GPU class. Either build one
(`lantern-engine engines build --model pt2 --max-seqlen 1024`, 15–60 min on one
GPU), pass `--engines-dir`, or let `--backend auto` fall back to `torch`.
For an unknown GPU (`unknown compute_cap`) add the compute capability to
`registry.GPUS` first.

## Hard error when `n_token > max_seq_len`

The TensorRT path has no PyTorch fallback above the engine's build-time
ceiling. Escalation ladder: rebuild with `-- --tri-chunk 16` or `8` →
`-- --no-trimul-hp` (pt2, halves tri-mul activation memory) → fewer
`--diffusion-samples` → `--backend jax` (memory linear in N) → `--backend
torch`. On 24 GB cards OF3 also needs `OF3_PER_SAMPLE_TOKEN_CUTOFF=0`;
lantern-engine sets it automatically for `rtx4090`.

## CUDA OOM on RTX 4090 at modest sizes

TensorRT allocates its execution context through PyTorch's caching
allocator; a fragmented pool fails a single large chunk. lantern-engine sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for every run. If it
still fails, lower `--diffusion-samples`.

## OF3: `OF3_PAIRING_MODE must be set`

OpenFold 3 needs to know whether the paired MSA is `precomputed` (CASP17
`af3_msa` inputs) or should be paired `online`. lantern-engine reads
`--pairing-mode`, then `$OF3_PAIRING_MODE`, then a `pairing_mode.txt`
sidecar next to the input or in its parent directory, and falls back to
`precomputed`.

## OF3: errors mentioning Unix sockets or `/tmp/of3-…`

OF3's data pipeline uses `multiprocessing.Manager` sockets whose path must
stay under 108 bytes. lantern-engine creates a short per-run `TMPDIR`
(`/tmp/of3-XXXX`) and removes it afterwards; `/tmp` must be writable on
compute nodes.

## `predict` exits 3: "runner exited 0 but produced no sample_<i>/"

The model's own runner swallowed a failure (OpenFold 3 does this when JAX
cannot initialise CUDA). Read `<out>/inference.log`; the usual cause is the
next item.

## JAX path: `Backend 'cuda' failed to initialize` / "CUDA-enabled jaxlib is not installed"

The venv has `jax` but no CUDA PJRT plugin: the lockfile's JAX extra
resolves to a jax version without one. `lantern-engine setup venv` installs
`jax[cuda12]==<locked version>` after the sync and `doctor` reports
`cuda plugin OK`; if you built the venv by hand, run
`uv pip install --python core/.venv/bin/python "jax[cuda12]==0.5.3"`.

## JAX path: slow first run, or OOM at large N

The first run compiles; the cache lives under `<cache_dir>/jax_<family>`
and is reused (measured: Protenix v2 35 → 5 s at 36 tokens, OpenFold 3 and
Boltz-2 3–4× faster from ~200 tokens on the second run; see
[`benchmark.md`](benchmark.md)). For N ≥ 2k on B200 set `allocator = "vmm"`
in `[jax]`. On consumer cards keep `LANTERN_ENGINE_JAX_TOKAMAX = "0"` unless the
tokamax venv is present; tokamax pays off above N ≈ 200–256 and is required
around N ≈ 2.9k.

`mem_fraction` (default 0.65) caps how much of the card XLA may take; the
rest is left to the PyTorch side of the hot-swap. With the default cap a
96 GB RTX PRO 6000 runs OpenFold 3 and Boltz-2 JAX to 1627 tokens and
Protenix v2 to 1627; 2004 tokens needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`
in the environment of `lantern-engine predict`/`submit` (Protenix v2 completes,
OpenFold 3 still does not), or `mem_fraction` in the site file when every
GPU the site serves has headroom for the PyTorch side at that fraction
(24 GB cards do not).

If a JAX run is far slower than PyTorch *and* its log shows
`cuDNN attention rejected … float32 … → XLA fallback`, something upstream
handed the pair stack fp32 activations (JAX promotes bf16 × fp32 to fp32,
cuDNN flash attention then refuses the fp32 q/k/v and XLA materialises a
4·H·N³-byte logits tensor). The wrappers cast activations to the compute
dtype and keep masks in it (`LANTERN_ENGINE_JAX_INPUT_DTYPE=fp32` restores the old
behaviour); a new module that multiplies by a mask must do the same.

## Boltz: `summary.json.seed` is 1 for every seed

The Boltz unified writer reads the seed from the `BT2_SEED` environment
variable, not from `--seed`. lantern-engine sets both. Results produced by older
Slurm scripts may carry `seed: 1` in `summary.json` even though the
directory name and the sampling seed are correct.

## Boltz: CCD / `mols` not found

`BOLTZ_CACHE` defaults to `<checkpoints>/boltz2`, which must contain
`mols/` and the checkpoint (`lantern-engine setup checkpoints` fetches both).
Override with `boltz_cache` in the site file.

## `sbatch: LANTERN_ENGINE_ROOT must be exported`

Slurm copies batch scripts to a spool directory, so `lantern-engine.sbatch`
cannot locate the checkout. `lantern-engine submit` and `engines build --slurm`
export `LANTERN_ENGINE_ROOT` and `LANTERN_ENGINE_SITE_FILE` for you; when calling `sbatch`
directly add `--export=ALL,LANTERN_ENGINE_ROOT=/path/to/lantern-engine`.

## `protenix` imports the wrong package

`core/protenix/`, `core/openfold3/` and `core/boltz/` (the per-family
backend code) shadow the upstream packages when `core/` is first on
`sys.path`. The entry points handle this themselves (they `chdir`/re-order
`sys.path`); `doctor` imports from `core/models/protenix` to reproduce that.
Do not add `core/` to `PYTHONPATH` ahead of `core/models/*` by hand.
