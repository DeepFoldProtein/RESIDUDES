# Site files

A site file says where things live on one machine or cluster. It is the
only place machine-specific paths are allowed; code never hardcodes them.
`residudes-engine setup site --checkpoints DIR --engines DIR` writes one from
`sites/local.example.toml`; `sites/ada.toml` and `sites/b200pod.toml` are
the team's clusters.

## Selection

1. `--site NAME` or `--site /path/file.toml`
2. `$RESIDUDES_ENGINE_SITE_FILE` (path) or `$RESIDUDES_ENGINE_SITE` (name or path)
3. hostname matched against each site's `hosts` patterns (fnmatch)
4. `sites/local.toml` if present
5. otherwise an error listing the available sites

`residudes-engine submit` exports the chosen file to every Slurm job, so the job
never re-detects the site.

## `[site]` keys

| key | required | meaning |
|---|---|---|
| `name` | no (defaults to file stem) | shown in `run.json` and logs |
| `hosts` | no | fnmatch patterns for auto-selection, e.g. `["ada", "ada-*"]` |
| `scheduler` | no | `slurm` · `pueue` · `none`; `submit` requires `slurm` |
| `core` | no (default `{residudes_engine_root}/core`) | the runtime tree whose entry points are executed |
| `venv` | **yes** | runtime venv, normally `{residudes_engine_root}/core/.venv` built by `residudes-engine setup venv` (torch + TensorRT + cuEquivariance + boltz + openfold3 + JAX with CUDA plugin) |
| `venv_jax` | for `af3` | the JAX 0.11 environment (`bash core/jaxenv/build.sh`): AlphaFold 3 on every backend, and optionally the tokamax fused kernels for the other models' JAX path |
| `af3_weights` | no (default `<checkpoints>/af3`) | directory holding AlphaFold 3's `af3.bin` (licensed per user by Google DeepMind, placed by hand) |
| `checkpoints` | **yes** | directory for model weights: `protenix-v2.pt`, `of3-p2-155k.pt`, `boltz2/boltz2_conf.ckpt` + `boltz2/mols/` (filled by `residudes-engine setup checkpoints`), `af3/af3.bin` unless `af3_weights` points elsewhere |
| `engines` | **yes** | root of TensorRT engines, `<gpu>/<model>_<kernel>_<maxN>/…` (filled by `residudes-engine engines build`) |
| `libstdcxx` | no | extra candidates for a libstdc++ carrying `GLIBCXX_3.4.32`; system, GCC 12/13 and conda paths are tried after these |
| `kalign` | no | kalign binary for Protenix template realignment |
| `cache_dir` | no (default `~/.cache/residudes-engine`) | Triton / cuEquivariance / JAX compilation caches |
| `boltz_cache` | no (default `<checkpoints>/boltz2`) | `BOLTZ_CACHE` (mols, CCD) |

Placeholders `{residudes_engine_root}`, `{home}`, `{site_dir}` and `$ENV_VARS` are
expanded. Every key can be overridden per invocation with
`RESIDUDES_ENGINE_<KEY>` (e.g. `RESIDUDES_ENGINE_VENV=/scratch/venv`, `RESIDUDES_ENGINE_LIBSTDCXX=a.so:b.so`).

## `[slurm]`

| key | default | used by |
|---|---|---|
| `partition` | `normal` | `sbatch -p` |
| `gres` | `gpu:1` | `sbatch --gres` |
| `cpus` | `4` | `sbatch -c` |
| `time` | `02:00:00` | `sbatch -t` (override with `--time`) |
| `[slurm.constraint]` | none | `--gpu <tag>` → `sbatch -C <value>`, e.g. `rtx4090 = "RTX4090"` |

## `[jax]`

| key | default | meaning |
|---|---|---|
| `dtype` | `bf16` | `--jax_dtype` |
| `mem_fraction` | `0.65` | `XLA_PYTHON_CLIENT_MEM_FRACTION` |
| `allocator` | unset | `XLA_PYTHON_CLIENT_ALLOCATOR` (`vmm` recommended for N ≥ 2k on B200) |
| `venv` | all `venv` | per model tag, which venv runs the JAX path: `{ pt2 = "venv_jax", of3 = "venv", bt2 = "venv", af3 = "venv_jax" }` (`af3` always needs `venv_jax`; its torch and TensorRT backends run there too) |
| `[jax.env]` | none | knobs applied on the JAX path (below) |
| `[jax.env.<tag>]` | none | per-model overrides |

JAX knobs (the CLI exports each under its user-facing name and the name
the runtime reads):

| knob | default | effect |
|---|---|---|
| `RESIDUDES_ENGINE_JAX_TOKAMAX` | `0` | `1` enables tokamax fused attention (needs the tokamax venv) |
| `RESIDUDES_ENGINE_JAX_CUEQ` | `0` | `1` runs triangle kernels through cuEquivariance-JAX |
| `RESIDUDES_ENGINE_JAX_PAIRFORMER_SCAN`, `RESIDUDES_ENGINE_JAX_TOKEN_TRANSFORMER_SCAN` | `1` | `lax.scan` over blocks (memory) |
| `RESIDUDES_ENGINE_JAX_ATTENTION_IMPL` | `auto` | `triton` / `mosaic` / `cudnn` / `xla` |

## Engines directory layout

```
<engines>/
├── rtx4090/     protenix_v2_vanilla  openfold3_p2_vanilla  boltz2_vanilla_896
├── rtx6000ada/  protenix_v2_cuequiv_4096  openfold3_p2_cuequiv_3072  boltz2_cuequiv_5120
└── b200/        protenix_v2_vanilla_5120  openfold3_p2_vanilla_4096  boltz2_vanilla_5120
    └── <model>_<kernel>_<maxN>/
        ├── pairformer_engine/trt/{rank0.engine, config.json}
        ├── token_transformer_engine/trt/…
        └── confidence_pairformer_engine/trt/…   (bt2 only)
```

The GPU tag comes from `nvidia-smi` compute capability (`8.9 → rtx4090`,
`12.0 → rtx6000ada`, `10.0 → b200`, `9.0 → h100`, `8.0 → a100`; see
`registry.GPUS`). `residudes-engine engines build` writes
`<engines>/<gpu>/<prefix>_<kernel>_<maxN>` and `predict` picks the known
directory for the GPU or, failing that, the largest locally built one.
Engines are bound to the SM they were built on and to the TensorRT
major.minor in the venv. `max_seq_len` is a build-time ceiling on tokens:
above it the TensorRT path errors out, use `--backend torch` or `jax`, or
build a larger engine.

## Examples

`sites/ada.toml` is a Slurm cluster: code and venv from this checkout, weights
and engines on the shared filesystem. `sites/b200pod.toml` is a pueue pod with
the tokamax venv. `sites/local.example.toml` is the template.

Check a site with `residudes-engine doctor --site <name>`.
