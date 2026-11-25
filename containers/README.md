# Containers

The software is distributed as one image per task, since the requirements of the tasks are
disjoint: the agent needs only the Python standard library and an API key; the QA stack is a CPU
scientific environment; MSA search requires two C programs and 260 GB of sequence databases; and a
predictor requires CUDA, TensorRT and per-user licensed weights.

Material that cannot be redistributed is excluded from the images and bind-mounted at run time:
AlphaFold 3 parameters, a Rosetta build, and the sequence databases. Where an upstream project
publishes an official image, that image is used as the base layer rather than rebuilt here.

| image | size | base | holds | mounted at run time |
|---|---|---|---|---|
| `residudes` | 0.5 GB | `python:3.12-slim` | the agent, gemmi, the Claude Agent SDK and CLI, typst, rsvg | the checkout, the work root |
| `residudes-qa` | 0.7 GB | `python:3.12-slim` | the CPU dependency set from `uv.lock` — no torch | the pool, `data/files` |
| `residudes-msa` | 0.1 GB | `python:3.12-slim` | HMMER, MMseqs2, the engine's `msa` path | **the databases** (`--af3-db`, `mmseqs_db`) |
| `residudes-qa-cu130` | 3.8 GB | `nvidia/cuda:13.0.3-runtime` | the same QA stack plus torch 2.11+cu130 | the same, plus the GPU (`--nv`) |
| `residudes-ef2` | 4.6 GB | `nvidia/cuda:13.0.3-runtime` | ESMFold2: python 3.12, torch 2.12, the Biohub transformers fork | the runner scripts, the HuggingFace weight cache |
| `residudes-engine-trt` | 10 GB | `nvidia/cuda:13.0.3-runtime` | Protenix v2, OpenFold 3, Boltz-2; PyTorch and TensorRT 10.16 | the checkout, checkpoints, and the **TensorRT engines**, which are built per GPU |
| `residudes-engine-jax` | 11.6 GB | `nvidia/cuda:13.0.3-runtime` | AlphaFold 3 with its C++ extension compiled, and the JAX backends | the checkout, **`af3.bin`**, which is licensed per user |
| `residudes-relax` | 0.6 GB | `debian:13-slim` | GROMACS 2024.2 and AmberTools 24.8, from conda-forge | **a Rosetta build**, which is licence-gated, and the relax pipeline |

Sizes refer to the Apptainer `.sif` files produced on the reference hardware. Each recipe consists
of three files: `install.sh`, which contains the installation steps and the rationale for them, and
a `Dockerfile` and an Apptainer `.def` file that both execute it. The two container formats
therefore cannot diverge.

One image carries additional licence restrictions. `residudes-engine-jax` includes the vendored
AlphaFold 3 source code (CC BY-NC-SA 4.0, **non-commercial**); continuous integration builds it on
a release tag but publishes it only on explicit request. No image contains `af3.bin`, a Rosetta
build, or the sequence databases.

## Building the images

The `Dockerfile` defines the build. Apptainer either converts the resulting OCI image or executes
the definition file directly; both invoke the same `install.sh`:

`containers/verify.sh` with no flag queries the registry for each image before deciding: an image
that is publicly readable is pulled, a private one is pulled if the host holds credentials, and
anything else is built locally. The three routes can also be taken directly:

```sh
# on a cluster: pull the image published by CI (no daemon or root privileges required)
apptainer build residudes.sif docker://ghcr.io/deepfoldprotein/residudes:latest
containers/verify.sh --pull                      # pull the core images, then verify them

# or build locally from the definition file
apptainer build residudes.sif containers/residudes/residudes.def

# on a workstation with a Docker daemon
docker build -t residudes:dev containers/residudes
```

A push to `main` publishes the core images to `ghcr.io/deepfoldprotein/<name>:latest`, together with
`:sha-…` tags and `:vX.Y` for release tags, via
[`.github/workflows/containers.yml`](../.github/workflows/containers.yml). **Every image is verified
before publication**: the agent image regenerates the analysis of the H1388 example, the QA image
runs its 92 tests, and the MSA image executes each binary it provides. An image that fails its
check is not published.

On the reference cluster, `podman` and `buildah` are installed but lack an `/etc/subuid` entry, so
rootless OCI builds fail; `apptainer build` works without privileges.

## Usage

The source checkout is bind-mounted rather than copied into the image, so a single image can serve
any branch:

```sh
apptainer run -B $PWD/residudes:/opt/residudes -B /store/work:/store/work residudes.sif doctor
apptainer run --nv -B /store/work residudes-qa-cu130.sif interface run-ppi-v4 ...
apptainer run -B /gpfs/database/af3:/db residudes-msa.sif \
    msa --fasta q.fasta --source af3 --af3-db /db
```

Job schedulers remain outside the containers. `sbatch` and `pueue` belong to the host, and the
agent submits to them from the host; a container provides the environment in which a submitted job
executes.

## Verification

`containers/verify.sh` performs an end-to-end check of an installation: the presence of the images,
the behaviour of the commands inside them, and the sequence databases they require (see
[`../docs/setup.md`](../docs/setup.md)).

```sh
containers/verify.sh                                     # the three core images
containers/verify.sh --gpu                               # also the four GPU images, each against the card
containers/verify.sh --all                               # additionally residudes-relax
containers/verify.sh --af3-db /path/to/af3 --deep        # the nine databases, plus one jackhmmer search
```

Each check prints one line; the script exits non-zero if any check fails. On the reference hardware
(two RTX 4090) the complete run reports 26 passed, 0 failed, 1 skipped, the skipped item being the
host virtual environments, which are unnecessary when the engine images are present.
