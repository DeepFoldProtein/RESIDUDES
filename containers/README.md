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
| `residudes` | 0.5 GB | `python:3.12-slim` | the agent, gemmi, the Claude Agent SDK, typst, rsvg, tmux, molterm with USalign | the checkout, the work root, **an agent CLI** |
| `residudes-qa` | 0.7 GB | `python:3.12-slim` | the CPU dependency set from `uv.lock` — no torch | the pool, `data/files` |
| `residudes-msa` | 0.1 GB | `python:3.12-slim` | HMMER, MMseqs2, the engine's `msa` path | **the databases** (`--af3-db`, `mmseqs_db`) |
| `residudes-qa-cu130` | 3.8 GB | `nvidia/cuda:13.0.3-runtime` | the same QA stack plus torch 2.11+cu130 | the same, plus the GPU (`--nv`) |
| `residudes-ef2` | 4.6 GB | `nvidia/cuda:13.0.3-runtime` | ESMFold2: python 3.12, torch 2.12, the Biohub transformers fork | the runner scripts, the HuggingFace weight cache |
| `residudes-engine-trt` | 10 GB | `nvidia/cuda:13.0.3-runtime` | Protenix v2, OpenFold 3, Boltz-2; PyTorch and TensorRT 10.16 | the checkout, checkpoints, and the **TensorRT engines**, which are built per GPU |
| `residudes-engine-jax` | 11.6 GB | `nvidia/cuda:13.0.3-runtime` | AlphaFold 3 with its C++ extension compiled, and the JAX backends | the checkout, **`af3.bin`**, which is licensed per user |
| `residudes-relax` | 0.6 GB | `debian:13-slim` | GROMACS 2024.2 and AmberTools 24.8, from conda-forge | **a Rosetta build**, which is licence-gated, and the relax pipeline |

All eight are published as `ghcr.io/deepfoldprotein/<name>:latest`. A public one is pulled with
no credentials, by either runtime:

```sh
GODEBUG=http2client=0 apptainer pull --disable-cache residudes.sif docker://ghcr.io/deepfoldprotein/residudes:latest   # no root, no daemon
docker pull ghcr.io/deepfoldprotein/residudes:latest
```

`GODEBUG=http2client=0` asks Apptainer's Go registry client for HTTP/1.1: its HTTP/2 transport is known to
drop multi-GB layers partway (`PROTOCOL_ERROR`, `unexpected EOF`), which is what the two engine images
used to hit. `--disable-cache` keeps the layers out of `~/.apptainer/cache`, which on a shared `/home` is
the disk that fills. Neither matters for the small images; both matter for the engines.

One OCI image serves both: Apptainer converts it to a SIF on the way in, which is why nothing is
published twice. `containers/verify.sh` asks the registry about each image and builds locally
whatever it cannot pull.

**No layer is larger than 4.3 GB**, and that is deliberate. The two engine images were one layer
each to begin with — `install.sh` ran as a single build step, giving 9.1 GB for TensorRT and
11.5 GB for JAX against about 3 GB for every other image — and a blob that size did not survive
the link: the pull died partway with `PROTOCOL_ERROR` over HTTP/2 and `unexpected EOF` over
HTTP/1.1, while the 3 GB images pulled cleanly on the same connection.

Splitting the installation did not help, because `uv sync` rebuilds an environment that an
earlier pass left incomplete, and the final layer was the whole 9.9 GB again. (A sync over an
environment that already matches the lock writes 994 B, so it is the incomplete state that
triggers the rebuild.) So the environment is built once in a builder stage and *then* split:
`install.sh split 4` bin-packs site-packages into four buckets of similar size, which the final
stage copies one per layer. The image is byte for byte what `uv sync --frozen` produced, the
total is unchanged, and the largest layer went from 9.1 GB to 4.3 GB.

`verify.sh` still retries a failed pull over HTTP/1.1 and builds locally if that fails too. That
path is now a safety net rather than the expected route.

Sizes refer to the Apptainer `.sif` files produced on the reference hardware. Each recipe consists
of three files: `install.sh`, which contains the installation steps and the rationale for them, and
a `Dockerfile` and an Apptainer `.def` file that both execute it. The two container formats
therefore cannot diverge.

No image carries an agent CLI. Claude Code, Codex and the others authenticate as a person and
are licensed to one, so an image cannot ship a usable copy: `residudes setup` picks from what the
machine has and says how to install one. molterm is in the agent image, because showing a
structure is part of the job and that one is ours to ship.

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
GODEBUG=http2client=0 apptainer pull --disable-cache residudes.sif docker://ghcr.io/deepfoldprotein/residudes:latest
containers/verify.sh --pull                      # pull the core images, then verify them

# or build locally from the definition file
apptainer build residudes.sif containers/residudes/residudes.def

# on a workstation with a Docker daemon
docker build -t residudes:dev containers/residudes
```

**Publishing happens from a machine with Docker**, not from CI
([`publish.sh`](publish.sh)). A GHCR package belongs to the repository that first pushed it, and
giving another repository's Actions write access to that package is a web-UI setting with no REST
or GraphQL equivalent — both are 404 while a plain read of the same package returns 200. Package
visibility is the same: public is a one-time setting someone makes by hand, whatever the
repository's visibility.

```sh
echo "$GHCR_TOKEN" | docker login ghcr.io -u <you> --password-stdin   # PAT with write:packages
containers/publish.sh                              # the five large images
containers/publish.sh residudes residudes-qa residudes-msa           # and the small ones
```

A push to `main` **builds and checks** every image it can through
[`.github/workflows/containers.yml`](../.github/workflows/containers.yml): the agent image
regenerates the analysis of the H1388 example, the QA image runs its 92 tests, and the MSA image
executes each binary it provides. CI's job is to catch a broken image, not to ship it.

### The five large images

They are not built in CI. A hosted runner has too little disk: with the preinstalled toolchains
deleted (41 GB free), with docker and buildkit moved to `/mnt` (41 GB), and with the wheel cache
disabled, the TensorRT and JAX builds still ended in `no space left on device`. They are built and
pushed with Docker from a machine that has the disk:

The script applies each recipe's build context (the engine images copy the vendored model
sources; the GPU QA image needs `residudes-qa`'s lockfile), runs the same check CI runs for the
small images, and removes each image locally afterwards, since they are 4–12 GB apiece.
`residudes-engine-jax` is skipped unless `PUBLISH_JAX=1`: it contains the vendored AlphaFold 3
code, which is CC BY-NC-SA 4.0.

Until they are in the registry, `containers/verify.sh` builds them locally with Apptainer, which
needs no daemon and no root.

On the reference cluster, `podman` and `buildah` are installed but lack an `/etc/subuid` entry, so
rootless OCI builds fail; `apptainer build` works without privileges.

## Usage

The source checkout is bind-mounted rather than copied into the image, so a single image can serve
any branch:

```sh
apptainer run -B $PWD/residudes:/opt/residudes -B /store/work:/store/work residudes.sif doctor
apptainer run --nv -B /store/work residudes-qa-cu130.sif interface run-ppi-v4 ...
apptainer run -B /data/databases/af3:/db residudes-msa.sif \
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
