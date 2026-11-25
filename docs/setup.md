# Setup

The software comprises four components with different hardware requirements. The **agent** requires
an API key and a CPU; the **QA stack** a CPU and a few gigabytes of storage; **MSA search** two C
programs and 630 GB of sequence databases; and the **predictors** a CUDA-capable GPU. Only the
components that will be used need to be installed. Each section below states the purpose of a
component and how to verify its installation.

The recommended installation route is the container images. Each component is provided as an image;
material that cannot be redistributed, namely the AlphaFold 3 parameters, a Rosetta build and the
sequence databases, is bind-mounted at run time
([`containers/README.md`](../containers/README.md)).

```sh
git clone https://github.com/DeepFoldProtein/RESIDUDES
cd RESIDUDES
containers/verify.sh                    # obtain the images, then check the installation
```

With no flag, the script asks the registry about each image at its real address and acts on the
answer: a publicly readable image is pulled, a private one is pulled if credentials for the
registry are present, and otherwise the image is built locally with `apptainer build`, which
needs no root. `--pull` and `--build` force one route.

Eight images, split so a login node never pulls a GPU stack. The four small ones are published
on every push to `main`; the large ones on a release tag.

| image | size | what it is for |
|---|---|---|
| `residudes` | 0.5 GB | the agent: its CLI, the Claude Code SDK, gemmi, typst |
| `residudes-qa` | 0.7 GB | the QA stack on CPU — scoring, clustering, plots |
| `residudes-msa` | 0.1 GB | jackhmmer, hmmer, kalign, MMseqs2 |
| `residudes-qa-cu130` | 3.8 GB | the QA stack with torch 2.11+cu130 |
| `residudes-ef2` | 4.6 GB | ESMFold2: its own python 3.12, torch, the Biohub transformers fork |
| `residudes-engine-trt` | 10 GB | Protenix v2, OpenFold 3, Boltz-2 on PyTorch and TensorRT |
| `residudes-engine-jax` | 11.6 GB | AlphaFold 3 and the JAX backends, with AF3's C++ extension built |
| `residudes-relax` | 0.6 GB | GROMACS and AmberTools |

`residudes-engine-jax` is **not** published automatically: it carries the vendored AlphaFold 3
code, which is CC BY-NC-SA 4.0, so pushing it is a deliberate act. Build it locally, or ask.

`verify.sh` is the end of every section here: it exits non-zero and names what is wrong. By
default it checks the three small images; `--gpu` adds the four GPU images and runs each against
the card, `--all` adds relax, and `--af3-db /path --deep` checks the databases and runs one real
jackhmmer search.

## 1. The agent

```sh
apptainer build images/residudes.sif containers/residudes/residudes.def
apptainer run -B $PWD/residudes:/opt/residudes -B /data/work:/data/work images/residudes.sif doctor
```

The image carries python, gemmi, the Claude Agent SDK, the Claude Code CLI (node 22) and typst. The
checkout is mounted rather than copied, so one image serves any branch.

Write `residudes/residudes.toml` once (`residudes setup` offers to):

```toml
[paths]
work_root = "/data/work"                 # dossiers live in <work_root>/targets/<T>/
targets_root = "/data/casp/targets"      # an existing residudes-engine layout, or empty
residudes_engine = "/path/to/RESIDUDES/residudes-engine"
residudes_qa = "/path/to/RESIDUDES/residudes-qa"
```

**Checked by:** `verify.sh` runs `residudes report` on the H1388 example inside the image and
compares the result with the `analysis.md` that ships with it — the figures, the clustering and the
numbers all have to come out the same.

## 2. The QA stack

```sh
apptainer build images/residudes-qa.sif containers/residudes-qa/residudes-qa.def
apptainer exec -B $PWD/residudes-qa:/opt/residudes-qa images/residudes-qa.sif \
    python -m pytest -q /opt/residudes-qa/tests          # 92 tests, about two minutes
```

That image is the **CPU** one: its dependencies come from `residudes-qa/uv.lock`, where torch is an
opt-in extra, so clustering, the interface features and the plots work and the GPU scorers do not.
For those, build with the `cu130` extra on an NVIDIA base and run with `--nv`.

Data files (`residudes-qa/data/files`, 60 of them) are fetched by `residudes-qa data fetch` and
mounted with the checkout.

## 3. MSA search and its databases

The tools are small and the databases are not: the image is 68 MB, the AlphaFold 3 database set is
**630 GB unzipped** (252 GB to download).

```sh
apptainer build images/residudes-msa.sif containers/residudes-msa/residudes-msa.def
bash residudes-engine/core/models/alphafold3/fetch_databases.sh /data/af3    # hours
```

Then tell the engine where they are, in the site file:

```toml
[msa]
af3_db_dir = "/data/af3"            # the nine files AF3's pipeline expects
mmseqs_db  = "/data/colabfold"      # optional: the ColabFold databases for --source mmseqs2
```

**Checked by:** `residudes-engine doctor` lists every database by the name AF3 looks for, with its
size, and fails the ones that are missing or far below their released size — a truncated download
otherwise surfaces hours later inside jackhmmer. `--deep` runs one real `jackhmmer -N 1` against
the smallest database to prove the tools execute.

```sh
containers/verify.sh --af3-db /data/af3 --deep
```

| database | what reads it | size |
|---|---|---|
| `uniref90_2022_05.fa` | jackhmmer, the deep protein MSA | 72 GB |
| `mgy_clusters_2022_05.fa` | jackhmmer, environmental | 129 GB |
| `bfd-first_non_consensus_sequences.fasta` | jackhmmer, **small** BFD | 18 GB |
| `uniprot_all_2021_04.fa` | jackhmmer, pairing for complexes | 108 GB |
| `pdb_seqres_2022_09_28.fasta`, `mmcif_files/` | hmmsearch, templates | 0.2 GB + 235 GB |
| `nt_rna…`, `rfam…`, `rnacentral…` | nhmmer, RNA | 95 GB |

## 4. Predictors

These need a GPU, and two of them need something you have to obtain yourself:

| | how |
|---|---|
| Protenix v2, OpenFold 3, Boltz-2 | the `residudes-engine-trt` image, or `residudes-engine setup venv` then `setup checkpoints` — the weights download either way |
| AlphaFold 3 | the `residudes-engine-jax` image (it contains AF3's compiled extension). The code is Apache-2.0, but **the parameters are granted per person** by Google DeepMind and may not be redistributed, so no image or mirror can supply them (see below) |
| ESMFold2 | the `residudes-ef2` image, with the runner scripts mounted; or its own python 3.12 env, see [`residudes-engine/docs/esmfold2.md`](../residudes-engine/docs/esmfold2.md) |
| TensorRT engines | built per GPU, inside the image or outside it: `residudes-engine engines build` — engines are not portable between cards, so no image ships them |

**AlphaFold 3 parameters.** Access is requested from Google DeepMind through their form, is
granted to an individual, and the terms forbid passing the parameters on; there is therefore no
link this project could fetch them from and no copy it could host. What is automated is
everything after the grant:

```sh
residudes-engine setup checkpoints --models af3 --af3-from ~/Downloads/af3.bin.zst   # or a directory, or your URL
```

That installs the file into `[site] af3_weights` with mode 600 and reports the size. Run without
`--af3-from`, the same command prints the request form and this instruction instead of failing
silently. The weights of the other four predictors download automatically.

Checkpoints, engines and `af3.bin` are bind-mounted in every case. The images hold the runtime,
the host holds the data:

```sh
apptainer exec --nv -B $PWD/residudes-engine:/opt/residudes-engine -B /data/checkpoints:/ckpt \
    images/residudes-engine-trt.sif residudes-engine predict --model pt2 --input input.json
```

**Checked by:** `residudes-engine doctor` — one line per model for the checkpoint and the engines,
plus the venvs, the CUDA libraries and the site paths.

## 5. Relaxation (optional)

`residudes relax` runs the campaign's GROMACS + Rosetta pipeline under position restraints, both
orders of the two engines. The `residudes-relax` image has GROMACS 2024.2 and AmberTools 24.8
(`tleap`, `antechamber`, `parmchk2`, `reduce` — the pipeline calls all four for ligands). Two
things are mounted: **a Rosetta build**, which is licence-gated, and the pipeline itself, which
lives in the campaign workspace rather than in this repository. Point `[tools] minimization` at
`gromacs/relax/minimization.sh`.

The official `gromacs/gromacs` images stop at 2022.2, so this one is built from conda-forge;
the cluster ran GROMACS 2024.1, which conda-forge does not have, hence 2024.2.

## Steps reserved for a human operator

None of the steps above submits anything to CASP. `residudes approve` and `residudes resolve` are
operator commands; `residudes package` writes the submission files and halts. Compute-intensive
work is submitted through `sbatch` on the host: a container provides the environment in which a
job executes and does not itself queue work.
