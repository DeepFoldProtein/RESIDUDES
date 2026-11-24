# Verification

What has been checked, on which hardware, and how to repeat it. Every row
below is a Slurm job on the team's cluster (RTX PRO 6000 Blackwell nodes),
run from this repository's own runtime: the venv built by `residudes-engine setup venv`
from the lockfile, weights fetched by `residudes-engine setup checkpoints`, engines
built by `residudes-engine engines build`. Numbers are the top sample's pLDDT.

## The three backends give the same answer

Target T1430 (36 residues, MSA and four templates). The reference is the
prediction made during CASP17 with the original TensorRT pipeline.

| backend | Protenix v2 | OpenFold 3 | Boltz-2 | AlphaFold 3 |
|---|---:|---:|---:|---:|
| TensorRT, engines built here | 89.5 | 76.8 | 94.9 | 89.0 |
| PyTorch | 89.5 | 77.4 | 95.0 | 86.7 |
| JAX | – | 77.4 | 95.0 | 86.1 |
| reference | 89.5 | 76.9 | 95.2 | 86.3 |

Differences of a few tenths come from BF16 arithmetic and kernel
nondeterminism; Boltz-2 seeds 2 and 3 agree as well. AlphaFold 3 draws a
random 1024-row subset of this target's 6157-row MSA at every recycle, so
its runs spread over a few pLDDT points even for one backend (the five
samples of one run span 84 to 89); the reference is the CASP17 run of the
JAX model. Across 178 benchmark
runs the top models of the TensorRT and JAX backends match the PyTorch top
model with median TM-scores of 0.87 to 1.00 per model
([benchmark.md](benchmark.md), Figure 3).

## Clone, set up, predict: the new-machine path

| step | result |
|---|---|
| `setup venv` on a compute node | `core/.venv` from `core/uv.lock`, 13 GB, 172 packages, one pass |
| `setup checkpoints` | all weights fetched and checked by size and sha256: OpenFold 3 from the publisher's bucket, Protenix v2 from a mirror (the publisher's URL refuses non-browser clients), Boltz-2 through its own downloader |
| `predict --backend torch` on the downloaded weights | matches the reference (table above) |
| `engines build --model bt2` | a Boltz-2 engine in 12 minutes on one RTX PRO 6000; `predict --backend auto` picked it up and matched the reference |
| `doctor` on the login node | paths, `GLIBCXX_3.4.32` libstdc++, venv imports of torch / TensorRT / JAX with its CUDA plugin, TensorRT plugin load |
| `submit --seeds 1,2,3` | one job per seed, each in its own `seed_<N>/` |

Not verified: the RTX 4090 engine class. Those jobs never received a slot on
the shared 4090 nodes and were cancelled after a day in the queue.

## From a sequence to a prediction

`residudes-engine msa` and `residudes-engine inputs` were run end to end and their
`input.json` predicted on all three models (PyTorch backend, seed 1).

| input | how it was made | Protenix v2 | OpenFold 3 | Boltz-2 |
|---|---|---:|---:|---:|
| T1430, DeepFold-PLM MSA | `msa --source plmmsa` (1602 sequences in 170 s) | – | – | 93.6 |
| + self template and pinned template (1UBQ chain A) | `--self-template A=<seed dir> --pin-template A=1ubq:A@1` | 89.6 | – | 95.2 |
| + ATP ligand | `--ligand ATP=ccd:ATP` | 91.5 | – | 68.4 |
| + NAG-NAG glycan on Asn9 | `--glycan A9:NAG-NAG`, two covalent bonds | 94.2 | – | 87.3 |
| AsnA (330 aa) + a 21-residue peptide, ColabFold | `msa --source colabfold`: 3503 / 12 sequences, paired MSA, two PDB templates per chain | – | 89.9 | 94.5 |
| AsnA homodimer with the 12AS A/B interface template | `--sto A2 --multimer-template 12as:A=A,B=B` | 94.4 (ipTM 0.96) | 92.4 (0.91) | 98.1 (0.96) |

All runs exited 0. The logs of all three models show the interface template
being read (Protenix's template featurizer, OpenFold 3's cross-chain template
rows, Boltz-2's multimer template patch). OpenFold 3 has no covalent-bond
input, so a glycan is modelled there as a non-covalent ligand; the adapter
says so in the log.

## AlphaFold 3 on three backends

AlphaFold 3 arrived after the sections above. Its JAX backend is DeepMind's
model as published, so the question is whether the PyTorch port and the
TensorRT engines built from it reproduce that model.

| check | how | result |
|---|---|---|
| every block against its Haiku module | `core/alphafold3/tests/compare_jax.py`, CPU, float32, random inputs with padding | relative error 1e-5 or below on all 16 blocks (transitions, triangle multiplications, pair attention, MSA attention, pairformer, template, diffusion transformer, atom cross-attention) |
| the engine's reference graph against the port | `core/alphafold3/tests/compare_trt_reference.py`, GPU | 1e-5 on the real tokens for the pairformer and the token transformer |
| the whole model against JAX on T1430 | `core/alphafold3/tests/compare_jax_e2e.py`, one process, same featurised input and seed | trunk embeddings within bf16 rounding with the MSA fixed to the same rows; the confidence head gives the same pLDDT and pTM on identical coordinates; with the full MSA (random 1024 rows per recycle) both models' samples span the same pLDDT range |
| the same complex on all three backends | H1335 (1627 tokens, three chains), one seed | top pLDDT 80.0 (PyTorch), 79.7 (TensorRT), 79.7 (JAX); pTM 0.694 / 0.693 / 0.69 |

The end-to-end check found one real bug on the way: the port's MSA subsampling
drew the first 1024 rows instead of a random 1024 every recycle, which lifted
pLDDT on deep-MSA targets by about five points. It also found that the port's
attention was running on PyTorch's unfused fallback path; with that fixed the
port is within a factor of two of JAX and the TensorRT engines close most of
the rest ([alphafold3.md](alphafold3.md), [benchmark.md](benchmark.md)).

## Unit tests

`python3 -m unittest discover -s tests` runs 25 tests in well under a second
without a GPU or the venv: plans for every model and backend, output
flattening, the verifier, the CLI parser, engine discovery, the site writer,
and the input layer's parsers and JSON assembly.

## Repeat it

```sh
residudes-engine setup venv --slurm            # or plain `setup venv` on a GPU machine
residudes-engine setup checkpoints
residudes-engine doctor
residudes-engine submit --model pt2 --input <input.json> --out runs/check/pt2 --seeds 1
residudes-engine verify runs/check/pt2/seed_1 --ref <a reference seed directory>
python3 -m unittest discover -s tests
```

`verify --ref` prints both confidence tables side by side; a top pLDDT within
half a point of the reference is the expected outcome.
