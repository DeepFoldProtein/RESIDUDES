# RESIDUDES

<img src="mascot/logo/logo_256.png" align="right" width="200" alt="the RESIDUDES mascot: an otter in a miner's helmet, lantern raised, waving">

Protein structure prediction software used by the DeepFold team in CASP17 (group 077). The repository contains
three components: an agent that carries a target from its sequence to a slate of five submitted models, halting at
the points where a human decision is required; a prediction toolkit; and a quality-assessment stack for ranking
the resulting models.

| Component | Description |
|---|---|
| [`residudes/`](residudes/) | campaign agent: target analysis, input construction, structural gates, slate assembly, and two human checkpoints |
| [`residudes-engine/`](residudes-engine/) | prediction toolkit: five predictors (Protenix v2, OpenFold 3, Boltz-2, AlphaFold 3, ESMFold2) from a single input file, on PyTorch, TensorRT or JAX backends |
| [`residudes-qa/`](residudes-qa/) | quality assessment: scoring, clustering and visualisation of a prediction pool |
| [residudes-decoy](https://residudes-decoy.deepfold.org) | released dataset: 726,097 unfiltered CASP17 predictions over 82 targets, each with the input that produced it ([description](docs/residudes-decoy.md)) |

The first three are in this repository; the fourth is the data they produced, released so that the predictions
behind the CASP17 entry, and the inputs that generated them, can be examined independently.

[![residudes drives residudes-engine and residudes-qa: a stage line above, the prediction lane and the QA lane below](docs/figures/overview.png)](docs/figures/overview.svg)

**The three components applied to one target.** The agent's stages are shown along the top, with the two
human checkpoints in orange. Below, a sequence is expanded into a pool of predictions, and the pool is reduced to a
ranked selection. All steps after slate assembly require human action.

## Installation

```sh
git clone https://github.com/DeepFoldProtein/RESIDUDES && cd RESIDUDES
containers/verify.sh --build                 # build the images, then check the setup end to end

cd residudes
bin/residudes setup                          # resources, databases, config, skills; with you
bin/residudes target add T1234 --fasta T1234.fasta --stoich A3
bin/residudes agent step T1234               # one step per stage; `daemon --target T1234` keeps going
```

[`docs/setup.md`](docs/setup.md) documents the complete installation: container images or virtual environments,
the 630 GB of MSA databases, and the three components that must be obtained by the user (AlphaFold 3 parameters, a
Rosetta build and a MODELLER licence key).

## Documentation

| Topic | Document |
|---|---|
| Example run | [`residudes/examples/H1388`](residudes/examples/H1388/): four agent steps at a cost of $2.15, with dossier, write-up, figures, the open checkpoint and all transcripts |
| Component documentation | [`residudes/README.md`](residudes/README.md) · [`residudes-engine/README.md`](residudes-engine/README.md) · [`residudes-qa/README.md`](residudes-qa/README.md) |
| Effect of model selection | [`residudes/docs/casp17_cases.md`](residudes/docs/casp17_cases.md): per-target intervals, including cases where selection lost accuracy |
| Setup and containers | [`docs/setup.md`](docs/setup.md) · [`containers/README.md`](containers/README.md) |
| Released sampling pool | [`docs/residudes-decoy.md`](docs/residudes-decoy.md) |
| Motivation and scope | [`docs/why.md`](docs/why.md) |
| Tests, CI and figure checks | [`docs/testing.md`](docs/testing.md) |
| Licence terms in full | [`docs/licences.md`](docs/licences.md) |

## Components

[![one dossier per target: thirteen stages, two checkpoints, three loops back](residudes/docs/figures/overview.png)](residudes/docs/figures/overview.svg)

`residudes`: the stage sequence of the agent. Human checkpoints are shown in orange.

[![a sequence to a structure: MSA and templates, one input file, five models, three backends](residudes-engine/docs/figures/overview.png)](residudes-engine/docs/figures/overview.svg)

`residudes-engine`: construction of a single input file, five predictors, and measured throughput per backend.

[![a prediction pool to a ranked one: interface scores, clustering, plots](residudes-qa/docs/figures/overview.png)](residudes-qa/docs/figures/overview.svg)

`residudes-qa`: scoring, clustering and visualisation of a prediction pool.

[![the residudes-decoy site: a predicted structure beside that target's models, each with its predictor, input variant, seed and confidence](docs/figures/residudes-decoy-home.png)](https://residudes-decoy.deepfold.org)

[residudes-decoy](https://residudes-decoy.deepfold.org): the released sampling pool, browsable per target and per
predictor, with every structure viewable in place. Each model carries the input that produced it — predictor,
input variant and seed — so a record can be reproduced rather than only scored
([layout and terms](docs/residudes-decoy.md)).

## Citation

Please cite the CASP17 abstract of group 077. The same reference covers the
[residudes-decoy](docs/residudes-decoy.md) pool, so that the software and the data it produced are cited together.
[`CITATION.cff`](CITATION.cff) carries equivalent metadata for GitHub's "Cite this repository" function.

```bibtex
@misc{residudes_casp17,
  title  = {Protein Complex Structure Prediction with RESIDUDES in CASP17},
  author = {Bae, Hanjin and Park, Gunwoong and Jo, Gyeonpil and Kim, Kunwoo and Kim, Minsoo and
            Yoo, Jaemin and Kim, Beomseok and Shin, Jaeoh and Kim, Hosung and Kim, Iljung and
            Noh, Yung-Kyun and Yoo, Jejoong and Joo, Keehyoung},
  year   = {2026},
  note   = {CASP17 abstract}
}
```

Use of structures from the pool additionally requires citation of the predictor that generated them; the
AlphaFold 3 subset requires the AlphaFold 3 paper.

## Licence

Code and documentation original to this repository are released under the **[MIT](LICENSE)** licence. Two
categories are excluded. First, vendored upstream code retains the licence of its authors; AlphaFold 3 is
CC BY-NC-SA 4.0 (**non-commercial**). Second, the **[residudes-decoy](docs/residudes-decoy.md) dataset** retains,
for each structure, the terms of the predictor that generated it: the 194,183 AlphaFold 3 structures are
non-commercial and **may not be used to train structure-prediction, scoring or ranking models**. Full terms are
given in [`docs/licences.md`](docs/licences.md) and should be consulted before any reuse.
