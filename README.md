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
residudes/bin/residudes setup                # does the installation with you, then checks it
```

`setup` asks first what this machine already has — the AlphaFold 3 databases are 630 GB and
`af3.bin` is licensed per person, so finding them matters more than fetching them — and then
obtains the published images, installs what is missing and runs `containers/verify.sh` at the end.
It hands the work to whichever agent CLI is present (Claude Code, Codex, Antigravity, Gemini);
`--manual` asks the questions itself, and with no CLI on the machine it falls back to that by
itself.

Then a target. The repository carries one to start with: H1388, the input of the run in
[`residudes/examples/H1388`](residudes/examples/H1388), so what yours does can be read against what that one did.

```sh
cd residudes
bin/residudes target add H1388 --fasta examples/H1388/H1388.fasta --stoich A1B1
bin/residudes chat H1388                     # talk to the agent; with a mandate it runs on by itself (auto)
bin/residudes agent step H1388               # or one step per stage; `daemon --target H1388` keeps going
```

`target add` grants the agent a mandate up to a packaged slate (`--review` keeps a person at both checkpoints),
so `daemon --target H1388` takes it the whole way: that run took 45 minutes and $6.81 of agent steps, with 60
prediction jobs. For a target of your own, the same lines with its FASTA and stoichiometry.

What a run hands back is a report, ray-traced at the end; here are all thirty pages of H1388's, from the target
and what was known about it through the MSAs and templates, the pool and its clusters, the two checkpoints and
the slate, to what the agent spent ([PDF](residudes/examples/H1388/report/report.pdf)).

[![all thirty pages of the H1388 report, from the target and its priors through the pool, the checkpoints and the slate to what the agent spent](residudes/docs/figures/report-pages.jpg)](residudes/examples/H1388/report/report.pdf)

`chat` opens that target's conversation: ask what it is doing, steer it, leave with `ctrl-b d`, come back to
the same thread. On **auto** the conversation also runs the target to a packaged slate without stopping for
you: whenever it has been quiet for a minute (the agent idle, nothing typed), residudes types the next step
into it (`[auto] Next step for H1388 ...`), and the agent answers its own checkpoints and approves the slate
on the record, deciding from the evidence what a person would have decided. Auto has no pause, only done: it
ends at PACKAGED (submitting stays yours), or, if several asks pass with nothing new recorded, it stops and
says it is stuck. `--auto` turns it on and grants the mandate if the target has none; `--no-auto` takes both
back; a plain `chat` turns it on for a target that already has a mandate (`target add` without `--review`).
There is **one agent per target**: the daemon leaves a target on auto to its conversation rather than starting
a second agent beside it, so there is one context and you can watch or interrupt. All of them live as windows
of a single tmux session, so `tmux ls` stays one line, you arrive in the target you asked for, and `ctrl-b w`
pages to the others (a ▶ on the tab marks a target on auto). `residudes` on its own lists every target: which
are in conversation, which are waiting on a person, and what moves each on (`▶ auto`, `✓ auto done`, `✗ auto
stuck`, `▶ daemon`, or `■ manual`: only when asked).

Ask about a shape and the agent answers with the structure: it opens
[MolTerm](https://github.com/vv137/molterm), the team's in-house terminal molecular viewer, in a pane beside
the conversation, drives it as you talk (colours, labels, distances, another model on top) and closes it when
you are done. Here it was asked how H1388's phosphoserine sits in the 14-3-3 pocket. With a tmux built with
sixel (3.4 or later) and a terminal that shows images (in VS Code, `terminal.integrated.enableImages`) the
pane draws pixels, as here; otherwise it draws with characters, slim by default. The agent reads its own
full-resolution snapshot either way.

[![a residudes chat window: the conversation on the left, molterm on the right with the phosphoserine, the pocket residues and their distances](docs/figures/chat-molterm.png)](docs/figures/chat-molterm.png)

<sub>The tmux window itself, text and picture, drawn by [`docs/tmux_shot.py`](docs/tmux_shot.py) `--images`.</sub>

[`docs/setup.md`](docs/setup.md) documents the installation in full: container images or virtual
environments, the MSA databases, and the three components that must be obtained by the user
(AlphaFold 3 parameters, a Rosetta build and a MODELLER licence key).

## Documentation

| Topic | Document |
|---|---|
| Example run | [`residudes/examples/H1388`](residudes/examples/H1388/): registered and left alone, to a packaged slate in 45 minutes - 15 agent steps, $6.81, 300 models predicted for the run, both checkpoints answered by the agent; dossier, write-up, figures and every transcript |
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

[MolTerm](https://github.com/vv137/molterm): the team's in-house terminal molecular viewer (C++, MIT licence), with
a vim-like command line, text or pixel rendering (Sixel, Kitty, iTerm2) and PyMOL export. It draws every deck
before the final one and the structure pane in a conversation; its MCP server,
[`molterm-mcp`](https://www.npmjs.com/package/molterm-mcp), is how an agent drives it.

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
