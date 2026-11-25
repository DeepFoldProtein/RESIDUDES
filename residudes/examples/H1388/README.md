# H1388: a complete recorded run, from the first command to the open checkpoint

Run on 2026-11-23, from an empty dossier: `residudes target add` and nothing else, so that what is here is what the
agent read, recorded and asked, rather than evidence handed to it.

H1388 is human 14-3-3 gamma with an 11-residue LARP1 peptide, declared A1B1.

```sh
bin/residudes target add H1388 --fasta casp17/targets/H1388/target.fasta --stoich A1B1
bin/residudes note H1388 "OPERATOR: this run is an example of the pipeline from the start, so do the analysis
  yourself. The team's own submission is on disk at targets/H1388/submit/ - do not read it, and do not read any
  native structure, until RETROSPECTIVE. Everything else under targets/H1388 is yours to use, as is the literature."
bin/residudes agent step H1388 --budget 3.0      # x4, one per stage
```

Four steps, **$2.15 in total, 11 to 24 turns each** ([`steps.json`](steps.json), [`agent_logs/`](agent_logs/)):
NEW → IDENTIFIED → HYPOTHESES → PLAN_REVIEW, where it stops, because a person resolves a checkpoint.

## Contents

| File | Contents |
|---|---|
| [`analysis.md`](analysis.md) | the document `residudes report` writes from the dossier |
| [`dossier.json`](dossier.json) | everything the agent recorded: priors, sites, decisions, evidence |
| [`steps.json`](steps.json) · [`agent_logs/`](agent_logs/) | cost and turns per step, and every transcript |
| [`notes/target.md`](notes/target.md) | the write-up a person reads, with its references |
| [`checkpoints/C1_plan.md`](checkpoints/C1_plan.md) | the open checkpoint: four questions and the thread |
| [`report/`](report/) | the deck, and the figures drawn from the dossier |
| [`pool.csv`](pool.csv) · [`qa/`](qa/) | the 1,750 existing models, their clusters and predicted DockQ |

| Topic | Document |
|---|---|
| what the steps actually found | [`docs/findings.md`](docs/findings.md) — both chains, the phosphosite, MSA depth, the figures |
| the pool, clustered | [`docs/clustering.md`](docs/clustering.md) — nine clusters, and the write-up that corrected its own gate |
| what was edited and withheld | [`docs/provenance.md`](docs/provenance.md) — one sanitised path, and the native structures it could not read |

## Termination at the plan checkpoint

[`checkpoints/C1_plan.md`](checkpoints/C1_plan.md) is the brief, and C1 is open with four questions, each carrying
the option the agent recommends and what it costs:

1. phosphorylation — SEP at A7 and the unmodified input as a control, or unmodified only;
2. stoichiometry — submit A1B1 and run A2B2 as a dimer-context hedge, or A1B1 alone, or submit the dimer;
3. GPU — ≤ 40 of the 400 GPU-h on top of the ~1,400 models already in the pool, or none, or +100;
4. the selection gate — the A7 phosphate within 4 Å of the B pocket first and confidence second, or confidence only.

The thread already has one message, and it is the agent correcting itself: after measuring 6A5S for the write-up
it posted that option (a) of question 4, its own recommendation, should not be used as written, and gave a revised
one that names the atom per variant — phosphate oxygens within 4 Å for the SEP models, OG within about 5.5 Å for
the unmodified ones. The questions a person answers are therefore the corrected ones.

Nothing past this point is the agent's to decide: `residudes resolve H1388 C1 --outcome proceed | loop:HYPOTHESES`
is a person's command, and the stage machine refuses it from the agent.
