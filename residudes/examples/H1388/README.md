# H1388: one target, registered and left alone, to a packaged slate

Run on 2026-11-26 from an empty dossier, under the default that `residudes target add` now sets: a mandate from the
person who registered the target, up to a packaged slate. Nobody answered a checkpoint or approved anything; the
agent did, on the record, and stopped where submission begins.

H1388 is human 14-3-3 gamma with an 11-residue LARP1 phosphopeptide, declared A1B1.

```sh
residudes target add H1388 --fasta H1388.fasta --stoich A1B1     # full automation, up to PACKAGED
residudes daemon --target H1388
```

**15 agent steps, $6.81, 197 turns; 45 minutes from registration to package**, predictions included
([`steps.json`](steps.json), [`agent_logs/`](agent_logs/)). The predictions were made for this run, by
residudes-engine built from this checkout: 60 Slurm jobs, 300 models, AF3, Protenix, Boltz-2 and OpenFold3.

```
NEW → IDENTIFIED → HYPOTHESES → PLAN_REVIEW ─C1─→ INPUTS_READY → RUNNING → POOL_READY → RESULTS_REVIEW ─C2─→
SLATE_PROPOSED → APPROVED → PACKAGED   (stops: submission is a person's)
```

## Contents

| File | Contents |
|---|---|
| [`analysis.md`](analysis.md) | the document `residudes report` writes from the dossier |
| [`dossier.json`](dossier.json) | everything recorded: priors, sites, seven decisions, evidence, both checkpoints |
| [`steps.json`](steps.json) · [`agent_logs/`](agent_logs/) | cost and turns per step, and every transcript |
| [`notes/target.md`](notes/target.md) | the write-up a person reads, with its references |
| [`checkpoints/`](checkpoints/) · [`brief_plan.md`](brief_plan.md) · [`brief_results.md`](brief_results.md) | C1 (plan) and C2 (results): the briefs, the questions, and the agent's own answers |
| [`inputs/`](inputs/) | the two inputs it built: phosphoserine on the peptide, 6S9K pinned, a deep MSA on the receptor |
| [`pool.csv`](pool.csv) · [`qa/`](qa/) · [`gates/`](gates/) | the 300 models, their interface scores, clusters and the pocket gate |
| [`slate_v2.md`](slate_v2.md) · [`package/`](package/) | the approved five and the files packaged from them, with the geometry checks |
| [`report/`](report/) | the deck, ray-traced at the end (`residudes slides T --final`), and the figures drawn from the dossier |

| Topic | Document |
|---|---|
| what the steps found and decided | [`docs/findings.md`](docs/findings.md) |
| how the slate scores against the native | [`docs/retrospective.md`](docs/retrospective.md) — and why that is not a blind result |
| what was edited, withheld and fixed mid-run, and the deck drawn after it | [`docs/provenance.md`](docs/provenance.md) |

## What it shows

- **Both checkpoints were opened and written as for a person**, then answered by the agent under the mandate.
  C1's brief asked four questions, each with options and a recommendation; the agent's reply took each one and
  closed two inconsistencies in its own plan before resolving. C2 came from the results review, which raised a
  peptide-register split between the predictors and a phosphate that only AF3 carried.
- **The slate is three AF3 models with the phosphate and two Protenix models without it**, chosen because the two
  agree on the peptide's register while Boltz-2, OpenFold3 and unmodified AF3 do not. All five pass every gate,
  and the packaged files pass the geometry checks with nothing to relax.
- **Everything decided under the mandate carries the mark.** `resolution.automated` and `approval.automated`, with
  who granted the mandate, so the log never blurs whether a person or the agent decided.
