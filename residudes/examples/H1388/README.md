# H1388: a target from the first command, through the analysis, to the plan a person has to answer

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

## What the four steps produced

[`analysis.md`](analysis.md) is the document; these are the parts worth looking at.

**It identified both chains from the sequence alone.** Chain B is 14-3-3 gamma (YWHAG, UniProt P61981, *Homo
sapiens*), Pfam PF00244, InterPro IPR000308 and three more, "all-alpha, 9 antiparallel helices; amphipathic groove
binds phospho-Ser/Thr peptides; obligate dimer via helices A–D". Target residues 2–232 are native 3–233, so
`input = native − 1`, and the acidic C-terminal tail 234–247 is not in the construct. Chain A is LARP1 542–552
(Q6PKG0).

**The chemistry the FASTA does not carry is the whole target.** The peptide has two serines, S5 and S7 (LARP1 S546
and S548). 14-3-3 binds only phosphorylated motifs, so one of them should be phosphoserine. The agent chose S7:
it sits in a pS-X-P motif (S7-L8-P9) and UniProt annotates S548 as a phosphosite, while S5 is S-A-S with no proline
at +2 and no annotation. That became `residudes decide --type variant`: SEP at chain A position 7, run beside the
unmodified input.

**MSA depth per chain, measured not assumed.** B is deep (uniprot 16,347, uniref90 6,536, mgnify 4,332, bfd 1,607;
no thin regions). A is an 11-residue peptide whose a3m files are query-only, so it runs single-sequence — which is
expected, not a failure, and the note says so.

**The structures that exist.** 23 exact-sequence human 14-3-3 gamma entries in the PDB, the best peptide-bound one
6A5S at 2.10 Å — rendered in [`report/figs/template_6a5s.jpg`](report/figs/template_6a5s.jpg).

## The figures the report draws for itself

One per kind of evidence, drawn from the dossier by `residudes report` with no job to run, and each its own page in
[`report/report.pdf`](report/report.pdf). They are separate on purpose: the species, the site and the motif are
different questions, and one picture of all three answers none of them.

| figure | what it says here |
|---|---|
| [`chain_map.svg`](report/figs/chain_map.svg) | both chains to scale with the residues a gate would use: S7 on the peptide, K49/R56/R131/Y132 in the phosphate pocket |
| [`literature.svg`](report/figs/literature.svg) | the five claims, their sources, and which priors and decisions share an identifier with each — plus the line that nothing read settles the 14-3-3:LARP1 complex itself |
| [`sites.svg`](report/figs/sites.svg) | the pocket and the motif residue by residue in their own sequences, with the ✱ where the phosphoserine goes, and the native numbering quoted beside the input numbering |
| [`motif.svg`](report/figs/motif.svg) | the arithmetic behind the target's whole decision: pS-X-P fits the peptide **only** at S7, S-A-S fits at S5, and the paper's full mode-I consensus (R-S-X-pS-X-P) fits nowhere — the peptide is a partial match |
| [`msa_coverage.svg`](report/figs/msa_coverage.svg) | coverage per residue for every MSA |
| [`clusters.svg`](report/figs/clusters.svg) | the existing pool four ways — clusters, predicted DockQ, confidence, and where each predictor sits (below) |

## The pool, clustered

The four agent steps stop at the plan checkpoint, but the CASP17 pool for this target already exists, so the
clustering page was computed afterwards, with the ordinary commands and no new predictions:

```sh
bin/residudes pool H1388                                                   # 1,750 models -> pool.csv
bin/residudes qa H1388 --sto A1B1 --what cluster --variants af3_msa,af3_msa_pS548,mmseqs2_gpu
bin/residudes qa H1388 --sto A1B1 --what interface                         # predicted DockQ per model
```

875 A1B1 models, nine clusters. The figure is the same UMAP four times — coloured by cluster, by predicted
DockQ, by the predictor's own confidence, and one small panel per predictor — then the clusters ranked by mean
predicted DockQ with what is in each, and four sentences the numbers support. The deck adds a page of **the
medoid of each cluster**, rendered from one view with the peptide as orange spheres, so the shape each cluster
claims is visible rather than asserted.

What it shows on this target:

- **cluster 1 holds all 125 `af3_msa_pS548` models and nothing else** — no unmodified model lands in it, no
  phosphoserine model lands anywhere else, and it has the highest confidence (0.960) and the highest mean
  predicted DockQ (0.66) of any cluster. The decision the agent made at HYPOTHESES separates the pool in two,
  which is what the person is deciding about at the checkpoint;
- the biggest cluster is not the best one: cluster 0 has 133 models and a mean predicted DockQ of 0.61;
- four clusters are one predictor's alone (bt2_jax 133, af3 125, of3 58, of3 31) — shapes no other predictor
  found, which is the case for hedging the slate across them rather than taking the top of one list.

The four columns the figure is drawn from ship with it
([`qa/cluster_A1B1_pool-befc310a_clustering.csv`](qa/cluster_A1B1_pool-befc310a_clustering.csv)), so the picture can
be checked against its numbers.

## The write-up

[`notes/target.md`](notes/target.md) is the document a person reads, written from the dossier in a separate
60-turn run once the analysis was done: identification, a subsection per chain, the biology that bears on the
structure, the sites and chemistry in input numbering with the native numbering beside them, what is uncertain, a
table of every structure that exists, the figures, and 20-odd numbered references. `residudes report` embeds it, so
[`analysis.md`](analysis.md) opens with it.

It is worth reading for one passage in particular. The plan recommends a gate of 4 Å between the A7 phosphate and
the B pocket, so the agent measured the distances in 6A5S itself: 2.4–2.8 Å from the phosphate oxygens, 3.6–3.9 Å
from the phosphorus, 4.2–5.1 Å from the serine OG. The unmodified runs have no phosphate, so for those the only
atom to measure from is OG — and measured from OG the reference structure itself fails a 4 Å gate. The note says
the spec has to name the atom or use about 5.5 Å. That is the agent finding the hole in its own recommendation
before a person answers the checkpoint.

## Where it stops

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

## What was edited after the run

One thing, everywhere it appears (logs, `analysis.md`, `notes/target.md`, `dossier.json`): the path of the
checkout the run used is written as `<RESIDUDES>`. Campaign paths under `/gpfs/deepfold/casp/casp17/` are left as
they were, because the write-up's counts are checkable against those files and a sanitised path would not be.
Nothing else in the transcripts was changed — including the names. The run predates the rename of `lantern`,
`lantern-engine` and `lantern-qa` to `residudes`, `residudes-engine` and `residudes-qa`, so the logs, the dossier
and the figures drawn from them still say the old ones: those are the commands that ran. This page uses the
current names, so `residudes target add` here is `lantern target add` in the log beside it.

## What it was not allowed to use

The operator note barred the team's own submission and any native structure until RETROSPECTIVE, so the analysis
here is what the agent found from the target files and the literature. One thing it could not be shielded from: the
`residudes-intake` skill quotes this target family as a worked case, and the agent said so in a note rather than
using it quietly.

The team's CASP17 answer, for comparison after the fact: the phosphosite was the target's lever, and adding it took
AF3's median DockQ from 0.24 to 0.82 ([`casp17_cases.md`](../../docs/casp17_cases.md)).
