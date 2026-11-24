---
name: residudes-consult
description: Work a target through with a person at the two checkpoints - the prediction plan after the analysis (PLAN_REVIEW) and the first results (RESULTS_REVIEW) - with a brief, questions that carry options and a recommendation, and a back-and-forth until the person decides to proceed or loop. Use at PLAN_REVIEW, at RESULTS_REVIEW, and whenever a person has replied in an open checkpoint.
---

# Checkpoints with a person

In CASP17 the big calls were made with a person, after reading what the agent had found. The decision to run the
custom C-terminal MSA, the choice of A3 over A1 and the repeat sampling of split architectures were all made that
way. residudes stops for this twice per target. The agent prepares; the person decides.

```
HYPOTHESES → PLAN_REVIEW ⇄ person → INPUTS_READY → RUNNING → POOL_READY → RESULTS_REVIEW ⇄ person → SLATE_PROPOSED
                  └─ loop:HYPOTHESES                                         ├─ loop:RUNNING     (more sampling)
                                                                             └─ loop:HYPOTHESES  (new hypotheses)
```

## 1. Plan review (after the analysis and literature)

Build the slide report first: `residudes slides T`. Add your own slides and PyMOL scenes as AGENTS.md describes.
People read the deck with the pictures, not the JSON. Then write a brief, `brief_plan.md` in the dossier
directory, that points to `report/report.pdf`, and open the checkpoint:
`residudes consult T --kind plan --brief brief_plan.md --ask "..." --ask "..."`. residudes appends the full analysis
(`residudes report`), so the brief can stay short.

Opening the checkpoint also redraws the figures and links them from the brief: the chain map, the literature
claims with what rests on each, the sites residue by residue, every place the recorded motifs can sit, and the
clustering once there is a pool. Refer to them rather than re-describing them in prose — and read them first
yourself, because a figure that disagrees with your sentence (a motif that fits elsewhere too, a site residue that
is not the letter you named) is the cheapest review you will get.

The brief, in this order:
1. **The target in three lines.** Construct, identity and family; anything from the literature that bears on structure.
2. **What we know**, each item with its evidence key:
   - what the literature says, with the papers named: the site a partner binds, the oligomeric state, the chemistry,
     and what is **not** known (no paper, an uncharacterised family — say so, so the person knows the confidence is
     thin). A site from the target's own experiments is a stronger prior than the same site class on a relative, and
     the brief should say which it is ([`docs/literature.md`](../../docs/literature.md));
   - public structures: an exact or near-identical PDB entry changes everything (CASP17: a sweep of the whole target set found 47 with one, and 8ARV
     was hidden by the near-identity filter);
   - MSA depth, and any regions without an MSA;
   - templates;
   - the stoichiometry evidence.
3. **What is uncertain, and how the plan covers it.** For example: no MSA for residues 125–249, so a custom MSA run;
   the copy number is undeclared, so an n-sweep.
4. **The plan.**
   - Stoichiometries: the primary and the hedges.
   - Input variants: MSA, templates, PTMs and glycans, ligands, construct.
   - Seeds per predictor and variant.
   - GPU hours against the budget.
   - The QA to run.
   Every item should already be a `residudes decide` with its evidence keys.
5. **Questions.** One `--ask` each, written so a person can answer in a line:
   `"Copy number: (a) A2 primary + A1 hedge [recommended: 3 predictors peak at A2], (b) A1 only, (c) add A4"`.
   Say what changes with each answer: seeds, GPU hours, the slate.
6. **The default.** Say what you will do if the deadline comes before an answer.

Questions worth asking at this checkpoint, from CASP17:
- **An undeclared stoichiometry.** A missing precedent is not a monomer. T1416 lost three slots by betting on A1.
- **A near-identical public structure.** Use it as a template, and run without it as a control.
- **Regions with no MSA.** A custom MSA (domain-split, metagenomic, plmMSA) gave +0.3–0.5 lDDT.
- **Construct.** Tags and scars, a ΔN soluble head, signal peptides.
- **Antibody targets.** Epitope priors from the antibody's own literature (escape mutants, mutagenesis, a mapped
  epitope), with the paper named. AF3 leans to CDR–CDR contacts, so the prior is what moves the pick: H2319's epitope
  gate gave model 1 DockQ 0.93 against 0.67 for the most confident model in the pool.
- **Sites that become gates.** An epitope, a recognition sequence, a pocket: say which gates the plan will apply at
  selection, so the person can object before the runs, and so the choice is on record: on H2321 it was not, and
  the submitted model turned out to fail one of the gates it was supposed to pass.
- **Chemistry in the description.** A "phosphopeptide" with no modification in the FASTA (H2388: the pS-X-P motif
  picked S7, DockQ 0.24 → 0.82). Name the candidates and the one you chose.
- **Glycans and ligands that change the fold.** Covalent glycans helped pt2 on T1363.

## 2. Results review (after the first round)

1. Run `residudes review T`. It flags these triggers:
   - `low_confidence`;
   - `stoichiometry_split` and `flat_nsweep`;
   - `low_plddt`;
   - `clashes`;
   - `missing_models` (silent failures);
   - `architecture_split` (the predictors' best models fall in different clusters);
   - `weak_interface`;
   - `missing_chains`: a run whose models lack a chain of the stoichiometry (M2377: 310 models without the RNA).
     Keep those runs out of the slate and find out why;
   - `empty_models`: summaries without a model file (left out of the pool; CASP17 had 64 across 7 runs);
   - `site_split` (from a `residudes gate` census): the most confident models sit at different sites;
   - `site_prior_without_gate`: the evidence names a site (epitope, pocket, recognition site) that no gate uses.
2. Add what you see with `--add "name: why"`. Look for:
   - single-sequence models that disagree with the MSA-based ones. Do not dismiss either side: a single-sequence
     model can carry the right interface while looking overconfident;
   - templated models that disagree with template-free ones. Templated ipTM is not comparable;
   - helices where cross-β is possible: stacking at about 4.8 Å means a fibril;
   - an interface on an unexpected face;
   - results that contradict a prior from the plan.
3. If there are no triggers, make the slate (`residudes-selection`).
4. If there are any, show them in pictures: a PyMOL scene of what you saw (`residudes render`), for example the split
   architectures superposed or the clash, on a slide in `report/agent.typ`. Then run `residudes slides T` and open a
   checkpoint: `residudes consult T --kind results --ask ...`. Show the evidence (numbers and model names), then give
   options with costs and a recommendation:
   - more seeds for the split architectures (`loop:RUNNING`), when the pool lacks good candidates. More seeds do not
     help when the ranker cannot pick between them. In M1439 the best model in the sample rose from lDDT 0.59 to
     0.63 between 15 and 1,920 models, while the top-ranking_score pick stayed at 0.54. A `site_split` is settled by
     what is known about the site (a gate or `--site`), not by sampling. The same holds for an architecture split,
     and T1417 is the measured case: over the models within TM 0.85 of each of three trimer architectures, af3's
     medians are ipTM 0.90 / interface score 0.789 for X against 0.92 / 0.787 for Y1, and on of3 ipTM puts Y1 first
     while the interface score puts Y2 first. Neither number separates them, and more of them would not;
   - a new variant, such as a custom MSA, a pinned template, a ligand or another construct (`loop:HYPOTHESES`);
   - another copy number (`loop:HYPOTHESES`);
   - proceed, and hedge the slate across the alternatives (`proceed`).

## 3. In the conversation

- When a person replies, the daemon gives you a step. Read the thread (`residudes checkpoints T C#`) and answer what
  they asked: in numbers, with model names, and briefly. Use `residudes reply T C# "..."`.
- If they ask for a change, make it: new `residudes decide` entries citing the checkpoint ("C1: person asked for A4").
  Then reply with what changed.
- **Never resolve a checkpoint yourself.** A person runs `residudes resolve T C# --outcome proceed|loop:RUNNING|loop:HYPOTHESES`.
- After `proceed` you continue: INPUTS_READY after the plan, the slate after the results. After a loop you are back at
  that stage with the person's decision on record.
