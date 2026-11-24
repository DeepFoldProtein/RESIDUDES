---
name: residudes-stoichiometry
description: Decide copy numbers and assembly state for a target from evidence. Use at dossier stage IDENTIFIED, on a stage-2 reissue, or whenever predictions disagree about the oligomeric state.
---

# Stoichiometry

CASP17 scorecard (58 stage-1 / stage-2 pairs): 38 right, 13 over-predicted (H targets), 7 under-predicted (T targets). Under-prediction was the expensive error: T1404 A3 → A17 cost the field position 81 % → 9 %.

## Evidence to collect

Record each item with `residudes evidence T stoichiometry_evidence '<json>'`:
- seq2symm prior;
- homolog assemblies (PISA, foldseek against assemblies);
- the empirical copy-number distribution of the family (e.g. lipoproteins 76.5 % A1);
- **what the papers say the assembly is**, with the source: the state on the particle or in the cell, SEC, native MS,
  EM of the whole particle, the family's biology. H2319's antigen is presented as a dimer on the mature particle, so
  it was run as a dimer with two Fvs although CASP declared the stoichiometry unknown (`residudes-intake`,
  [`docs/literature.md`](../../docs/literature.md));
- an **n-sweep** A1…A6+ across at least two predictors: a sharp peak at one n is positive evidence, a plateau is not;
- interface conservation, and the screw rise/twist for filaments.

## Rules

1. **Absence of evidence is not evidence of a monomer.**
   - No precedent, no template or no conserved interface does not refute an oligomer.
   - T1416: ipTM peaked at A3 on two predictors, but A1 was made primary for lack of precedent. The answer was A3.
2. **Multiple predictors reproducing the same n is positive evidence.** Weigh it above "no homolog" from a stale database.
3. **A1 is not the default for T targets.** Open assemblies (filaments, arcs, rings) get a large n or a hedge across n.
4. **Siblings together.** Re-check every target of a family or a multi-target release together; errors repeated across siblings.
5. **Stage 2 overrides.** When the organisers declare a stoichiometry, adopt it and re-plan (`residudes stage T HYPOTHESES`).
6. **Hedge in the slate, not in the primary.** Model 1 is the best-supported n. Put alternatives in models 2–5, and do not put all five on the same n when the evidence is split.

Finish with `residudes decide T --type stoichiometry --chose <n> --evidence ... --alt <other n> --confidence ...`.
