# H1388 plan review

Slides: /store/deepfold/lantern/targets/H1388/report/report.pdf (agent slides at the end: 6A5S template scene with the phosphate pocket, plan at a glance).

## The target in three lines
- A = 11-aa peptide PKGLSASLPDL, human LARP1 (Q6PKG0) residues 542-552; B = human 14-3-3 gamma (YWHAG, P61981) 3-233. Given A1B1. (`construct`, `identity`)
- 14-3-3 proteins bind phosphoserine/threonine motifs in an amphipathic groove; the peptide very likely carries pS548 (= A7, pS-L-P, mode-I-like), although the FASTA names no modification. (`literature`, `motif`, `chemistry`)
- No paper or structure of 14-3-3:LARP1 itself; placement rests on the family prior (strong, many gamma:pS-peptide crystal structures), not on target-specific experiments.

## What we know
- Literature (`literature`): 14-3-3 mode I/II phosphopeptide recognition (Yaffe 1997; Rittinger 1999), phosphate held by K49/R56/R127/Y128 (zeta numbering) = gamma K50/R57/R132/Y133. LARP1 S548 annotated phosphosite in UniProt. NOT known: whether the experiment had one or two peptides on the dimer; the CASP description text was not available to me.
- Public structures (`public_structures`): 23 exact-sequence 14-3-3 gamma entries, several peptide-bound (6A5S 2.1 Å, 2B05, 6GKG, 6FEL). LARP1 appears only in mTORC1 cryo-EM (9T76/9T77) - none with 14-3-3.
- MSA (`msa_depth`): B deep (uniref90 N=6536, uniprot N=16347); A is a peptide, query-only - expected.
- Templates (`templates`): AF3 default search returns 91 gamma hits including peptide-bound ones.
- Stoichiometry (`stoichiometry_evidence`): 14-3-3 is an obligate dimer in all 6/6 gamma assemblies checked; CASP gives A1B1 (one groove).
- Sites (`binding_site`, `motif`): B 49/56/131/132 pocket; A S7 phosphosite. S5 rejected (S-A-S, no Pro at +2).

## What is uncertain and how the plan covers it
- Phospho site / whether phosphorylated: run unmodified and SEP-A7 side by side (D1); gate selects the pS7-in-pocket register either way.
- Peptide register / direction in the groove (an 11-mer can slide): gate on S7 at the pocket, plus check against the 6A5S/2B05 peptide backbone; custom template only if the pool splits (D4).
- Monomer vs dimer context: A1B1 submitted, A2B2 run and cut to one pair (D2) - the dimer may give a better groove.

## The plan (all recorded as decisions D1-D6)
- Stoichiometry: A1B1 primary; A2B2 hedge, cut to a B:A pair.
- Inputs: af3_msa, af3_msa_pS548, mmseqs2_gpu; peptide single-sequence; no ligand/metal.
- Sampling: existing CASP17 pool, ~1.4k models (af3 25 seeds x 5 per sto/variant; bt2_jax and of3 unmodified). No new GPU now.
- GPU: reserve <=40 of 400 GPU-h: rerun pt2 (all runs left no model files) and of3/bt2 with SEP if needed.
- QA/selection: `lantern pool`, gate "A7 OG/P within 4 Å of B K49/R56/R131/Y132 side chains", interface QA, then af3 ranking within gated models; slate alternates predictors, SEP variant leads.

## Questions
1. Phosphorylation: (a) treat A7 as pS548 and lead with SEP models, unmodified as control [recommended: pS-X-P motif, 14-3-3 requires a phosphate], (b) unmodified only, (c) equal weight.
2. Stoichiometry context: (a) A1B1 submitted, A2B2 run as a context hedge and cut to one pair [recommended], (b) A1B1 only, (c) submit the A2B2 dimer.
3. Extra GPU: (a) only the <=40 GPU-h reserve (pt2 rerun, SEP for of3/bt2) [recommended], (b) none - existing pool only, (c) +100 GPU-h more af3 seeds with a custom 6A5S template.
4. Gate: (a) pS7-in-pocket gate applied before confidence [recommended], (b) confidence only.

## Default if no answer before the deadline
1a, 2a, 3a, 4a: import the pool, gate on pS7 in the pocket, slate of five A1B1 models led by af3 SEP.
