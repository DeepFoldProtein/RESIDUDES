# H1388 — checkpoint C1 (plan)

Opened by agent at stage PLAN_REVIEW.

## Questions

1. Phosphorylation: (a) A7 = pS548 (SEP), lead with SEP models, unmodified as control [recommended: pS-L-P motif, 14-3-3 needs a phosphate], (b) unmodified only, (c) equal weight
2. Stoichiometry: (a) submit A1B1, run A2B2 as dimer-context hedge cut to one B:A pair [recommended], (b) A1B1 runs only, (c) submit the A2B2 dimer
3. GPU: (a) only <=40 of 400 GPU-h (pt2 rerun, of3/bt2 with SEP) on top of the existing ~1.4k-model pool [recommended], (b) none, (c) +100 GPU-h af3 seeds with a custom 6A5S template
4. Selection gate: (a) A7 phosphate within 4 A of B K49/R56/R131/Y132 first, then confidence [recommended], (b) confidence only

## Brief

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

## The analysis behind it

Stage **PLAN_REVIEW**. 12 kinds of evidence and 6 decisions, recorded by agent.

Sequence: 280 residues. Given stoichiometry: A1B1.

## What the target is

| | recorded |
|---|---|
| chains | H1388,subunit1 — 11 aa, H1388,subunit2 — 233 aa |
| stoichiometry given by CASP | A1B1 |
| `identity.B.family` | 14-3-3 protein family; Pfam PF00244; InterPro IPR000308, IPR023409, IPR023410, IPR036815 |
| `identity.B.fold` | all-alpha, 9 antiparallel helices; amphipathic groove binds phospho-Ser/Thr peptides; obligate dimer via helices A-D |
| `identity.B.uniprot` | P61981 |
| `identity.A.uniprot` | Q6PKG0 |
| `identity.B.name` | 14-3-3 protein gamma (YWHAG), Homo sapiens |
| `identity.B.hhpred_top` | 4ZQ0_C 14-3-3 Prob 100, 3IQU 14-3-3 sigma, 2BR9 14-3-3 epsilon ... (B/hhpred.hhr) |
| `identity.A.name` | La-related protein 1 (LARP1), Homo sapiens, residues 542-552 |
| `identity.A.context` | LARP1 is an mTORC1 substrate regulating TOP mRNA translation; S548 annotated phosphoserine in UniProt |
| `construct.chains.A (subunit1, 11 aa).note` | No modification in FASTA. Two serines: S5 = LARP1 S546, S7 = LARP1 S548. UniProt annotates S548 as Phosphoserine (modified-residue feature). |
| `construct.chains.B (subunit2, 232 aa).numbering` | input residue i (i>=2) = native i+1 |
| `construct.chains.A (subunit1, 11 aa).seq` | PKGLSASLPDL |
| `construct.chains.A (subunit1, 11 aa).maps_to` | human LARP1 (UniProt Q6PKG0) residues 542-552, exact match |
| `construct.chains.B (subunit2, 232 aa).maps_to` | human 14-3-3 gamma (YWHAG, UniProt P61981, 247 aa); target residues 2-232 = native 3-233 identical |
| `construct.chains.B (subunit2, 232 aa).differences` | target S1 replaces native M1-V2 (likely cloning/protease scar, e.g. remnant of a tag); native C-terminal tail 234-247 (SDQQDDDGGEGNN, acidic, disordered in 14-3-3 crystals) is truncated |
| `construct.given_stoichiometry` | A1B1 |

## The sequence itself

Counted from the dossier's FASTA, so this is arithmetic and not a prediction. A repetitive window or a long run of one residue is where an MSA and a template are most often thin.

| chain | residues | charged D/E/K/R | hydrophobic A/V/I/L/M/F/W/Y | G/P | S/T | Cys | His | longest run | most repetitive 30-mer |
|---|---|---|---|---|---|---|---|---|---|
| H1388,subunit1 | 11 | 18% | 36% | 27% | 18% | 0 | 0 | 1× S | – |
| H1388,subunit2 | 233 | 30% | 41% | 4% | 12% | 3 | 4 | 3× A | 7/30 identical, from residue 64 |

## How much the MSAs cover

| MSA | sequences | ≥50% coverage | columns | median coverage | min coverage | median identity |
|---|---|---|---|---|---|---|
| `af3/uniref90.a3m` | 6,536 | 4,692 | 233 | 4,982 | 456 | 0.627 |
| `af3/uniprot.a3m` | 16,347 | 13,544 | 233 | 13,908 | 1,454 | 0.643 |
| `af3/mgnify.a3m` | 4,332 | 1,860 | 233 | 2,374 | 74 | 0.547 |
| `af3/bfd.a3m` | 1,607 | 957 | 233 | 993 | 34 | 0.305 |


## Templates and public structures

The identity is quoted as the search reported it; an empty cell means the search did not give one.

| entry | identity | search | what it reported |
|---|---|---|---|
| `6a5s` | – | `templates.B` | 23 exact human 14-3-3 gamma PDB structures (see public_structures); best peptide-bound: 6A5S 2.10 A, 2B05 2.55 A, 6GKG, 6FEL. AF3 default template search (A/B msas/af3/templates_hmmsearch.a3m) - check the B one contains gamma entries; the near-identity filter may hide 100% matches, so a custom template of 6A5S/2B05 with peptide pocket is an option |
| – | – | `templates.A` | no template; peptide placement comes from the 14-3-3 groove (binding_site/motif gates) |
| – | – | `templates.source` | RCSB search 2026-11-23 |
| – | – | `public_structures.exact_sequence_sweep` | **method**: RCSB sequence search, chain B, identity>=95%, e<1e-10, 2026-11-23; **n_entities**: 23; **ids**: 2B05 3UZD 4E2E 4J6S 4O46 5D3E 6A5S 6BYJ 6BYL 6BZD 6FEL 6GKF 6GKG 6S9K 6SAD 6Y4K 6Y6B 6ZBT 6ZC9 7A6R 7A6Y 8XI3 9CI3; **note**: human 14-3-3 gamma, many as dimer with a bound phosphopeptide in the amphipathic groove: 2B05 (pSer peptide, 2.55 A), 6A5S (TFEB motif, 2.10 A), 6FEL/6Y4K (CaM…; **source**: RCSB search API; entry titles from data.rcsb.org |
| `9t76` | – | `public_structures.peptide` | **method**: RCSB seqmotif GLSASLPD + full-text 'LARP1 14-3-3'; **hits**: LARP1 present in 9T76/9T77 (LARP1 bound to mTORC1, cryo-EM, released 2026-11-18) and 11HG/11JJ/8XP2/8XP3/9ED0 (large assemblies); none is a 14-3-3:LARP1 comple…; **source**: RCSB search API 2026-11-23 |
| `4zq0` | – | `public_structures.foldseek` | not run: chain B has 23 exact-sequence (100% coverage of 3-233) structures; family fold unambiguous (hhpred top hits 4ZQ0, 3IQU, 2BR9 all Prob 100). |
| – | – | `public_structures.hhpred_caveat` | B/hhpred.hhr top hits are 14-3-3 of other isoforms/species; no gamma entries listed (clustered DB), so templates must be added explicitly if wanted. |

## Every prior, as it was recorded

The sections above are read out of these entries. This is the whole record, including the fields they leave out, each at the stage it was found.

### construct
*NEW, by agent*

- **chains**:
  - **A (subunit1, 11 aa)**:
    - **seq**: PKGLSASLPDL
    - **maps to**: human LARP1 (UniProt Q6PKG0) residues 542-552, exact match
    - **note**: No modification in FASTA. Two serines: S5 = LARP1 S546, S7 = LARP1 S548. UniProt annotates S548 as Phosphoserine (modified-residue feature).
    - **source**: UniProt Q6PKG0 REST json, /store/deepfold/lantern/targets/H1388/dl/Q6PKG0.json
  - **B (subunit2, 232 aa)**:
    - **maps to**: human 14-3-3 gamma (YWHAG, UniProt P61981, 247 aa); target residues 2-232 = native 3-233 identical
    - **differences**: target S1 replaces native M1-V2 (likely cloning/protease scar, e.g. remnant of a tag); native C-terminal tail 234-247 (SDQQDDDGGEGNN, acidic, disordered in 14-3-3 crystals) is truncated
    - **numbering**: input residue i (i>=2) = native i+1
    - **source**: UniProt P61981 REST json, /store/deepfold/lantern/targets/H1388/dl/P61981.json
- **given stoichiometry**: A1B1

### identity
*NEW, by agent*

- **B**:
  - **uniprot**: P61981
  - **name**: 14-3-3 protein gamma (YWHAG), Homo sapiens
  - **family**: 14-3-3 protein family; Pfam PF00244; InterPro IPR000308, IPR023409, IPR023410, IPR036815
  - **fold**: all-alpha, 9 antiparallel helices; amphipathic groove binds phospho-Ser/Thr peptides; obligate dimer via helices A-D
  - **hhpred top**: 4ZQ0_C 14-3-3 Prob 100, 3IQU 14-3-3 sigma, 2BR9 14-3-3 epsilon ... (B/hhpred.hhr)
  - **source**: UniProt P61981; /gpfs/deepfold/casp/casp17/targets/H1388/B/hhpred.hhr
- **A**:
  - **uniprot**: Q6PKG0
  - **name**: La-related protein 1 (LARP1), Homo sapiens, residues 542-552
  - **context**: LARP1 is an mTORC1 substrate regulating TOP mRNA translation; S548 annotated phosphoserine in UniProt
  - **source**: UniProt Q6PKG0

### public_structures
*NEW, by agent*

- **exact sequence sweep**:
  - **method**: RCSB sequence search, chain B, identity>=95%, e<1e-10, 2026-11-23
  - **n entities**: 23
  - **ids**: 2B05 3UZD 4E2E 4J6S 4O46 5D3E 6A5S 6BYJ 6BYL 6BZD 6FEL 6GKF 6GKG 6S9K 6SAD 6Y4K 6Y6B 6ZBT 6ZC9 7A6R 7A6Y 8XI3 9CI3
  - **note**: human 14-3-3 gamma, many as dimer with a bound phosphopeptide in the amphipathic groove: 2B05 (pSer peptide, 2.55 A), 6A5S (TFEB motif, 2.10 A), 6FEL/6Y4K (CaMKK2 motifs), 6GKG (caspase-2), 7A6R (DAPK2), 6BYL (O-GlcNAc-Thr peptide)
  - **source**: RCSB search API; entry titles from data.rcsb.org
- **peptide**:
  - **method**: RCSB seqmotif GLSASLPD + full-text 'LARP1 14-3-3'
  - **hits**: LARP1 present in 9T76/9T77 (LARP1 bound to mTORC1, cryo-EM, released 2026-11-18) and 11HG/11JJ/8XP2/8XP3/9ED0 (large assemblies); none is a 14-3-3:LARP1 complex. No public structure of this complex found.
  - **source**: RCSB search API 2026-11-23
- **foldseek**: not run: chain B has 23 exact-sequence (100% coverage of 3-233) structures; family fold unambiguous (hhpred top hits 4ZQ0, 3IQU, 2BR9 all Prob 100).
- **hhpred caveat**: B/hhpred.hhr top hits are 14-3-3 of other isoforms/species; no gamma entries listed (clustered DB), so templates must be added explicitly if wanted.

### literature
*NEW, by agent*

- **claims**:

| claim | source |
|---|---|
| 14-3-3 proteins bind phosphoserine motifs; mode I consensus R-S-X-pS-X-P, mode II R-X-[Y/F]-X-pS-X-P; the phosphopeptide lies in the amphipathic groove of each monomer in extended conformation | Yaffe et al. 1997 Cell, PMID 9428519 |
| 14-3-3 is a cup-shaped dimer; each monomer has its own peptide-binding groove; dimer interface formed by N-terminal helices | Liu et al. 1995 Nature, PMID 7603574 (14-3-3 zeta crystal structure) |
| The phosphate is coordinated by a conserved basic pocket: zeta K49, R56, R127, Y128 -> in 14-3-3 gamma native K50, R57, R132, Y133 = input chain B K49, R56, R131, Y132 (residue identities checked against FASTA) | Yaffe 1997 PMID 9428519; many 14-3-3 gamma:phosphopeptide structures e.g. 2B05, 6A5S, 6GKG |
| LARP1 S548 is an annotated phosphoserine (large-scale phosphoproteomics); peptide PKGLSApSLPDL fits pS-X-P (S548-L549-P550) with R/K upstream at -5 not -3 (G545 at -3, L544 at -4); S546 (S-A-S) does not fit pS-X-P | UniProt Q6PKG0 modified-residue feature; motif from PMID 9428519 |
| LARP1 is an mTORC1 substrate controlling 5'TOP mRNA translation; recent cryo-EM 9T76/9T77/9T7A/9T7B show LARP1 TOS/XTOS motif bound to mTORC1 (RAPTOR), a different region from 542-552 | RCSB 9T76-9T7B titles, 'mTORC1 recruits substrates via an expanded TOS (XTOS) motif' (2026) |

- **not known**: No paper or structure found on the 14-3-3:LARP1 complex itself (Europe PMC 'LARP1 14-3-3' 50 hits, none on the complex; one web search). The FASTA names no modification; the CASP description text was not available to this step - if it says phosphopeptide, the pSer is S7. Whether one or two peptides bind the 14-3-3 dimer in the experiment is not known; given stoichiometry A1B1.
- **searched**: Europe PMC REST 2026-11-23; WebSearch; target_notes/H1388/papers does not exist

### binding_site
*NEW, by agent*

- **chain**: B
- **residues**: 49, 56, 131, 132
- **what**: phosphate pocket of 14-3-3 gamma (input numbering; native K50 R57 R132 Y133); peptide chain A pS7 expected here, peptide in amphipathic groove
- **source**: Yaffe et al. 1997 PMID 9428519; conserved in 14-3-3 gamma:phosphopeptide crystals 2B05, 6A5S

### motif
*NEW, by agent*

- **chain**: A
- **residues**: 7
- **what**: LARP1 S548 = peptide S7, 14-3-3 mode-I-like pS-X-P (S7-L8-P9); S5 rejected (S-A-S, no Pro at +2)
- **source**: UniProt Q6PKG0 (S548 phosphoserine); PMID 9428519

### chemistry
*NEW, by agent*

- **modification**: phosphoserine (CCD SEP) on chain A residue 7 (LARP1 S548)
- **candidates**:
  - **S7**: chosen: pS-X-P motif, UniProt-annotated phosphosite
  - **S5**: rejected: S-A-S has no Pro at +2; not annotated
- **basis**: 14-3-3 binds phosphorylated motifs only (PMID 9428519); FASTA carries no PTM
- **run as**: variant next to unmodified input

### stoichiometry_evidence
*IDENTIFIED, by agent, 2 entries (latest shown)*

- **given**: A1B1 (CASP; chain A = LARP1 peptide 11 aa, chain B = 14-3-3 gamma 3-233)
- **homolog assemblies**:
  - **method**: RCSB assembly 1 + rcsb_struct_symmetry, 2026-11-23
  - **entries**:
    - **2B05**: tetrameric, A2B2 C2 (PISA)
    - **6A5S**: tetrameric, A2B2 C2 (PISA)
    - **6FEL**: tetrameric, A2B2 C2 (PISA)
    - **7A6R**: tetrameric, A2B2 C2 (PISA)
    - **6GKG**: tetrameric, 14-3-3 A2 + 2 peptides
    - **6BYL**: tetrameric, 14-3-3 A2 + 2 peptides
  - **summary**: 6/6 human 14-3-3 gamma:phosphopeptide structures = 14-3-3 homodimer with one peptide per groove (2:2); no 1:1 assembly
- **family prior**: 14-3-3 obligate homodimer via N-terminal helices (Liu 1995 PMID 7603574); one peptide per monomer groove (Yaffe 1997 PMID 9428519); the dimer interface (helices A-D) is in the construct (residues 3-233 keep the N-terminus)
- **seq2symm**: not run: family assembly unambiguous from 6/6 PDB assemblies
- **n sweep**: not run as a scan; the pertinent hypotheses are A1B1 (scored unit) and A2B2 (native context). A 14-3-3 dimer with a single peptide (A1B2) is a third option
- **interpretation**: The CASP A1B1 is the scored unit (one groove + one peptide), almost certainly an ASU or interface chosen from a 2:2 crystal. Modelling A2B2 and extracting one 14-3-3:peptide pair is the biological environment; groove geometry of the monomer is the same either way, so A1B1 and A2B2 should agree on peptide placement. Use A2B2 as a cross-check / source for alternatives, not as the submitted stoichiometry.
- **hindsight caveat**: results for A1B1 and A2B2 already exist on disk from the team run; not read for this decision

### msa_depth
*PLAN_REVIEW, by agent, 3 entries (latest shown)*

- **af3/uniref90.a3m**:
  - **path**: /gpfs/deepfold/casp/casp17/targets/H1388/B/msas/af3/uniref90.a3m
  - **length**: 233
  - **n**: 6536
  - **n cov50**: 4692
  - **coverage min**: 456
  - **coverage median**: 4982
  - **median identity**: 0.627
  - **thin regions**: none
- **af3/uniprot.a3m**:
  - **path**: /gpfs/deepfold/casp/casp17/targets/H1388/B/msas/af3/uniprot.a3m
  - **length**: 233
  - **n**: 16347
  - **n cov50**: 13544
  - **coverage min**: 1454
  - **coverage median**: 13908
  - **median identity**: 0.643
  - **thin regions**: none
- **af3/mgnify.a3m**:
  - **path**: /gpfs/deepfold/casp/casp17/targets/H1388/B/msas/af3/mgnify.a3m
  - **length**: 233
  - **n**: 4332
  - **n cov50**: 1860
  - **coverage min**: 74
  - **coverage median**: 2374
  - **median identity**: 0.547
  - **thin regions**: none
- **af3/bfd.a3m**:
  - **path**: /gpfs/deepfold/casp/casp17/targets/H1388/B/msas/af3/bfd.a3m
  - **length**: 233
  - **n**: 1607
  - **n cov50**: 957
  - **coverage min**: 34
  - **coverage median**: 993
  - **median identity**: 0.305
  - **thin regions**: none

### templates
*NEW, by agent*

- **B**: 23 exact human 14-3-3 gamma PDB structures (see public_structures); best peptide-bound: 6A5S 2.10 A, 2B05 2.55 A, 6GKG, 6FEL. AF3 default template search (A/B msas/af3/templates_hmmsearch.a3m) - check the B one contains gamma entries; the near-identity filter may hide 100% matches, so a custom template of 6A5S/2B05 with peptide pocket is an option
- **A**: no template; peptide placement comes from the 14-3-3 groove (binding_site/motif gates)
- **source**: RCSB search 2026-11-23

### existing_pool
*HYPOTHESES, by agent*

- **root**: /gpfs/deepfold/casp/casp17/targets/H1388/results
- **found**: existing pool (CASP17 run, 2026-06) for A1B1 and A2B2; input variants af3_msa, af3_msa_pS548, mmseqs2_gpu
- **per combo**: af3: 25 seeds x 5 samples per sto/variant (af3_msa, af3_msa_pS548, mmseqs2_gpu); bt2_jax and of3: af3_msa and mmseqs2_gpu (no pS548 variant); pt2: inference.log ends 'done' but no model files (only ERR/log) in every sto/variant
- **templates B**: af3 default hmmsearch: 1578 hits, 91 are 14-3-3 gamma, so close templates are present without a custom template
- **msa A**: peptide (11 aa): uniref90/uniprot/mgnify contain only the query (expected for a peptide); A templates_hmmsearch 34 lines, irrelevant

### report
*PLAN_REVIEW, by agent*

- **pdf**: /store/deepfold/lantern/targets/H1388/report/report.pdf
- **with retro**: False
- **figures**: template_6a5s.jpg

## Decisions and the priors they rest on

| id | stage | type | chose | rests on | confidence | rejected |
|---|---|---|---|---|---|---|
| D1 | NEW | variant | af3 input with SEP at chain A position 7 (LARP1 pS548), run next to the unmodified input | `construct`, `literature`, `motif`, `chemistry` | medium | no modification only: kept as baseline control, 14-3-3 binding requires the phosphate |
| D2 | IDENTIFIED | stoichiometry | A1B1 submitted (given; one 14-3-3 gamma groove + LARP1 pS548 peptide). Run A1B1 and A2B2 (native 14-3-3 dimer, a peptide in each groove); A2B2 models are cut to | `construct`, `public_structures`, `literature`, `stoichiometry_evidence` | high | A1B2 (dimer, one peptide): rejected as primary, no precedent in 6/6 assemblies; low-value  |
| D3 | HYPOTHESES | variant | MSA: use the existing deep chain-B MSAs (af3_msa: uniref90 N=6536, uniprot N=16347; plus mmseqs2_gpu variant). No crawl/plmMSA. Peptide A runs single-sequence ( | `construct`, `msa_depth`, `existing_pool` | high | family crawl / plmMSA for B (MSA already deep); paired MSA for LARP1 peptide (meaningless for 11 aa) |
| D4 | HYPOTHESES | variant | Templates: AF3 default template search (91 of 1578 hits are 14-3-3 gamma, peptide-bound gamma included); no custom template now. Fallback if peptide register/gr | `public_structures`, `templates`, `existing_pool` | medium | custom 6A5S/2B05 template now; MODELLER assembly (dimer is easy for predictors) |
| D5 | HYPOTHESES | variant | No ligand or metal: 14-3-3 binds the phosphopeptide without cofactors. Only chemistry variant is pS548 SEP (D1) | `literature`, `chemistry` | high | add free phosphate/ions (no evidence) |
| D6 | HYPOTHESES | sampling | No new GPU now: CASP17 pool already holds af3 25 seeds x5 for A1B1 and A2B2 x (af3_msa, af3_msa_pS548, mmseqs2_gpu) plus bt2_jax and of3 (unmodified), ~1.4k mod | `binding_site`, `motif`, `stoichiometry_evidence`, `existing_pool` | medium | fresh 100+ seed sweep (pool already large for an 11-aa peptide on rigid 14-3-3); drop pt2 entirely |

## Prior → decision map

Each column is a decision; ● means the decision cites that prior.

| prior | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| `construct` | ● | ● | ● |  |  |  |
| `identity` |  |  |  |  |  |  |
| `public_structures` |  | ● |  | ● |  |  |
| `literature` | ● | ● |  |  | ● |  |
| `binding_site` |  |  |  |  |  | ● |
| `motif` | ● |  |  |  |  | ● |
| `chemistry` | ● |  |  |  | ● |  |
| `stoichiometry_evidence` |  | ● |  |  |  | ● |
| `msa_depth` |  |  | ● |  |  |  |
| `templates` |  |  |  | ● |  |  |
| `existing_pool` |  |  | ● | ● |  | ● |
| `report` |  |  |  |  |  |  |
| a person's note |  |  |  |  |  |  |

Found but not cited by any decision: `identity`, `report`.

## Between the agent and a person

- 08:20 **person**: dossier created
- 08:20 **person**: OPERATOR: this run is an example of the pipeline from the start, so do the analysis yourself. The team's own submission for this target is on disk at /gpfs/deepfold/casp/casp17/targets/H1388/submit/ - do not read it, and do not read any native structure, until RETROSPECTIVE. Everything else under targets/H1388 (target.fasta, inputs/, results/, msas/) is yours to use, as is the literature. Record w
- 08:21 **agent**: FYI: replay hindsight - the lantern-intake skill quotes H2388 (14-3-3 gamma + LARP1 phosphopeptide PKGLSASLPDL, SEP at peptide position 7) which looks like the same complex as H1388. Those outcome lines are hindsight; decisions here cite evidence collected in this step.
- 08:31 **agent**: FYI: lantern-consult skill quotes a CASP17 'H2388' phosphopeptide case (pS-X-P picked S7, DockQ 0.24->0.82) that matches this target's motif; if this is a replay, that line is hindsight and is not used to drive D1 - D1 rests on the motif/chemistry/literature evidence. Template 6A5S downloaded to /store/deepfold/lantern/targets/H1388/dl/pdb/6A5S.cif (928 kB) for the plan slide.
- 08:32 **agent**: FYI: msa_depth re-recorded in lantern msa --depth shape (chain B af3 MSAs; same numbers) because the per-chain summary shape broke lantern slides; peptide A is query-only (11 aa), see existing_pool.msa_A.

## References

What the analysis read, and which priors rest on it. A prior whose source is not written down cannot be checked against the paper later, and `lantern retro` cannot score it.

| # | source | supports |
|---|---|---|
| 1 | UniProt Q6PKG0 REST json, /store/deepfold/lantern/targets/H1388/dl/Q6PKG0.json | `construct` |
| 2 | UniProt P61981 REST json, /store/deepfold/lantern/targets/H1388/dl/P61981.json | `construct` |
| 3 | UniProt P61981; /gpfs/deepfold/casp/casp17/targets/H1388/B/hhpred.hhr | `identity` |
| 4 | UniProt Q6PKG0 | `identity` |
| 5 | RCSB sequence search, chain B, identity>=95%, e<1e-10, 2026-11-23 | `public_structures` |
| 6 | RCSB search API; entry titles from data.rcsb.org | `public_structures` |
| 7 | RCSB seqmotif GLSASLPD + full-text 'LARP1 14-3-3' | `public_structures` |
| 8 | RCSB search API 2026-11-23 | `public_structures` |
| 9 | Yaffe et al. 1997 PMID 9428519; conserved in 14-3-3 gamma:phosphopeptide crystals 2B05, 6A5S | `literature`, `binding_site`, `motif`, `chemistry` |
| 10 | 14-3-3 obligate homodimer via N-terminal helices (Liu 1995 PMID 7603574); one peptide per monomer groove (Yaffe 1997 PMID 9428519); the dimer interface (helices A-D) is in the construct (residues 3-233 keep the N-terminus) | `literature`, `stoichiometry_evidence` |
| 11 | RCSB 9T76-9T7B titles, 'mTORC1 recruits substrates via an expanded TOS (XTOS) motif' (2026) | `literature` |
| 12 | RCSB assembly 1 + rcsb_struct_symmetry, 2026-11-23 | `stoichiometry_evidence` |
| 13 | RCSB search 2026-11-23 | `templates` |

**No source recorded**: `existing_pool`. Say where each came from (a paper, a structure, an experiment, or a calculation run here).
