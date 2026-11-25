# H1388 plan review

Deck: `<WORK>/targets/H1388/report/report.pdf`. My own slide is the last one: the 6S9K pocket.

## 1. The target in three lines
- **Chain A.** An 11-mer, `PKGLSASLPDL`: human LARP1 542–552 (Q6PKG0), with `input = native − 541`.
- **Chain B.** Human 14-3-3 gamma (P61981 3–234) with an N-terminal S. The C-terminal tail 235–247 is absent.
- **Stoichiometry.** A1B1, as declared.
- **The question.** This is a phosphopeptide-in-groove problem. Everything turns on the peptide's register and direction in the amphipathic groove.

## 2. What we know
- **Chemistry** (`chemistry`, `motif`).
  - UniProt records LARP1 S548 (= A7) as a phosphoserine.
  - A7 is the only serine followed by X-Pro: S7-L8-P9, the pS-X-P core of the 14-3-3 motifs (Yaffe 1997, PMID 9428519).
  - I rejected S5 (no +2 Pro).
  - The FASTA carries no modification, and I do not have the organisers' description.
- **Literature** (`literature`).
  - The 14-3-3 groove binds pS/pT peptides in an extended conformation, with a turn at the +2 Pro (PMID 9428519).
  - Structures exist for all seven human isoforms (Yang 2006, PMID 17085597).
  - **Not known:** I found no paper on 14-3-3 and LARP1 pS548. The peptide lacks the Arg at −3/−4 of mode 1 and mode 2. I read the abstracts only, and there are no team PDFs.
- **Public structures** (`public_structures`, `templates`, `peptide_analog`).
  - Chain B has 23 PDB entities at 100 % identity. Chain A has none.
  - The closest analog is **6S9K** (1.60 Å), with the caspase-2 peptide *DL-pS-L-P-F*. It shares pS, +1 Leu and +2 Pro with LARP1's *SA-pS-L-P-D*. This is a strong register prior.
- **MSA** (`msa_depth`). Chain B has 10,377 sequences, which is deep. Chain A is single-sequence, as expected for an 11-mer. No deepening is needed.
- **Site** (`binding_site`).
  - The phosphate pocket is B49, B56, B131 and B132 in input numbering (native K50, R57, R132, Y133).
  - In 6S9K the phosphate is within 3.5 Å of R57, R132 and Y133, but not of K50.
- **Stoichiometry** (`stoichiometry_evidence`). 14-3-3 is a constitutive dimer, but A1B1 is declared.

## 3. What is uncertain, and how the plan covers it
- **Whether the peptide was phosphorylated in the experiment.** The plan runs SEP-A7 and unmodified af3 arms side by side (D1, D5), and the gate decides between them.
- **Register and direction in the groove.** Without a −3/−4 Arg the pose is less constrained. The plan covers this with the pocket gate, the 6S9K pinned template (D4), and more seeds only if the results split.
- **Dimer.** The declared A1B1 is the scored unit, so no A2B2 arm is planned (see Q2).

## 4. The plan (D1–D6)
- **Stoichiometry.** A1B1 only (D2).
- **Inputs.**
  - MSA: ColabFold for B; A single-sequence (D3).
  - Templates: B search templates plus 6S9K chain A pinned at rank 1; no multimer template (D4).
  - PTM: SEP at A7 for af3 and Protenix; for Boltz-2 and OF3 only if their converters carry it (D5).
- **Seeds** (D6), about 300 models in all:

  | arm | seeds × samples |
  |---|---|
  | af3 + SEP (lead) | 20 × 5 |
  | af3 unmodified (control) | 10 × 5 |
  | Protenix + SEP | 10 × 5 |
  | Boltz-2 | 10 × 5 |
  | OF3 | 10 × 5 |

- **GPU.** About 5 GPU-h, capped at 20 of the 400 h budget.
- **QA and gates.**
  - The pocket gate: the pS7 phosphate within 4 Å of B56, B131 and B132.
  - Then interface QA, and confidence ranked within each predictor.

## 5. Questions
- **Q1: Pocket gate residues.**
  - (a) B56, B131 and B132, which is what 6S9K actually contacts. **[recommended]**
  - (b) All four, including B49 (K50).

  B49 is not within 3.5 Å in 6S9K, so requiring it would reject correct poses. This changes the gate spec only.
- **Q2: Dimer.**
  - (a) A1B1 only, as declared. **[recommended]**
  - (b) Add an A2B2 arm of 5 af3 seeds as a check of the groove geometry, at about +1 GPU-h. It would not be submitted.
- **Q3: Pinned 6S9K template.**
  - (a) Pin it in the af3 lead arm, with the unpinned af3 control. **[recommended]**
  - (b) Run no pinned template.
- **Q4: Unmodified arm in the slate.**
  - (a) Unmodified models can enter the slate only if they pass the pocket gate. **[recommended]**
  - (b) Use SEP models only.

## 6. The default
If no answer comes by the deadline, I take every recommended option, proceed to INPUTS_READY, and run D6 as written.
