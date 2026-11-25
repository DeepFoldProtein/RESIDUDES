# H1388: human 14-3-3 gamma with a LARP1 peptide

## 1. Identification

| field | value |
|---|---|
| CASP id | H1388 |
| organisers' description | not in the dossier (only the FASTA was registered) |
| declared stoichiometry | A1B1 (dossier `stoichiometry.given`) |
| chain A | 11 aa, `PKGLSASLPDL`, human LARP1 (Q6PKG0) 542–552 |
| chain B | 233 aa, human 14-3-3 gamma (P61981, YWHAG) 3–234 with an N-terminal S |
| method | not known |
| dossier created | 2026-11-26 |

## 2. The chains

### Chain A: LARP1 542–552
- It is an exact, single-occurrence substring of Q6PKG0 at 542–552, as counted here from the UniProt FASTA. `input = native − 541`.
- It has two serines. S5 is native S546. S7 is native S548, which UniProt Q6PKG0 records as a phosphoserine.
- It is a linker segment with no Pfam domain. There is no public structure of it (`residudes structures`, 0 entities at ≥ 90 % identity).

### Chain B: 14-3-3 gamma
- Input 2–233 is identical to P61981 3–234, so `input = native − 1`.
- Input S1 stands in for native M1–V2. I infer it is a cleavage or cloning remnant; that is not sourced.
- The C-terminal tail, native 235–247 (`SDQQDDDGGEGNN`), is absent.
- The whole construct is the 14-3-3 domain: Pfam PF00244; InterPro IPR000308, IPR023410, IPR036815 and IPR023409 (UniProt cross-references).
- The phosphoserine-contact residues, in input numbering, are K49, R56, R131 and Y132 (native K50, R57, R132, Y133). I checked each identity against the FASTA.

## 3. The biology that bears on the structure

14-3-3 proteins bind phosphoserine or phosphothreonine motifs. Peptide libraries gave two consensuses, RSXpSXP (mode 1) and RXY/FXpSXP (mode 2). The bound peptide lies extended in the amphipathic groove, with a turn at the pS+2 proline [1]. 14-3-3 is a dimer, with one groove per protomer, and the dimer can bind two motifs at once [1]. Structures exist for all seven human isoforms bound to ligands [2].

In this peptide only S7 is followed by X-Pro (S7-L8-P9), which is the pS-X-P core of both motifs. There is no Arg at −3 or −4 (L4, G3), so the peptide fits neither full consensus; there is a Lys at −5 (as counted here). UniProt records Ser548 (= S7) as phosphorylated [3].

I found no paper on the 14-3-3–LARP1 pS548 interaction itself.

## 4. Sites and chemistry (input numbering)

| what | where | becomes | source |
|---|---|---|---|
| phosphoserine, CCD SEP | A7 (LARP1 S548); S5 rejected, as it has no +2 Pro | an af3 input **variant**, next to the unmodified input | [1], [3] |
| motif pS-X-P | A7–A9 | motif figure | [1] |
| phosphate pocket | B49, B56, B131, B132 | **contact gate**: A7 to B56/B131/B132 | UniProt P61981 sites; 6S9K check below |

The pocket check, as counted here in 6S9K (14-3-3 gamma with the caspase-2 phosphopeptide, 1.60 Å): the SEP139 phosphate oxygens lie within 3.5 Å of R57, R132 and Y133 (native numbering). K50 is not within 3.5 Å in that entry.

## 5. What is uncertain
- **Is the peptide phosphorylated in the experiment?** The FASTA carries no modification, and I do not have the organisers' description. The plan answers this by running both the SEP-A7 and the unmodified input and gating on the pocket contact.
- **Copy number.** 14-3-3 is a dimer in the family [1], but A1B1 is declared. Plan: model A1B1 as declared, and possibly A2B2 as a check of the groove geometry. That is a decision for the plan checkpoint.
- **Peptide register and direction in the groove.** The pose is set by pS7 and the +2 Pro turn. Without a basic −3/−4 residue the register is less constrained than for a canonical mode-1 peptide.

## 6. The structures that exist

Chain B has 23 PDB entities at 100 % identity and 100 % coverage (`residudes structures H1388`, 2026-11-26). Chain A has none. The local PDB mmCIF mirror is from 2026-04-25 (215 days old).

| PDB | method | res. (Å) | state | template use |
|---|---|---|---|---|
| 6S9K | X-ray | 1.60 | caspase-2 phosphopeptide | yes, best peptide-bound; pocket checked |
| 6ZBT / 6ZC9 | X-ray | 1.80 / 1.90 | Nedd4-2 peptide | yes |
| 3UZD | X-ray | 1.86 | 14-3-3 gamma | yes (receptor) |
| 6A5S | X-ray | 2.10 | TFEB peptide | yes |
| 4E2E | X-ray | 2.25 | TH/TPH-activator complex | receptor |
| 7A6Y / 7A6R | X-ray | 2.50 / 2.70 | DAPK2 peptide | yes |
| 2B05 | X-ray | 2.55 | phosphopeptide | yes |
| 6GKF / 6GKG | X-ray | 2.60 / 2.85 | caspase-2 | yes |
| 6BZD | X-ray | 2.67 | R57E mutant, GlcNAc peptide | no: the pocket is mutated |
| 5D3E | X-ray | 2.75 | CFTR peptide | yes |
| 6SAD | X-ray | 2.75 | doubly phosphorylated peptide | yes |
| 6FEL | X-ray | 2.84 | CaMKK2 peptide | yes |
| 4O46 | X-ray | 2.90 | influenza NS1 tail | yes |
| 6BYJ / 6BYL | X-ray | 2.90 / 3.35 | O-GlcNAc peptides | a different chemistry |
| 6Y4K / 6Y6B | X-ray | 3.00 / 3.08 | CaMKK2 | yes |
| 8XI3 | EM | 3.00 | mouse SCMC complex | no |
| 4J6S | X-ray | 3.08 | TH N-terminal peptide | yes |
| 9CI3 | EM | 3.96 | LRRK2 complex | no |

The ColabFold template hits for chain B agree with this (6byj_A 100 %, qcov 2–233, E 1.5e-116; `msas/colabfold/msa/templates_B.m8`). AF3's own search hides these near-identical entries, so a peptide-bound entry has to be passed in explicitly. I did not run HHsearch: chain B already has exact-sequence structures, and an 11-mer cannot be placed by a profile search.

## 7. MSA depth and figures

These come from `residudes-engine msa --source colabfold`, then `residudes msa H1388 --depth`:

| chain | N | N cov ≥ 50 % | median coverage | thin regions |
|---|---|---|---|---|
| A | 0 | 0 | 0 | the whole peptide (expected for an 11-mer) |
| B | 10,377 | 8,105 | 7,660 | none |

There is 1 paired row. No deepening is needed.

Figure: `report/figs/msa_coverage.svg`, the per-residue coverage drawn by `residudes msa --depth` from `msas/colabfold/msa/unpaired_{A,B}.a3m`.

## 8. References
1. Yaffe MB et al. (1997) The structural basis for 14-3-3:phosphopeptide binding specificity. *Cell*. PMID 9428519. I read the abstract only, through the Europe PMC REST API; the full text would give the zeta pocket residues explicitly.
2. Yang X et al. (2006) Structural basis for protein-protein interactions in the 14-3-3 protein family. *PNAS*. PMID 17085597, PMC1859916. I read the abstract only.
3. UniProt P61981 (1433G_HUMAN) and Q6PKG0 (LARP1_HUMAN), rest.uniprot.org JSON and FASTA, read 2026-11-26. The local copies are `<WORK>/targets/H1388/dl/P61981.fasta` and `dl/Q6PKG0.fasta`.
4. PDB 6S9K, `<WORK>/targets/H1388/dl/pdb/6S9K.pdb`, for the pocket contacts counted here.
5. I found nothing on the 14-3-3–LARP1 pS548 interaction in Europe PMC. The team folder `/data/casp17/target_notes/H1388/papers/` does not exist.
