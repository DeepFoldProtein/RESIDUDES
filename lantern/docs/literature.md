# Background reading: what the literature gives a target

A CASP target arrives as a FASTA file and one line of description. Everything else that decides the prediction —
where the partner binds, how many copies there are, which chemistry the sequence does not carry, which published
structure is the right template, whether the loop in question moves — is in the papers. In CASP17 the team read
them target by target, and that reading is what the gates and the input variants were built from.

This page says what to search, what to bring back, and how a finding becomes something lantern can act on. The
stage is NEW → IDENTIFIED (skill `lantern-intake`); the reading is discussed with a person at PLAN_REVIEW
(`lantern-consult`).

## What it was worth in CASP17

| target | what the reading gave | what it changed |
|---|---|---|
| **H2319** | the antibody's epitope from escape selection, and the current apo structures of the antigen | model 1 DockQ **0.93** against 0.67 for the most confident model in the pool (see below) |
| **H2321** | the sibling antibody's escape mutants (S611/R615) | the gate narrowed 12,810 models to 246; the submitted model scored 0.85 against 0.59 for the top ranking_score, but it fails the core gate and the record of how it was picked is missing ([`casp17_cases.md`](casp17_cases.md)) |
| **M2415** | which pocket the splicing modifier binds (the −1A bulge of the 5′ss:U1 duplex, 8R62/8R63) | the ligand landed 8.8 Å from its site, against 51.8 Å for the most confident pose |
| **H2388** | the 14-3-3 recognition motif R-S-X-pS-X-P, which picks one of the peptide's two serines | the phosphoserine went into the input: AF3 median DockQ 0.24 → **0.82** |
| **M1439** | the homing endonuclease's recognition sequence, found at nt 20–30 of the intron | recorded, but no gate used it; model 1 DockQ 0.03. The miss is why `lantern review` now flags a site prior with no gate |
| **T1331** | the family (a terpene synthase, so the isoprenoid synthase superfamily) | the family MSA crawl: AF3 median lDDT 0.33 → **0.70** |
| **T1416** | no paper on this protein; the nearest characterised relative is a *B. cereus* CotH kinase (5JD9/5JDA, 13 % identity) | said plainly in the plan: the fold is a prediction, not a homology model |
| **E2450** | the organisers' UniProt id and template were both wrong; the paper named the real constructs | the whole target was modelled from the right sequences |

Each row is one target and is here because the reading changed the outcome; the set is not a sample of every
target where the literature was read, and there is no count of the times it made no difference. The row that
matters most is M1439, where the prior was found and then not used.

## H2319, worked through

An antibody–antigen complex: a divergent astrovirus capsid spike plus the Fv of a neutralising antibody, with the
copy number not declared. Four things came out of the reading, and each one became a different kind of input.

| from the papers | recorded as | what it drove |
|---|---|---|
| The antibody's epitope: escape selection put it on one surface loop, around two residues (Q542/P543), and the epitope is conformational (the antibody does not bind the denatured protein) | `epitope`, with the paper as its source | the `contact` gate: a CDR atom within 6 Å of those residues |
| The antigen's own crystal structures are recent and were **not** in the default template database (an apo spike at 1.46 Å, and a second one from the other clade) | `public_structures` | the template search was redone: af3's median DockQ went from 0.24 to **0.73** |
| Published complexes exist for the *classical* members of this virus family, but this clade's spike differs by about 3.7 Å RMSD | `public_structures`, as a rejected option | they were **not** used as docking templates; the epitope from the escape data was the usable prior instead |
| On the mature particle the antigen is presented as a dimer, and the antibody binds intact particles | `stoichiometry_evidence` | the complex was run as the dimer with two Fvs, although CASP declared the stoichiometry unknown |

The gates then ran in the order the evidence supported, on 42,373 models:

| gate | from | models left |
|---|---|---|
| spike core within 1.0 Å of the apo crystal | the structure the reading found | 11,185 (all medium or better) |
| a CDR atom within 6 Å of the escape-mutant site | the antibody paper | 6,542 |
| no inter-chain clash | sanity | 6,349 |

Model 1 was the highest rubric score among those: **DockQ 0.93**. The most confident model in the pool scored 0.67,
af3's own most confident 0.71, and the best model in the pool 0.94. Without the reading, neither the template nor
the gate exists, and the pick is the 0.67 one.

As a lantern spec, each gate says in its note which reading put it there (`lantern gate T --spec …`):

```json
{"gates": [
  {"name": "core", "type": "rmsd", "select": "A:27-520,B:27-520", "ref": "8ufo.cif", "max": 1.0,
   "note": "the apo spike crystal the literature search found; same sequence, so the protomer should match it"},
  {"name": "spike_dimer", "type": "interface", "a": "A", "b": "B", "ref": "8ufo.cif", "min_fraction": 0.6,
   "note": "the antigen is presented as a dimer on the mature particle: the dimer interface must survive"},
  {"name": "epitope", "type": "contact", "site": "A:542,543", "partner": "C,D", "within": 6.0,
   "note": "source: escape selection against the antibody (its own paper) puts the epitope here"},
  {"name": "vh_vl", "type": "interface", "a": "C", "b": "D", "min_contacts": 25,
   "note": "an Fv is one paired module; a model that unpairs it is wrong"}]}
```

## What to search

1. **The construct itself.** The paper behind the sequence, if there is one: what was expressed, which residues, which
   tag, which mutations. Organiser metadata can be wrong (E2450), so check it against the paper.
2. **Each partner, separately.** For a complex, the antigen and the antibody are different literatures. An antibody's
   own paper carries the epitope, the affinity and whether the epitope is conformational.
3. **The paper behind every public structure you found.** A PDB entry says what was in the crystal; its paper says
   which state that is, what is missing and why, and whether the biological assembly is the one in the file.
4. **The family.** Needed for the MSA crawl (`lantern msa --crawl --family PF…`) and for the oligomeric state. The
   family's characterised members are the homologs worth a template.
5. **The experiment the target came from.** The method (X-ray, cryo-EM, NMR, SAXS) decides what the native will be:
   a crystal contact is not an interface, an ensemble target is not one structure.
6. **Nucleic-acid chains.** The recognition sequence, the secondary structure, the modification. A protein-sequence
   search will not find any of it.

## What to bring back

Read with the dossier's keys in mind. A finding that is not one of these is background; a finding that is one of
these changes a command.

| what a paper gives | evidence key | what it drives |
|---|---|---|
| the residues a partner binds: escape mutants, mutagenesis, a crystal contact | `epitope`, `paratope`, `binding_site`, `pocket`, `recognition_site`, `interface_prior`, `active_site`, `motif` (the keys `gate --init` reads) | a `contact` gate and the site census (`lantern gate`) |
| a structure of the target or a close homolog, and which state it is | `public_structures`, `templates` | the af3 template variant; an `rmsd` gate for a part that should be rigid |
| the oligomeric state: on the particle, in solution, in the family | `stoichiometry_evidence` | the copy numbers run, and the hedge |
| chemistry the FASTA does not carry: a modification, a ligand, a metal, a cofactor | evidence (the residue you chose, the candidates you rejected, and the motif or paper that picked it), then the `lantern decide --type variant` that adds it | an input variant (`SEP`, `ccd:ZN`, …) |
| the family, for a thin MSA | `msa_crawl` | `lantern msa --crawl --family PF…` |
| the affinity, and whether binding is conformational | `literature` | whether one state is enough, and how much to hedge |
| what is **not** known: no paper, no structure, an uncharacterised family | `literature` | said plainly in the plan brief, so the person knows the confidence is thin |

## How to record it

```sh
lantern evidence T literature '{"epitope": "escape selection put the antibody on Q542/P543",
                                "source": "J Virol 2024, PMC11323900", "affinity": "KD 0.4 nM",
                                "note": "conformational epitope: no binding to the denatured protein"}'
lantern evidence T epitope '{"select": "A:542,543", "source": "escape mutants, J Virol 2024 (PMC11323900)"}'
```

- **Every site carries its source.** `lantern gate --init` copies it into the drafted gate's note, so whoever checks
  the gate can check the number; `lantern review` flags `site_without_source`. A source is a `source`-like field, or a
  citation in the text (a PMID, a PMC id, a DOI, "et al", a PDB id).
- **Convert the numbering.** Papers number the native protein, or its precursor; the gate needs the input sequence's
  numbering. Say in the evidence which numbering you converted from, and check one residue's identity against the
  FASTA (`lantern gate --init` does this check when it has to guess the chain).
- **Say when a prior is from a relative.** "The same site class on another serotype" is weaker than the target's own
  escape data, and the plan brief should say which one it is.
- **Absence is not evidence.** No paper, no precedent and no homolog do not argue for a monomer or against a
  stoichiometry (`lantern-stoichiometry`). Several predictors agreeing on a copy number does argue for it.
- **Do not let a paper overrule the models silently.** If the reading and the pool disagree, that is a results
  checkpoint (`lantern consult --kind results`), not a quiet choice.

## Where it shows up later

- `lantern report T` prints each decision next to the evidence keys it cites, so the plan shows which paper reached
  which choice.
- `lantern gate --init` drafts a gate per site, with its source in the note.
- `lantern review T` flags a site prior that no gate uses (`site_prior_without_gate`), and one whose source is not
  recorded (`site_without_source`).
- `lantern retro T` scores each decision against the native, so a prior that was wrong is a lesson and not a habit.

See also: [the gate reference](gates.md), [what CASP17 taught lantern](casp17_cases.md), and the
`lantern-intake` skill.
