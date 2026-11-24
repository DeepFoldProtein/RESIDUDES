---
name: residudes-intake
description: Identify a new target and design its inputs - construct mapping, identity, the literature on the target and its partners (the site a partner binds, the chemistry, the oligomeric state, each with its source), public structures (as multimer templates), MSA depth (family crawl, plmMSA, RNA homologs), chemistry the FASTA does not carry (PTMs, ligands, metals), sites that become selection gates, and template/PTM/ligand variants with residudes-engine. Use at dossier stages NEW and HYPOTHESES.
---

# Intake and input design

Record every finding with `residudes evidence T <key> '<json>'`. Record every choice with `residudes decide`.

## Identify (stage NEW → IDENTIFIED)

`residudes report` builds the front of the analysis document out of the fields below, so a finding recorded under
one of these names appears there as a row; the same finding written into free prose reaches the appendix only.
Record them per chain, not per target: a complex needs the organism, the family and the construct of each subunit.

| evidence key | fields the report reads | where it appears |
|---|---|---|
| `identity` | `organism` (and its lineage), `family` / `fold` / `domains` with InterPro or Pfam ids and the residue range of each domain, `uniprot`, `basis` | "What the target is", and the chain map |
| `construct` | `length`, `tag`, `native_region`, `mutations`, `n_term`, `notes` | the same table |
| `literature` | a list of `{claim, source}`, plus `not_known` and `searched` | "What the literature says, claim by claim", and its figure |
| `binding_site`, `epitope`, `paratope`, `pocket`, `active_site`, `motif`, … | `chain` and `residues` (the numbering a gate uses), `what`, `source` | "The sites the analysis found", its figure, and the chain map |
| `chemistry` | the modification and the residue it sits on ("phosphoserine (CCD SEP) on chain A residue 7"), `candidates` as `{residue: "chosen/rejected: why"}` | the sites figure (a ✱ on the residue) and the motif figure |
| `msa`, `msa_depth` | `depth` per database, and per file `n`, `n_cov50`, `coverage_median`; plus `effective` and `assessment` | "How much the MSAs cover" |
| `templates`, `public_structures` | one field per search, each naming the entry and the identity as the search gave it ("6w3w_A 37 % id, qcov 99/122, E 6e-31") | "Templates and public structures" |

Two sections need nothing recorded, because they are computed here: the sequence itself (composition, the longest
run of one residue, the most repetitive window) is counted from the FASTA, and **every motif written anywhere in
the evidence is scanned against the sequence** — "pS-X-P" becomes a figure showing each place it fits, which of
them carries the residue you recorded, and whether a consensus from a paper fits nowhere at all. So write the
pattern as a pattern (`pS-X-P`, `R-X-[Y/F]-X-pS-X-P`), not as prose, and record the residue you chose in
`residues`: the figure then either agrees with you or shows that it does not.

The pool, once there is one, has its own section and figure from `residudes qa T --what cluster` (clusters, their
sizes, the best confidence and predicted DockQ in each). Nothing to record: it is read from residudes-qa's output.

1. **Construct.** Map the given sequence to the native protein: tags, signal peptides, truncations, mutations.
   - Organiser metadata can be wrong: E2450 had the wrong UniProt id and template. Check them yourself.
2. **Identity.** Find an exact UniProt substring match, then the family (InterPro / Pfam). For antibodies, note germline and CDRs.
   - For each chain, record what is known about it from public structures and sequences, with the source (PDB id,
     paper). M1439: the protein was 97.6 % identical to I-AniI (3EH8) plus a 23-aa N-terminal extension; I-AniI is a
     LAGLIDADG homing endonuclease and group I intron splicing factor (1P8K); the template hits with nucleic acid
     were DNA homing sites only (1P8K, 3EH8, 3UVF); its RNA-binding relative bI3 maturase had no RNA in its structure
     (2AB5); a sequence search found the I-AniI recognition core (UGAGGAGGUUU) at nt 20–30 of the intron.
3. **Read the literature on the target and on each partner.** This is where the gates and the variants come from, and
   it is the step that paid most in CASP17. Reference page: [`docs/literature.md`](../../docs/literature.md).
   - **What to search:** the construct's own paper; each partner separately (for a complex, the antigen and the
     antibody are different literatures); the paper behind every public structure you found (it says which state the
     file is, and whether its biological assembly is the one deposited); the family; the experimental method the
     target came from; for nucleic-acid chains, the recognition sequence and secondary structure.
   - **What to bring back** — read with these keys in mind, because each one changes a later command:
     the residues a partner binds (`epitope`, `paratope`, `binding_site`, `pocket`, `recognition_site`,
     `interface_prior`, `active_site`, `motif` — the keys `gate --init` reads → a gate);
     a structure and which state it is (`public_structures`, `templates` → the af3 template variant, an `rmsd` gate);
     the oligomeric state on the particle, in solution or in the family (`stoichiometry_evidence` → the copy numbers);
     chemistry the FASTA does not carry (→ an input variant); the family for a thin MSA (`msa_crawl`); the affinity
     and whether binding is conformational; and **what is not known** — no paper, an uncharacterised family — which
     belongs in the plan brief so the person knows the confidence is thin (T1416: no paper on this protein, nearest
     characterised relative 13 % identical).
   - **H2319** is the worked case: the antibody's epitope came from escape selection in its own paper (Q542/P543) and
     became the `contact` gate; the antigen's recent apo crystals were missing from the default template database, and
     redoing the search took af3's median DockQ from 0.24 to 0.73; published complexes of the *classical* members of
     the family were rejected as docking templates (≈3.7 Å RMSD from this clade); and the particle literature said the
     antigen is presented as a dimer, so it was run as a dimer with two Fvs although CASP declared the stoichiometry
     unknown. Model 1 scored DockQ 0.93, against 0.67 for the pool's most confident model.
   - **Record it** as `literature` evidence, with a `source` per claim, and give **every site its own source**
     (a PMID, PMC id, DOI, "et al", or a PDB id). `residudes gate --init` copies the source into the gate's note and
     `residudes review` flags `site_without_source`: a gate whose source is written down can be checked against it, and
     `residudes retro` can score the prior afterwards.
   - **Convert the numbering.** Papers number the native protein or its precursor; gates need the input sequence's
     numbering. Say which numbering you converted from, and check one residue's identity against the FASTA.
   - **How to find and read them**, from this cluster (all four endpoints answer; checked 2026-11-22):
     ```sh
     curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=<terms>&format=json&pageSize=10"
     curl -s "https://europepmc.org/articles/<PMCID>?pdf=render" -o paper.pdf   # open-access full text
     curl -s "https://data.rcsb.org/rest/v1/core/entry/<ID>"                    # the entry, its paper, the assembly
     curl -s "https://rest.uniprot.org/uniprotkb/<ACC>.json"                    # annotations, sites, references
     ```
     Then `pdftotext paper.pdf -` and read it. Web search and fetch, where the harness has them, are for the rest.
     **Expect to be refused.** A publisher's site answers an automated fetch with 403 (H2338: the paper naming the
     antibody's target domain, at the journal and through Europe PMC alike), and PMC can serve a browser challenge
     instead of the article. In that order: the Europe PMC REST full text
     (`…/rest/<PMCID>/fullTextXML`, which is not the blocked web page), the abstract from the search result, an
     Internet Archive snapshot of the PDF (`archive.org/wayback/available?url=…`, then the `if_/` form of the
     snapshot URL), the team's own folder, or ask the person for the PDF at the plan checkpoint. **Never fill the
     gap from memory**: record what you could not read, and what it would have told you.
     The team's own PDFs per target are on this cluster at
     `/gpfs/deepfold/casp/casp17/target_notes/<T>/papers/` (an absolute path; there is no `target_notes` inside the
     dossier). Look there before downloading anything, and say which file a claim came from. The rest of that folder
     is the team's own analysis of the target: on a replay it is hindsight, so read only `papers/` unless the
     operator says otherwise.
4. **Public structures first, for every target.**
   - Run an exact-sequence sweep against current PDB seqres and a structural search with a **current** foldseek DB.
   - In CASP17 a sweep of the whole target set found 47 with a near-identical public structure, and several
     (8ARV, 7QV6, 3RIQ, 8YIV) were found only late, after the predictions had been run.
   - AF3's near-identity template filter hides 100 % matches, so a template search inside AF3 is not enough.
   - **Check the template database's date.** H2319: the default search missed the current spike structure; with an
     updated search (8UFN for the spike) af3's median DockQ went from 0.24 to 0.73.
   - Evidence key: `public_structures`.
5. **Sites.** Anything that says where a partner binds is a selection gate later (`residudes-selection`): an epitope
   from escape mutants (H2321: S611/R615; H2319: Q542/P543), a recognition sequence (M1439: nt 20–30), a pocket from
   an NMR or crystal structure (M2415: the −1A bulge of the 5′ss:U1 duplex, 8R62/8R63), an active site. Record it as
   evidence with a name the review recognises (`epitope`, `paratope`, `binding_site`, `pocket`, `recognition_site`,
   `interface_prior`, `active_site`, `motif`),
   with residues in input-sequence numbering **and its source** (the paper, the structure, the experiment that puts
   the partner there). `residudes review` flags a site no gate uses, and one whose source is not recorded.
6. **Chemistry the FASTA does not carry.** Read the organisers' description for modifications, ligands and ions:
   phospho-, glyco-, acetyl-, lipid-, metal, cofactor, "bound to", "in complex with".
   - **H2388**, "14-3-3 gamma bound to LARP1 derived phosphopeptide": the FASTA had no modification and named no
     residue. The peptide (PKGLSASLPDL) had two serines. 14-3-3 binds R-S-X-pS-X-P, a phosphoserine followed by
     X-Pro; only S7 (S7-L-P) fits, S5 (S5-A-S) does not. S7 is LARP1 Ser548. The input became
     `"modifications": [{"ptmType": "SEP", "ptmPosition": 7}]` on the peptide; MSA and templates unchanged, af3 only.
     AF3 median DockQ went from 0.24 to 0.82, and model 1 scored 0.91: the phosphate sits on K49, R56, R131 and Y132
     of the receptor's phosphate pocket.
   - The steps: the description names the chemistry → list the candidate residues → narrow them with the binding
     motif or the literature → record the choice and the rejected candidates (`residudes decide --type variant`) → run
     it as a variant next to the unmodified input.
   - CCD codes: SEP (pSer), TPO (pThr), PTR (pTyr), MLY/M3L (methyl-Lys), ALY (acetyl-Lys), NAG (glycan, as a bonded
     ligand). Metals and cofactors are ligands (`ccd:ZN`, `ccd:FAD`); a metal needs a template to stay bound (H2441 Zn).
7. **MSA depth.** Build the default MSA with `residudes-engine msa`, then run `residudes msa T --depth`. That gives N, the
   number of sequences covering at least 50 % of the query, coverage per position, and thin regions; record it as
   `msa_depth`, along with a coverage plot. A near-empty MSA is the single biggest lever:

   | target | default MSA | deeper MSA | AF3 median lDDT |
   |---|---|---|---|
   | T1331 | query only | family crawl, 26,835 | 0.33 → 0.70 (model 1 0.71, TM 0.86) |
   | T1357 | 9 | plmMSA (OT alignment), 2,345 SSB-focused | 0.59 → 0.88 |
   | T2371 | 7 (Neff 1.1) | plmMSA, 1,471 (score ≥ 1.3) | 0.37 → 0.87 |

   The same jump held for every predictor (T1331: of3 0.33 → 0.66, pt2 0.30 → 0.71, bt2 0.55 → 0.68). These are
   three targets where the MSA was the limiting factor, not an average: on a target that already has a deep MSA,
   deepening it further changes nothing, so measure the depth first and act on what you find.
   - **A cheap test of whether the MSA is the problem:** run ESMFold2 on the single sequence. On T2371 it reached
     median lDDT 0.89 with no MSA while af3 on the default MSA sat at 0.37. If ef2 single-sequence is confident and
     the MSA-based runs are not, deepen the MSA.
8. **When the MSA is thin, deepen it.** Thin means N₅₀ under about 30, or a thin region of 15 or more residues.
   - **Find the family.** Use the UniProt/InterPro annotations of the target or its nearest relatives, HHpred hits,
     foldseek hits of a predicted model, or the literature. T1331 was a terpene synthase, so the isoprenoid synthase
     superfamily IPR008949, Pfam PF19086.
   - **Crawl:** `residudes msa T --crawl --family PF… [--family IPR…]`. It:
     - checks that the query hits each Pfam family (hmmscan) and drops those it does not hit;
     - pulls UniProt members and Pfam seeds;
     - runs hmmsearch over UniRef90 and MGnify (sbatch) and clusters the hits at 90 %;
     - aligns every representative straight to the query with OTalign.
     An HMM alone cannot align remote families: on T1331, 218 of 281 query residues fell into insert states. On
     T1331 the crawl gave 28,968 sequences, 27,319 of them covering at least half the query, with no thin region.
   - **Or** `residudes msa T --plm`: a PLM-embedding search with OTalign through plmMSA, when no family is known
     (T1357, T2371). Restricting the query to the domain that lacks homologs helped T1357 (SSB-focused, 2,345 against
     1,686 for the whole chain).
   - When it finishes, run `residudes msa T --depth` again. The deeper MSA is a variant in the plan
     (`residudes decide --type variant` citing `msa_depth` and `msa_crawl`), and the person sees its coverage on the slides.
9. **RNA and DNA chains need their own homologs.** Search Rfam, nt and RNAcentral with the chain's sequence; M1439's
   intron had 19 homologues from bacteria, archaea and an alga. A protein MSA tool will not find them.

10. **Write it up: `notes/target.md` in the dossier.** The evidence keys are the record; this is the document a
    person reads. `residudes report` embeds it, `residudes slides` and `residudes export` carry it. Write it when the
    identification is done and keep it current as later stages change it. Sections, in this order:
    1. **Identification.** A table: CASP id, the organisers' description, the declared or expected stoichiometry,
       each chain with its length and what it is, the experimental method if it is known, the dates.
    2. **The chains, one subsection each.** What the chain is, its mapping to the native sequence *with the offset*,
       the landmarks (domains, CDRs, catalytic residues, a cleavage site, a tag), and what published structures
       show about it. H2338: chain C is P03951 19–387, `input = native − 18`, apple domains A1–A4 at 2–85/92–175/
       182–265/273–356.
    3. **The biology that bears on the structure.** What binds where, which functional state the construct is in,
       how the partners are presented in the cell or on the particle. Two or three paragraphs, not a review.
    4. **Sites and chemistry**, in input numbering, each marked as what it will become: a gate, a census, a variant.
    5. **What is uncertain**, and what in the plan answers it. Name the question, not a list of risks.
    6. **The structures that exist, as a table.** One row per entry a search returned: the PDB id, the identity and
       the coverage as the search reported them, which chain it matches, what state it is in (apo, peptide-bound,
       a different isoform) and whether it is usable as a template. H1388: 23 exact-sequence human 14-3-3 gamma
       entries, best peptide-bound 6A5S at 2.10 A. A sentence saying "templates exist" is not this table.
    7. **Figures.** A note that says "deep MSA" is weaker than one that shows coverage per residue, so link what was
       drawn: the coverage plot `residudes msa T --depth` writes to `report/figs/msa_coverage.svg`, a map of each
       chain with its domains and the sites that will become gates, and a picture of the reference structure a gate
       will use (`residudes render`). Caption each with what it shows and where the numbers come from. Link them as
       paths relative to the dossier (`report/figs/chain_map.svg`), never as absolute paths: the note is read
       embedded in `analysis.md`, copied into an exported package and rendered on a git host, and an absolute path
       survives none of those.
    8. **References.** Numbered, with PMID / PMC / DOI and the local file for anything you actually read, and a line
       for anything you could not get and what it would have told you.
    Rules: every claim carries its source where it is made; no claim from memory; say which numbering you converted
    from; keep it as long as it needs to be and no longer. A reader should be able to check any sentence. Five
    habits from the notes that turned out to be worth reading, all of them about where a number came from:
    - **give the path.** The references are the files you read, each with what it gave, by absolute path. A reader
      checking a count has to be able to open the same file.
    - **mark what you counted yourself.** A number you derived - residues covered, hits in an a3m, distances in a
      structure - is written "as counted here", not left looking like something a database said.
    - **mark what you cannot source.** "Not recorded", "rests on the template hits alone, so treat it as
      unsourced". An inference labelled as one is useful; an inference that reads like a fact is not.
    - **correct the record when it is wrong, and show the count.** T1416's construct evidence said "mutations:
      none detected"; the note found the input differs from the current UniProt entry and matches the obsolete
      one, counted it, and judged it harmless. A correction says what it is, how it was checked, and what changes.
    - **caption a figure with what drew it.** The command, the input file, the Slurm job if there was one, and
      what the colours mean. A picture whose provenance is in the text is evidence; one without is decoration.

## Design variants (stage HYPOTHESES → INPUTS_READY)

- **The complex itself is public:** use the released structure as an af3 multimer template, with its missing
  residues built (`residudes-modeller`, one FASTA record per distinct chain). H2422 (9F8E asymmetric unit, 19 of 1,172
  residues missing) and H2423 (7ACZ, 77 of 1,182): af3's median DockQ went from 0.35 to 0.94 and from 0.78 to 0.92.
  Per-chain templates at 89–94 % identity were already in the default input; the multimer template is what placed
  the chains.
- **A homolog with the assembly:** when a public structure or close homolog has the oligomer in its biological assembly,
  build the whole assembly on the target's sequence with MODELLER and give it to af3 as a multimer template
  (`residudes-modeller`). On T2461o (A24; 3VCD, 63 % identical) that took the median DockQ from 0.46 (no template) and
  0.62 (the homolog's assembly used directly) to 0.88.
- **Templates, PTMs and metal ions only act in af3.** pt2, bt2 and of3 ignore multimer templates. So judge a template variant by af3 models only.
  - H2340: the 8TVF template gave af3 DockQ 0.11 → 0.61, but an of3 model from the same variant was submitted (−0.27 lDDT).
- **Template indices are full-sequence positions** (combine.py). Check the auth/label chain mapping: `--pin-template B=4kbq:C` once resolved to chain D silently.
- **Monomers ignore `--pin-template`**, and metals need a template to stay bound.
- **Every variant has a reason.** Each one gets a `residudes decide --type variant` with its evidence. Build inputs with `residudes-engine inputs` / `residudes-engine prepare`.
