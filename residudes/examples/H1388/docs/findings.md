# What the steps found and decided

Each line below is in [`dossier.json`](../dossier.json) with its evidence; this page puts them in order.

## Analysis (NEW → HYPOTHESES, three steps, $1.57)

- **Both chains identified**: 14-3-3 gamma (UniProt P61981, chain B, 233 residues) and an 11-residue peptide of
  LARP1 (Q6PKG0) around S548 (chain A). The construct mapping and identity are evidence `construct`, `identity`.
- **23 public structures of the receptor at 100% identity**, the best peptide-bound ones 6S9K (1.60 Å, a caspase-2
  phosphopeptide), 6ZBT (1.80 Å) and 6A5S (2.10 Å); none of the complex. Evidence `public_structures`, `templates`.
- **The literature put the phosphate first.** 14-3-3 binds phosphoserine motifs; the peptide carries the mode-I
  motif with S548 at the phosphoacceptor position. Evidence `literature`, `motif`, `binding_site`, `chemistry`.
- **MSA depth**: 10,377 sequences for the receptor, the peptide alone. Evidence `msa_depth`.

## The plan and its checkpoint (PLAN_REVIEW)

Seven decisions, D1–D7. The plan: phosphoserine at peptide position 7 (D1, D5), A1B1 as declared (D2), the
ColabFold MSA on the receptor and none on the peptide (D3), 6S9K pinned as the first template (D4), and 60 GPU
jobs - AF3 with the phosphate for 20 seeds, AF3 without it for 10 as a control, Protenix, Boltz-2 and OpenFold3
with it for 10 each, five samples per seed (D6).

[`checkpoints/C1_plan.md`](../checkpoints/C1_plan.md) asked four questions. The agent answered each, and named two
inconsistencies in its own decisions before resolving: one decision put the phosphate on AF3 only while a later
one put it on Protenix as well, and the gate text required all three pocket residues while its answer passed
models at two of three. It resolved C1 with `proceed`.

## The pool (RUNNING → POOL_READY)

All 60 jobs finished; 300 models imported. The pocket gate (phosphate contact with B49, B56, B131, B132)
passed 235 of 300: AF3 106 of 150 (the unmodified control carries no phosphate to place), Protenix 50 of 50,
Boltz-2 46 of 50, OpenFold3 33 of 50.

## The results checkpoint (RESULTS_REVIEW)

The standard review raised one trigger, and the agent added three of its own:

- **missing models** - a false positive, and said so: the check expected every runner in a variant to match AF3's
  20 seeds, while the plan gave the others 10.
- **a peptide-register split.** Superposed on the receptor, AF3 with the phosphate and Protenix put the peptide
  core (A4–A10) in one register, a median 0.56 Å from the top AF3 model; Boltz-2 sat 3.05 Å away, OpenFold3 6.8 Å,
  unmodified AF3 7.9 Å. Whole-complex TM clustering does not separate them (evidence `peptide_agreement`).
- **the phosphate was carried by AF3 alone.** Protenix, Boltz-2 and OpenFold3 wrote SER at A7 in their models; the
  modification did not reach those runners. Protenix still found AF3's register without it.

C2 was resolved with `proceed`.

## The slate

[`slate_v2.md`](../slate_v2.md): three AF3 models with the phosphate (seeds 20, 9, 7) and two Protenix models
(seeds 5, 10), all in the agreed register, all passing every gate. The agent read it against the gates once more,
approved it under the mandate, and packaged it: [`package/checks.md`](../package/checks.md) - both chains, lengths
right, no breaks, no non-proline cis peptides, no clashing pairs, nothing to relax.
