---
name: lantern-modeller
description: Build the whole assembly with MODELLER on the target's own sequences - threaded onto a homolog's biological assembly, or a released structure of the complex with its missing residues filled in - and give it to AF3 as a multimer template. Use when the PDB has the complex or a homolog in the same oligomeric state (a ring, a cage, a filament segment, a conformational state) and the template-free predictions get the subunits right but the assembly wrong or inconsistent; at HYPOTHESES as an input variant, or at RESULTS_REVIEW as a loop.
---

# Whole-assembly homology models as AF3 multimer templates

## What it did in CASP17

- **T2461o: A24, 191 aa.** 3VCD, a designed octahedral cage 63 % identical to T2461, has the 24-mer as its
  biological assembly. The default input had per-chain templates only (3PAC, 3NWG, 3N79, 8T6N). Interface accuracy
  (AF3 median DockQ, 40 models each) by variant:

  | variant | DockQ (median) |
  |---|---|
  | `af3_msa`, no assembly template | 0.46 |
  | `mmseqs2_gpu` | 0.46 |
  | `af3_3vcd`, the 3VCD assembly itself as the multimer template | 0.62 |
  | `af3_mdl`, the T2461 sequence threaded onto 3VCD's 24-mer by MODELLER | **0.88** |

  The modelled 24-mer kept 3VCD's cage (protomer backbone RMSD 0.07 Å to 3VCD). Submitted model 1 came from
  `af3_mdl`: DockQ 0.88, QS 0.96, lDDT 0.91; field #2 of 71. The protocol analysis counts it as the agent's largest
  gain (+0.32 DockQ).
- **H2422 / H2423: A2B2, the complex itself released.** The 9F8E and 7ACZ asymmetric units had all four chains, with
  19 of 1,172 and 77 of 1,182 residues missing. With those residues built and the result given to af3 as the
  multimer template, af3's median DockQ went from 0.35 to 0.94 and from 0.78 to 0.92 (model 1: 0.98 and 0.96).
  Per-chain templates at 89–94 % identity were already in the default input and did not place the chains.
- **E2366: A3, 1,269 aa.** MODELLER models of the trimer's states (`af3_modeller_1up`, `_2up`, `_closed`) went in as
  multimer templates. With `AF3_TPL_WEIGHT=4.0`, one seed gave ranking 0.89 / ipTM 0.83 on all five samples. That beat
  the best of 32 seeds at weight 1.0 (0.86 / 0.80).

## Why a MODELLER model beats the homolog's own assembly

**Same sequence, so a 1:1 mapping.** `combine.py --multimer-template model.cif` maps every query chain to a template
chain of the identical sequence:
- no `queryIndices`/`templateIndices` to translate;
- no index-space trap. `combine.py` counts the full sequence while the aligned builder counts observed residues, and
  on H1340 the two silently disagreed.

**No missing residues.** Loops the crystal did not resolve are built, so the template covers every position.

**The homolog's assembly geometry is kept.** Chain placement comes from the homolog. MODELLER only rebuilds each
chain on its own sequence.

## When to use it

- **Prerequisite:** a public structure or close homolog has the oligomer in its biological assembly, and its copy
  number matches your hypothesis.
  - Check the RCSB assembly, not only the asymmetric unit (`lantern-intake`: public structures, templates).
  - 63 % identity was enough for T2461o; a released structure of the complex itself (100 %) is the easy case.
- **The default runs get the fold but not the assembly.** For example: the n-sweep peaks at the right n but the
  interfaces scatter; DockQ-like interface scores are low; a ring is open or packed wrong.
- **Not when the homolog's oligomer is in doubt.** AF3 copies the template's assembly. Keep a template-free control
  variant, and never rank templated against template-free models by ipTM: templated ipTM measures how closely the
  model copied the template.

## Steps

1. **Expand the homolog's assembly** into explicit chains:
   `python3 casp17/gen_inputs/expand_template_assembly.py --in-cif 3vcd.cif --out-cif 3vcd_asm1.cif`
   (first biological assembly by default; `--assembly N` picks another).
2. **Build the model through sbatch.** A 24-mer of 191 aa is about 4,600 residues per model.
   ```sh
   # a homolog's assembly: one FASTA record, every chain gets it
   python3 skills/lantern-modeller/scripts/build_assembly_model.py --template 3vcd_asm1.cif \
       --target target.fasta --copies 24 --out-dir <dossier>/modeller --models 5
   # a released complex: one record per distinct chain; each template chain gets the record it matches best
   python3 skills/lantern-modeller/scripts/build_assembly_model.py --template 9f8e.cif \
       --target h2422.fasta --out-dir <dossier>/modeller [--chain-map A=heavy,B=light]
   ```
   The script:
   - renames chains to single characters;
   - assigns a target sequence to every template chain and writes `chain_map.tsv` (chain, residues, record, identity):
     check it before submitting;
   - writes the PIR (template = N chains, target = the assigned sequences, '/' between chains), aligned per chain by
     `align2d`;
   - builds the models with `AutoModel` and keeps the best by DOPE (`scores.tsv`);
   - writes `best.cif` with `_entity_poly_seq`. AF3 silently drops a multimer template without it.

   `--dry-run` writes the alignment input without MODELLER, as a check before you submit.
3. **Make the input:** `gen_input.sh` / `combine.py --multimer-template <dossier>/modeller/best.cif`.
   - It is an af3-only variant. pt2, of3 and bt2 ignore multimer templates, so judge it by af3 models.
   - Record it with `lantern decide --type variant`, citing `public_structures` and `templates`.
4. **Optionally raise the template weight** to 2–4 (`AF3_TPL_WEIGHT`, af3 `--template-weight`) when the template is
   trustworthy (E2366).
5. **Check before you rely on it:**
   - Did MODELLER keep the assembly? Superpose one protomer of `best.cif` on the template (T2461o: 0.07 Å).
   - Do the models follow the template? Measure TM against `best.cif`, or make it a gate (`lantern gate`, type
     `rmsd` with `"ref": "best.cif"`).
   - Do the interfaces stay put? Run `lantern-qa` interface and cluster.
   - Is the template-free control, at the same n, consistent with it, or does it contradict it?
   - Put the template, a template-free model and the result side by side on a slide (`lantern render`).

## Licence and setup

MODELLER needs a licence key, free for academic use. It is not installed on the cluster, and the agent does not
obtain keys. If `python -c "import modeller"` fails, ask a person with `lantern note T "NEEDS HUMAN: MODELLER licence
key ..."`. They install it with `KEY_MODELLER=<key> uv pip install modeller` into a venv, or from conda's `salilab`
channel, and record where it is. Until then, the dry run and the template-free variants still go ahead.
