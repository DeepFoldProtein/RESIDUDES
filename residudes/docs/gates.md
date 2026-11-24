# Structural gates (`residudes gate`)

A gate is a structural test a model must pass before its confidence counts: the epitope is bound, the rigid core
matches a reference, the expected interface is there, nothing clashes. In CASP17 the team chose model 1 by gates
first and confidence second, which scored above every confidence rule tried on the same pools: over 37 complexes,
mean DockQ 0.651 against 0.564 for the highest ranking_score across predictors ([metrics](metrics.md); paired difference +0.088,
Wilcoxon p = 0.001) and against 0.592 for af3's own confidence (+0.059, p = 0.009). See [`casp17_cases.md`](casp17_cases.md).

## The workflow

`<dossier>` below is the target's folder, `<work_root>/targets/<T>/`. `--sto` defaults to the stoichiometry given at
`residudes target add --stoich`.

```sh
bin/residudes gate T --init --sto A2B1C1           # draft gates/draft_A2B1C1.json from the dossier's site evidence
$EDITOR <dossier>/gates/draft_A2B1C1.json        # partners, thresholds, references
bin/residudes gate T --spec <dossier>/gates/draft_A2B1C1.json --sto A2B1C1 --name epitope   # sbatch, every model
bin/residudes jobs T                               # wait until the gate job is COMPLETED
bin/residudes gate T --show --sto A2B1C1 --name epitope   # pass counts, census -> evidence `gates`
bin/residudes slate T --sto A2B1C1                 # proposes only from the models that pass
bin/residudes slate T --sto A2B1C1 --site bulge    # ... and sit at one census site
```

- **The run.** Without `--local`, every evaluation is one sbatch job (`--workers` CPUs, `--hours` limit), so it needs
  a Slurm host. With `--local` it runs here with up to 4 workers and refuses pools over 100 models. 1,360 models took
  35 s on 16 cores, 2,120 protein–RNA models 88 s, and H2319's 43,030 antibody–spike models 2 min 40 s.
- **Chain ids are checked.** A spec naming a chain the stoichiometry's models do not have is refused at submit.
- **What the result reaches.** The analysis (`residudes report`), the slides (one slide per gate set), the results
  review and `residudes approve` all read it. Approve names any slate model that fails a gate.
- **Several gate sets** live side by side under different `--name`s. The slate uses the most recent one for the
  stoichiometry, `--gates NAME` picks one, and `--gates none` ignores them all (say why in the selection decision).

## The spec

```json
{"gates": [
  {"name": "core", "type": "rmsd", "select": "A:27-520,B:27-520", "ref": "8ufo.cif",
   "ref_chains": {"A": "A", "B": "B"}, "max": 1.0},
  {"name": "spike_dimer", "type": "interface", "a": "A", "b": "B", "ref": "8ufo.cif",
   "ref_chains": {"A": "A", "B": "B"}, "min_fraction": 0.6},
  {"name": "vh_vl", "type": "interface", "a": "C", "b": "D", "min_contacts": 25},
  {"name": "epitope", "type": "contact", "site": "A:611,615", "partner": "C,D", "within": 6.0},
  {"name": "clash", "type": "clash", "cutoff": 2.0, "max": 5}],
 "census": {"probe": "ligand", "within": 5.0, "sites": {"bulge": "B:1-12", "sm": "D:20-40"}}}
```

**Chains and residues.** Model chains are A, B, C, … in input order (AA, AB, … past Z), the same in every
predictor's `model.cif`. Residue numbers are positions in the input sequence (1..N).

**Copies.** In a stoichiometry like A2B1C1, chains A and B are the two copies of entity A, C is entity B, and D is
entity C.

**Selections.** Write them as `"A:1-500,611,B:3-9"`: a chain prefix holds until the next one, and `"A"` alone means
the whole chain.

**Partners.** Give model chain ids (`"C,D"` for the Fab of an A2B1C1 model), `"ligand"` (every chain with no
protein or nucleic-acid residue),
`"ligand:LIG"` to pick a ligand by residue name, or a residue selection (`"A:7"`, `"A:5-9,B"`). A selection counts
only those residues: "the peptide's phosphoserine sits in the phosphate pocket" is
`{"site": "B:49,56,131,132", "partner": "A:7", "within": 5.0, "min_residues": 3}`. A census probe takes the same
forms.

| type | required | optional (default) | passes when |
|---|---|---|---|
| `rmsd` | `select`, `ref`, `max` | `ref_chains` (same ids), `min_pairs` (10) | CA (C4′ for nucleic acids) RMSD of the selection to the reference, after superposition, is ≤ `max`. Model and reference residues are paired by a sequence alignment per chain, so the reference keeps its own numbering. |
| `interface` | `a`, `b`, and `min_contacts` or `ref` + `min_fraction` | `cutoff` (5.0 Å), `ref_chains` | at least `min_contacts` residue pairs between groups `a` and `b` have heavy atoms within `cutoff`; and/or at least `min_fraction` of the reference's pairs are present. `a` and `b` take chains or residue selections (`"A:400-420"`) |
| `contact` | `site`, `partner`, `within` | `min_residues` (1) | a partner atom is within `within` Å of the site, touching at least `min_residues` site residues |
| `clash` | – | `cutoff` (2.0 Å), `max` (5) | at most `max` inter-chain heavy-atom pairs are closer than `cutoff`. A glycan or ligand bonded to the protein in its own chain (< 1.75 Å) is a bond, not a clash. Use it for interpenetrating chains, not for packaging-level clashes: relaxation fixes those |
| `symmetry` | `chains` (a ring in order, e.g. `"A,B,C"`), `max` | `min_pairs` (10) | the ring superposed on itself shifted by one copy (A→B→C→A) has CA RMSD ≤ `max`: a closed, even Cn ring. On T1416's trimers af3, pt2 and of3 gave 0.2–0.3 Å; a broken single-sequence model gave 30 Å |
| `stacking` | – | `chains` (all), `min` (4.4 Å), `max` (5.3 Å) | the median distance from each chain to its nearest neighbour (CA centroids) lies in [min, max]: cross-β stacking. On T2463's 24-mers the Boltz-2 (JAX) stacks gave 4.77–4.81 Å and the helical bundles of the other predictors 8.4–10.5 Å. It tests the fibril hypothesis, nothing more: the released T2464 is an α lattice |

**Every model also gets `complete`.** It passes when the model has every chain the stoichiometry implies. Runs that
lost a chain fail it. M2377 lost the RNA in 310 models, and M1439's Boltz-2 RNA-secondary-structure runs hold only the
protein.

**Census.** It labels each model by the named site its probe (a ligand or chains) touches most within `within`
(default 5 Å), or `other`. It is not a gate: use it to see where the pool puts the partner, and to restrict the slate
with `--site`.

**Rules.**
- Gate names are letters, digits and `_`, and unique; `complete` and `census` are reserved.
- Thresholds must be numbers.
- A `notes` list and a `note` on each gate are allowed and ignored by the engine.
- `ref` paths are resolved against the spec's folder.

## Drafting from the dossier (`--init`)

`residudes gate T --init --sto S` writes `gates/draft_<S>.json` the first time and `draft_<S>_2.json`, `_3`, … after
that. It never overwrites an existing draft, so edit the newest one. It holds:

- **A contact gate for each site in the evidence.** Site keys are `epitope`, `paratope`, `binding_site`, `pocket`,
  `recognition_site`, `active_site`, `interface_prior` and `motif`. It reads residues it can trust:
  - `A:611,615` as written;
  - three-letter codes (`Ser548`);
  - lists of two or more one-letter residues (`S611/R615`).

  Target ids (H1392), CDRs (H3), stoichiometries (A24) and subsites (S1) are not read as residues. Always write model
  chain ids, as AGENTS.md asks: in A2B1C1, B is the second copy of A, and the Fab's heavy chain is C. A letter is
  read as that model chain; where it is also an entity whose first chain differs, the note names that chain.

  Residues written without a chain ("K49, R56, R131 and Y132 of 14-3-3") go to the one entity whose sequence has those
  residues at those numbers. Failing that, they go to the chain the text names ("on chain B"). Only if neither works
  do they go to chain A, flagged as a guess.
- **The partner is every other entity, never another copy of the same one.** Otherwise a homo-oligomer's epitope
  would pass through its own neighbour. Where no other entity exists, the gate is left to you, unless the dossier
  records a ligand.
- **An assembly sanity gate** (`assembled`: chain A touches the rest). It is the only generic gate drafted.
  - A symmetry gate is only suggested, in the notes. On CASP17's released natives a ring gate on every entity with
    three or more copies failed every model of lattices, fibrils and some rings (T2430, T2463, T2395).
  - A clash gate is not drafted: relaxation fixes clashes at packaging. The best models of H2340 (templated af3,
    DockQ 0.77) had 77–117 atom pairs under 1.5 Å, while confidently wrong models had none.
- **Notes** for sites without readable residues, and for `public_structures` or `templates` evidence that may give
  you a reference for an `rmsd` gate.

## Output

`<dossier>/gates/<name>_<sto>.csv` has one row per model:
- `key`, `runner`, `ranking_score`, `rs_z_runner`, `n_chains`;
- `<gate>_value`, `<gate>_pass` and `<gate>_*` extras (`_pairs`, `_fraction`, `_residues`);
- `census`, `pass` and `path`.

The spec as evaluated is kept next to them as `<name>_<sto>.json`. `<name>_<sto>.summary.json` holds the counts that
`--show` records as evidence `gates`:
- `n` and `n_pass`;
- passes per gate;
- per predictor, the models, the passes and the best passing model;
- census counts per predictor, and the runs with missing chains.

## An example: a phosphopeptide in its pocket (H1388, the real pool)

(For a replay of H1388 or H2388 this section is hindsight.)

The task was 14-3-3γ (chain B) with an 11-residue LARP1 peptide (chain A) in which residue 7 is the phosphoserine. The
875 A1B1 models were gated three ways:

| gate | passes |
|---|---|
| chain-level: the peptide within 6 Å of 3 of the 4 phosphate-pocket residues (B:49,56,131,132) | 873 of 875: it tells nothing apart |
| residue-level: peptide residue 7 (`"partner": "A:7"`) within 5 Å of 3 of the 4 | 210 of 875 |
|   … af3 with the phosphoserine in the input (variant `af3_msa_pS548`) | 125 of 125 |
|   … af3 without it (`af3_msa`, `mmseqs2_gpu`) | 4 of 125, 4 of 125 |
|   … OpenFold 3 and Boltz-2 (JAX) without it | 10–29 of 125 per variant |

So the residue-level gate keeps the models whose phosphoserine sits in the pocket and drops the rest, with no
native structure involved. In CASP17 that input took AF3's median DockQ from 0.24 to 0.82
([`casp17_cases.md`](casp17_cases.md)). The chain-level version of this gate could not say "residue 7" at all,
which is why partner selections exist.

## Where the thresholds come from, and what they cost

The numbers in a spec are judgement, not standards. The defaults are conventions — 6 Å between heavy atoms is the
usual definition of a residue-residue contact, 5 Å for a ligand probe — and the rest are set per target from what
the analysis found: a core that should match an apo crystal within about 1 Å, a stacking spacing of 4.4–5.3 Å
because the measured cross-β stacks sit at 4.77–4.81 Å while a loose bundle sits at 8.4–10.5 Å, a ring RMSD of
3 Å because closed rings of equal copies come out at 0.2–0.3 Å and a broken one at 30 Å.

Two of those are backed by a measured separation. The others are a choice, and on H2321 the choice was decisive in
the wrong direction: the core gate at 1.0 Å excluded, by 0.123 Å, the model that turned out to be the best in the
pool ([`casp17_cases.md`](casp17_cases.md)). A threshold tight enough to decide anything is tight enough to be
wrong, which is why every gate result keeps the **measured value** per model, not just pass or fail: a near miss
stays visible in `gates/<name>_<sto>.csv` and in `residudes gate --show`, and the slate's own rule is recorded
beside it. If a gate removes a model a person would have kept, that should be arguable afterwards from the files.

## Name the atom, when the chemistry is the point

A `contact` gate measures between the nearest atoms of two selections, which is the right default and hides a trap
when the residue in question carries a modification. On H1388 the plan proposed 4 Å between the peptide's
phosphoserine and the four pocket residues of 14-3-3. Measured in the reference structure 6A5S, that distance is
2.4–2.8 Å from the phosphate oxygens, 3.6–3.9 Å from the phosphorus and 4.2–5.1 Å from the serine OG
([`examples/H1388/notes/target.md`](../examples/H1388/notes/target.md), measured by the agent that proposed the
gate). The unmodified variant of the same run has no phosphate, so OG is the only atom it can be measured from,
and at 4 Å the reference structure itself fails the gate.

Two ways out, and the spec should say which: name the atoms the gate measures from, or set the threshold for the
worst case the run contains. What it must not do is carry a number that was chosen for one chemistry and applied
to models that do not have it.

## Where gates come from in practice

Everything in this table comes from reading about the target, so each site is recorded with its source (the paper,
the structure or the experiment that puts the partner there): `gate --init` copies the source into the drafted gate's
note, and `residudes review` flags a site whose source is missing. [`literature.md`](literature.md) covers that reading.

| what you know | gate |
|---|---|
| escape mutants or a literature epitope | `contact`: the epitope residues, partner the antibody chains, within 6 Å |
| an apo structure of a rigid part | `rmsd`: that part against the apo structure, ≤ 1 Å |
| a known obligate interface (a dimer, VH–VL) | `interface` with `ref` + `min_fraction`, or `min_contacts` |
| an NMR or crystal pocket for a ligand | `census` over the candidate pockets, then `slate --site` |
| a recognition sequence in a nucleic acid | `contact` or `census` on those nucleotides, probe the protein |
| none of the above | `complete` (always) and `assembled`: do not invent a gate to have one |
