---
name: residudes-selection
description: Score, gate, cluster and select models from a prediction pool, and propose a slate of 5. Structural gates from what the target analysis found (residudes gate), residudes-qa scores and clusters, then confidence within each predictor. Use at dossier stages POOL_READY (QA and gates) and RESULTS_REVIEW (the slate, once the first results are reviewed), and APPROVED for packaging checks.
---

# Selection

## What CASP17 measured

Released structures, model 1 chosen by each rule from the same pool. DockQ here is CASP's wave average over
interfaces, weighted by native contacts ([`docs/metrics.md`](../../docs/metrics.md)):

| rule | mean lDDT (56 targets) | mean DockQ (37 complexes) |
|---|---|---|
| **our model 1: gates, then confidence** | **0.824** | **0.651** |
| af3 ranking_score 1st | 0.815 | 0.592 |
| af3 ipTM/pTM 1st | 0.812 | 0.589 |
| ranking_score 1st, all predictors | 0.786 | 0.564 |
| ipTM/pTM 1st, all predictors | 0.768 | 0.548 |
| random pick (pool mean) | 0.772 | 0.503 |
| best model in the pool (oracle) | 0.852 | 0.751 |

Paired per target, ours minus the rule: against af3's own confidence +0.009 lDDT [0.002, 0.017] and +0.059 DockQ,
and we lose on 9-11 targets; against ranking_score over all predictors +0.038 [0.019, 0.060] and +0.088. So most
of the margin is in not ranking across predictors, whose confidences are on different scales, and a smaller part
in the gates. One campaign, released natives, thresholds set per target by a person who saw the pool
([`docs/casp17_cases.md`](../../docs/casp17_cases.md)).

Confidence alone, compared across predictors, was barely better than a random pick. The gain came from structural
gates first and confidence second. The gates came from the target analysis: an epitope from escape mutants, a
reference structure for a rigid part, a binding site from the literature.

## Steps

1. **Score and cluster the pool with residudes-qa** (through sbatch for GPU stages):
   - `residudes qa T --sto S --what interface`: DockQ/ICS/IPS predicted per interface. For antibody targets add
     `--mode ab --ab-chains CD --ag-chains AB` (model chain ids). Its scores were weak in CASP17 (H2442 ρ −0.28), so
     the slate leaves them out unless `RESIDUDES_USE_AB_INTERFACE=1` is set; let gates lead there;
   - `residudes qa T --sto S --what cluster`.
2. **Gates first: `residudes gate T --spec gates.json --sto S`.** Write the gates the target analysis supports, one
   JSON per target (a second set gets `--name`). `residudes gate T --init --sto S` drafts one from the dossier: a
   contact gate for each site in the evidence (`epitope`, `binding_site`, `pocket`, ... with residues it can read) and an
   assembly sanity gate, each with a note on where it came from. No generic clash or symmetry gate is drafted: on
   CASP17's released natives they removed good models (see `docs/gates.md`). Edit the draft: partners, thresholds,
   references. The full spec reference is `docs/gates.md`. The command evaluates every model of the stoichiometry through
   sbatch (1,360 models take about 35 s on 16 cores) and writes `gates/<name>_<S>.csv`. `residudes gate T --show
   --sto S` then prints the pass counts per gate and per predictor, the best passing model of each, and the census,
   and records evidence `gates`. Gate types (residue numbers are input-sequence positions; chain ids A, B, … in input
   order are the same in every predictor's model):

   | type | passes when | from CASP17 |
   |---|---|---|
   | `rmsd` | CA RMSD of `select` to a reference ≤ `max` (residues paired by alignment, so the reference keeps its numbering) | H2319/H2321: spike core within 1.0 Å of apo spike 8UFO |
   | `interface` | ≥ `min_contacts` residue pairs between groups `a` and `b`, and/or ≥ `min_fraction` of a reference's pairs | spike dimer interface intact; VH–VL paired |
   | `contact` | a `partner` atom (chains, or `ligand`) within `within` Å of the `site` | H2321: paratope within 6 Å of S611/R615 (2A2 escape mutants); H2319: on the 7C8 epitope (Q542/P543) |
   | `clash` | ≤ `max` inter-chain heavy-atom pairs closer than `cutoff` | only for chains through each other: relaxation fixes ordinary clashes, and the best H2340 models had 77–117 pairs under 1.5 Å |
   | `symmetry` | a ring of copies (`"A,B,C"`) maps onto itself shifted by one copy within `max` Å (CA RMSD) | homo-oligomers: the T1417 replay agent wrote this check itself before the tool had it |
   | `stacking` | the median nearest-neighbour chain spacing is about 4.8 Å (cross-β) | T2463: Boltz-2 stacks 4.8 Å, other predictors' helical bundles 8–10 Å |

   Every model also gets `complete`: it has every chain the stoichiometry implies. In M2377, 310 models came from runs
   that had lost the RNA chain.

   ```json
   {"gates": [
     {"name": "core", "type": "rmsd", "select": "A:27-520,B:27-520", "ref": "8ufo.cif", "ref_chains": {"A": "A", "B": "B"}, "max": 1.0},
     {"name": "spike_dimer", "type": "interface", "a": "A", "b": "B", "ref": "8ufo.cif", "ref_chains": {"A": "A", "B": "B"}, "min_fraction": 0.6},
     {"name": "vh_vl", "type": "interface", "a": "C", "b": "D", "min_contacts": 25},
     {"name": "epitope", "type": "contact", "site": "A:611,615", "partner": "C,D", "within": 6.0},
     {"name": "interpenetration", "type": "clash", "cutoff": 1.0, "max": 20}]}
   ```
   Gate a stoichiometry on what you know, not on what you hope. Every gate cites the evidence it comes from in the
   selection decision. When a checkpoint answer changes a gate (a person drops one, or replaces it with another), edit
   the spec and rerun `residudes gate` before proposing the slate: a gate judged by hand and written as evidence is
   not what `residudes approve` checks, and it then reports slots failing a gate the decision already removed. For a fibril hypothesis, `stacking` checks cross-β spacing (about 4.8 Å); it tests that hypothesis, not correctness.

   Effect in CASP17, DockQ of the pick from the same pool:

   | target | gates, then confidence | highest ranking_score, all | highest ranking_score, af3 | best in pool |
   |---|---|---|---|---|
   | H2319 | **0.93** | 0.67 | 0.71 | 0.94 |
   | H2321 | 0.85 (see below) | 0.59 | 0.50 | 0.85 |

   H2319 is the clean case: model 1 passed every gate and was the top rubric score among the 6,349 that did.
   H2321 is not, and is the more useful one to remember. The same gates left 246 of 12,810 models, the submitted
   model scored 0.85, and yet that model **fails** the core gate (1.123 Å against a 1.0 Å threshold) and ranks
   29th by the rubric, whose top passer scores 0.58. No record says how it was actually chosen. Two lessons: the
   epitope gate alone does not separate DockQ on this target (median 0.57 among passers against 0.59 among
   failures, because wave DockQ is dominated by the spike-dimer interface), and a pick that is not written down
   cannot be defended afterwards, which is why `residudes slate` records the rule and the models it chose.

   Note which way round it failed: the unrecorded human choice found the best model in the pool (0.855, equal to
   the oracle) while the written rubric would have taken a 0.58. The lesson is not that the rubric knows better.
   It is that whatever the person saw that day was worth keeping, and it was lost.
3. **A census when the partner can sit in more than one place.** Put a `census` in the spec: each model is labelled by
   the named site its probe (a ligand or a chain) contacts most.
   - **M2415.** 815 ligand poses split into the 5′ss bulge 444, the Sm site 186, the U1-70K RRM 110 and other 75; of3
     put about 99 % at the bulge. The literature (NMR, 8R62/8R63) puts risdiplam-class modifiers at the −1A bulge of the
     5′ss:U1 duplex. Model 1 was the highest ranking_score inside the bulge: ligand 8.8 Å from the released pose,
     against 51.8 Å for the highest ranking_score overall (highest ligand ipTM: 10.3 Å, largest pose cluster: 9.8 Å).
   - `"census": {"probe": "ligand", "within": 5.0, "sites": {"bulge": "B:1-12", "sm": "D:20-40", "rrm": "E:1-90"}}`,
     then `residudes slate T --sto S --site bulge`.
4. **Rank inside each predictor, then compare across.**
   - Never rank models of different predictors by raw ranking_score. Over 588,291 pool models the medians were bt2
     0.84, af3 0.67, of3 0.58, pt2 0.58 and ef2 0.47, so a raw ranking hands every slot to bt2.
   - Do not normalise across predictors with a z either (see the slate numbers below). `residudes slate` orders each
     predictor by its own confidence, leads with af3's best, and alternates predictors after that. Within a predictor
     it uses a robust z (median and MAD, with a floor): a plain z let a few failed models scored 100 squash the rest
     of their predictor.
   - Never rank templated against template-free models by ipTM. The templated score answers "did I copy the template".
     The slate still orders a predictor's inputs together by ranking_score: a z per input did worse on CASP17's
     released natives (model 1 DockQ 0.535 → 0.503). So when a template may be the wrong one, read the inputs
     separately at the results review and choose with `--models`.
5. **Know what each signal is good for.**
   - Interface XGB worked for homo-oligomers (T2417 ρ +0.74) and failed for antibody–antigen and hetero interfaces
     (H2442 −0.28).
   - Cluster population is a tie-breaker, not a selector: the biggest basin was wrong for H2389, H2391 and T1363.
   - Consensus or centroid: cap the number of members per predictor, or same-predictor siblings inflate it.
   - Never judge from sample_0 alone: read all samples of a seed. Free runs need 25 or more seeds before calling an
     interface absent.
   - **When confidence does not track quality, more seeds do not help the pick.** M1439 (protein–RNA): ranking_score
     against lDDT gave ρ −0.15, −0.24 and −0.23 for af3, of3 and pt2. From 15 to 1,920 models, the best model in the
     sample rose from lDDT 0.59 to 0.63, but the top-ranking_score pick stayed at 0.54. The identification had named
     the I-AniI recognition core in the intron (nt 20–30), but no gate or census used it. 92 % of poses put the
     protein on the RNA's 3′ end, and model 1 reproduced 76 % of the native protein-side interface but 17 % of the RNA
     side. A site named at IDENTIFIED belongs in a gate or a census.
6. **Slate.**
   - Start from `residudes slate T --sto A3`, or a hedge across stoichiometries (`--sto A2:3,A1:2`, slots in that order).
     When a gate result exists, only the models that pass are candidates; `--gates NAME` picks one gate set, and
     `--gates none` ignores them (say why in the decision). The rules then apply mechanically:
     - model 1 is af3's most confident candidate;
     - the other slots alternate predictors in a fixed order measured on CASP17's released natives (af3, bt2,
       bt2_jax, pt2, of3, ef2: each one's own top model averaged DockQ 0.536, 0.493, 0.473, 0.448, 0.397, 0.369),
       each ordered by its own confidence (with the interface score when there is one);
     - one model per cluster and per seed in slots 1–3, and at least two predictors.

     Over CASP17's released natives, model 1 by af3's ranking_score averaged DockQ 0.535 over 53 complexes (lDDT
     0.783 over 31 monomers). The raw ranking_score across predictors gave 0.504 / 0.770, and a per-predictor z
     0.440–0.468 / 0.745: a z rewards a model for standing out within its own predictor, not for being right. Run
     through the slate's own picking (caps, seed diversity), the change took model 1 from 0.440 to 0.535 and the best
     of five from 0.598 to 0.620 (complexes; monomers 0.747 → 0.783 and 0.785 → 0.801).
   - If fewer models pass than there are slots, the proposal says so. Fill the rest by judgement, or loosen a gate
     and record why.
   - Model 1 is the best-supported hypothesis. Models 2–5 hedge across stoichiometry, architecture, site, cluster and
     predictor. Do not fill slots 1–2 with two models of one predictor that are 1 Å apart.
   - Record the chosen five with `residudes slate T --models M1,M2,M3,M4,M5` (names as in the proposal). Any model that
     fails a gate is listed under the table. Then `residudes decide T --type selection ...`, listing each model with its
     evidence (`gates` among them) and the rejected alternatives. Then `residudes stage T SLATE_PROPOSED` and stop: a
     person approves.
   - Every submitted model has a selection record. Two CASP17 models (M2377v1, H2388) had none, and their reasoning
     had to be reconstructed from files afterwards.

## Packaging (after APPROVED)

`residudes package T --server` (or `--human`) does the steps below for the approved slate:
- the geometry checks, written to `package/checks.md`;
- a polymer-only copy of each model;
- the CASP file, from the team's casp17-utils (`casp17 prepare`).

It then moves the target to PACKAGED. Read `checks.md` and tell a person what needs a look. `--check-only` runs the
checks at any stage. You never submit.

Check geometry before formatting:
- clashes, including homo pairs;
- non-Pro cis peptides;
- C–N bond lengths and chain breaks;
- chain and residue counts against the sequence and stoichiometry.

Formats:
- TS files carry ATOM records only; ligands go in LG with a TER per receptor chain.
- One model per file.
- mmCIF needs the polymer records (`_entity_poly_seq`, `label_seq_id`).
- More than 26 chains need two-letter chain ids.
