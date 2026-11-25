# What CASP17 taught residudes

The DeepFold team's CASP17 analysis against released structures (group 077, 154 targets, 1,337,148 models) shows
where the campaign gained and lost. Each case below names the residudes station where it happens and the command or
skill that now does it, so the next campaign does not depend on someone remembering.

Numbers are against released structures. AF3 medians are over the pool of each variant. What each metric measures, and where it is defined, is in [`metrics.md`](metrics.md).

**How to read the cases.** Each one is a single target, scored after its structure was released, and the set is
selected: a case is here because the intervention changed the outcome enough to measure. It is not a sample of
every target the intervention was tried on, and the count of times it made no difference was not kept, so the
per-target jumps below (0.24 to 0.82, 0.33 to 0.70) say what the intervention *can* do, not what it does on
average. The two findings that are averages over many targets, and carry intervals, are the selection comparison
immediately below and the slate order after it. Cases where the campaign got it wrong are kept in the same table
on purpose — M1439, M2377v1, the generic gates, T1416's copy number — because a list of successes would be the
easiest table to assemble and the least informative.

## Selection: gates first, confidence second

Model 1 chosen by each rule from the same pool (56 targets for lDDT, 37 complexes for DockQ):

Every rule picks from the same pools, so each comparison is paired per target. Δ is our model 1 minus the rule, a
positive value favouring ours; the bracket is a bootstrap 95 % CI of the mean difference, W/T/L counts targets we
win, tie (|Δ| ≤ 0.005) and lose, and *p* is a Wilcoxon signed-rank test.

| rule | mean lDDT | Δ lDDT (W/T/L) | mean DockQ | Δ DockQ (W/T/L) |
|---|---|---|---|---|
| **our model 1: gates, then confidence** | **0.824** | — | **0.651** | — |
| af3 ranking_score 1st | 0.815 | +0.009 [0.002, 0.017] (18/27/11) *p* = 0.04 | 0.592 | +0.059 (21/7/9) *p* = 0.009 |
| af3 ipTM/pTM 1st | 0.812 | +0.012 [0.003, 0.023] (19/27/10) *p* = 0.04 | 0.589 | +0.062 (20/8/9) *p* = 0.007 |
| ranking_score 1st, all predictors | 0.786 | +0.038 [0.019, 0.060] (28/22/6) *p* < 0.001 | 0.564 | +0.088 (23/6/8) *p* = 0.001 |
| ipTM/pTM 1st, all predictors | 0.768 | +0.056 [0.019, 0.103] (27/23/6) *p* < 0.001 | 0.548 | +0.103 (21/8/8) *p* < 0.001 |
| random pick | 0.772 | +0.052 (42/7/7) | 0.503 | +0.149 (29/2/6) |
| best in pool | 0.852 | — | 0.751 | — |

Read the top two rows together: against af3's own confidence, which is what a one-predictor pipeline would use, the
gain is **+0.009 lDDT and +0.059 DockQ**, and we lose on 9–11 targets of the 37–56. The large margins are against
rules that rank across predictors, and those exist because raw confidence is not comparable between them: over
588,291 pool models the ranking_score medians were bt2 0.84, af3 0.67, of3 0.58, pt2 0.58 and ef2 0.47. The honest
summary is that most of the measured advantage comes from *not* ranking across predictors, and a smaller, still
positive part from gating on what the analysis found.

Scope: 56 targets, of which 37 are complexes with a DockQ, from one campaign; domain units are excluded and
alternative reference structures averaged to one target. These are the targets whose natives were released, not a
held-out benchmark, and the gate thresholds were chosen per target by a person who could see the pool.

## The slate's own order

`residudes slate` proposes the first five from the models that pass the gates. Its order was checked on the same
released natives, by scoring the pool's models with OpenStructure and running each rule through the slate's own
picking (caps, seed diversity):

| rule for model 1 | complexes (53), mean DockQ | monomers (31), mean lDDT |
|---|---|---|
| **af3's most confident model, then predictors alternate (the slate now)** | **0.535** (best of 5: **0.621**) | **0.783** (best of 5: **0.801**) |
| raw ranking_score across predictors | 0.504 | 0.770 |
| per-predictor robust z | 0.468 | 0.745 |
| per-predictor plain z (the slate until 2026-11-22) | 0.440 (best of 5: 0.598) | 0.747 (best of 5: 0.785) |
| the slate, but a z per predictor *and input* (templated and template-free on equal footing) | 0.503 (best of 5: 0.588) | 0.730 (best of 5: 0.795) |
| the pool's best model | 0.701 | 0.820 |

**How far these differences can be trusted.** Each rule picks from the same pools, so the comparison is paired per
target; the interval below is a 95 % bootstrap over targets (20,000 resamples) of the mean paired difference
against the rule the slate now uses.

| the slate now, against | complexes (n = 53) | monomers (n = 31) |
|---|---|---|
| per-predictor plain z, the previous rule | **+0.095** (+0.034, +0.164); better on 25, worse on 14, unchanged on 14 | **+0.037** (+0.007, +0.073); better on 15, worse on 10 |
| raw ranking_score across predictors | +0.031 (−0.031, +0.088) | +0.013 (+0.002, +0.032) |
| per-predictor robust z | +0.068 (−0.010, +0.146) | +0.039 (+0.010, +0.074) |

So the change away from the plain z is supported on both metrics, and on complexes it is the largest effect here —
but it is not uniform: it lowered model 1 on 14 of 53 complexes. Against the simplest baseline, taking the highest
ranking_score across predictors, **the complex-level advantage is not resolvable at this sample size** (the
interval spans zero), and on monomers it is +0.013, which is real but small. Fifty-three complexes from one
campaign is a small, non-independent sample: the targets are of uneven difficulty, several are variants of the
same biological assembly, and the pools differ in size. Treat the ordering of the top three rules as unsettled and
the retreat from the plain z as the finding.

A z rewards a model for standing out within its own predictor, not for being right. Within a predictor, the inputs share one ordering
by ranking_score: putting each input on its own footing did worse. A person still reads the inputs separately when
a template may be the wrong one (the ipTM of a templated run is not comparable with a template-free one). Each predictor's own top model
set the order the others alternate in (mean DockQ over 57 complexes: af3 0.536, bt2 0.493, bt2_jax 0.473, pt2 0.448,
of3 0.397, ef2 0.369).

## The cases

| case | station | what happened | in residudes now |
|---|---|---|---|
| **T1331** | IDENTIFIED → INPUTS READY | AF3 found no homologue (query only); a family MSA of 26,835 sequences took AF3's median lDDT from 0.33 to 0.70 (model 1: 0.71, TM 0.86), and every predictor moved alike | `residudes msa --depth` flags it; `residudes msa --crawl --family …` rebuilds it (the replay's own crawl: 28,968 sequences, 27,319 covering half the query) |
| **T1357** | INPUTS READY | 9 sequences by default; plmMSA (OT alignment, SSB-focused) 2,345: median lDDT 0.59 → 0.88 | `residudes msa --plm`; intake: focus the query on the domain without homologs |
| **T2371** | INPUTS READY | 7 sequences (Neff 1.1); plmMSA 1,471: 0.37 → 0.87. ESMFold2 on the single sequence already reached 0.89 | intake: ef2 single-sequence as the cheap test of whether the MSA is the problem |
| **H2388** | INPUTS READY | "14-3-3 gamma bound to LARP1 derived phosphopeptide", no modification in the FASTA; the pS-X-P motif picked S7 of two serines; AF3 median DockQ 0.24 → 0.82, model 1 0.91. No note recorded the decision | intake: chemistry the FASTA does not carry, the steps and CCD codes; every variant is a `residudes decide` |
| **H2422, H2423** | IDENTIFIED | the complexes were released (9F8E, 7ACZ); as af3 multimer templates with the missing residues built: median DockQ 0.35 → 0.94, 0.78 → 0.92 | `residudes-modeller` takes a released hetero complex (one FASTA record per chain, `chain_map.tsv`) |
| **H2319** | IDENTIFIED → POOL READY | the reading supplied all of it: the antibody's epitope from escape selection in its own paper, the antigen's current apo crystals (missing from the default template search, which was redone: af3 0.24 → 0.73), the classical members' complexes rejected as docking templates, and the particle literature's dimer. The gates then took the pick to DockQ 0.93 (top ranking_score 0.67; best in pool 0.94) | intake: read the literature on each partner and check the template database's date ([`literature.md`](literature.md)); `residudes gate` types `rmsd`, `interface`, `contact`, `clash` |
| **H2321** | POOL READY | the counter-case to H2319. The same gates (paratope within 6 Å of S611/R615, spike core ≤ 1.0 Å from 8UFO, dimer intact, Fv paired, no clashes) left 246 of 12,810 models, and the submitted model 1 scored DockQ 0.85, the best in the pool. But that model **fails** the core gate at 1.123 Å and ranks 29th by the rubric; the rubric's top passer scores 0.58, and how model 1 was really chosen was never recorded - and it found the pool's best model, where the rubric would have taken a 0.58, so what was lost is a judgement that worked. The epitope gate alone does not separate DockQ here (median 0.57 among passers, 0.59 among failures) because wave DockQ weights each interface by its native contacts ([`metrics.md`](metrics.md)) and the spike dimer dominates | `residudes gate`; `residudes review` flags `site_prior_without_gate` and `site_without_source` |
| **T2417o** | HYPOTHESES | all four predictors peaked at A3 (ipTM 0.85–0.94) with no trimer template and a flat seq2symm; A3 was right. Its sibling T1416 went A1 for lack of precedent and was A3 | `residudes-stoichiometry`: absence is not evidence; several predictors at one n is |
| **T2461o** | INPUTS READY | 3VCD (63 % identical) has the 24-mer; MODELLER threaded T2461 onto it (protomer RMSD 0.07 Å): median DockQ 0.46 → 0.62 (3VCD directly) → 0.88 | `residudes-modeller` |
| **M1439** | SLATE → RETROSPECTIVE | ranking_score against lDDT ρ −0.15 to −0.24; from 15 to 1,920 models the pick stayed at lDDT 0.54 while the best in the sample rose to 0.63. The recognition core (nt 20–30) was found but no gate used it; model 1 DockQ 0.03 | `residudes gate` census over named sites; `site_split` trigger; consult: more seeds do not fix a ranker that cannot pick |
| **M2415** | POOL READY | a pocket census of 815 poses (bulge 444, Sm 186, RRM 110, other 75), the literature's bulge, then confidence: ligand 8.8 Å vs 51.8 Å for the top ranking_score | `residudes gate` census + `residudes slate --site bulge` |
| **M2377v1** | RETROSPECTIVE | runs with contact restraints and the JAX ports lost the RNA chain (310 models); model 1 had no selection record | `complete` in every gate result; `residudes review` flags `missing_chains`; the slate is a `residudes decide`. Run on the M1439 pool, the check found another one: the 40 Boltz-2 models of the RNA-secondary-structure variant hold only the protein |

## Where the numbers come from

Every figure on this page is recomputable, but most of the inputs are too large or too raw for the repository and
live on the DeepFold cluster. What exists, and where:

| claim | data | code |
|---|---|---|
| selection: gates vs confidence (56 targets, 37 complexes) | `RESIDUDES/talk/selection_vs_confidence/tables/*.csv`, 344 kB: one row per target per rule, with the field z-scores | `…/scripts/selection_vs_confidence.py` and `…_headline.py` |
| the slate's order (53 complexes, 31 monomers) | `$WORK_ROOT/ci/slate_policy.csv`, `zpolicy.csv`: model 1 per target under each rule | `$WORK_ROOT/ci/slate_policy.py`, `z_by_variant.py` |
| the gate audits (generic gates hurt) | `$WORK_ROOT/ci/generic_gates*.csv`, 1.4 MB: one row per model with each gate's value | `…/generic_gates*.py` |
| per-target scores behind every case | `gt_analysis/all/pool_scores.csv`, 67 MB: every scored pool model | `gt_analysis/` (OpenStructure, sbatch) |
| engine speed-ups | [`residudes-engine/docs/benchmark/results.csv`](../../residudes-engine/docs/benchmark/results.csv), 18 kB, **in this repository** | `residudes-engine/bench/` |

Not here, and not reproducible from this repository alone: the 1,337,148 pool models themselves (tens of TB of
`model.cif`), and any per-target score against a native that CASP has not yet released. `residudes export` strips the
latter unless asked for it, which is why the example dossiers carry analyses but no retrospectives.

## Where the stations are

The station names follow the dossier's stage machine (`residudes/AGENTS.md`):
IDENTIFIED → HYPOTHESES → PLAN REVIEW → INPUTS READY → RUNNING → POOL READY → RESULTS REVIEW → SLATE PROPOSED →
APPROVED → SUBMITTED → RETROSPECTIVE.
