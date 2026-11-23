# The metrics these docs quote

Every number elsewhere in this repository is one of the measures below. Scores against a released structure are
computed with OpenStructure; confidences are written by the predictors themselves and measure nothing about the
true structure.

## Measured against a reference structure

| metric | what it measures | range | reference |
|---|---|---|---|
| **lDDT** | agreement of local inter-atomic distances, without superposition, so a domain that moves does not sink the rest | 0–1 | Mariani et al. 2013, *Bioinformatics* 29(21):2722, [doi:10.1093/bioinformatics/btt473](https://doi.org/10.1093/bioinformatics/btt473) |
| **DockQ** | one interface, combining the fraction of native contacts with ligand and interface RMSD | 0–1; ≥ 0.23 acceptable, ≥ 0.49 medium, ≥ 0.80 high | Basu & Wallner 2016, *PLOS ONE* 11(8):e0161879, [doi:10.1371/journal.pone.0161879](https://doi.org/10.1371/journal.pone.0161879) |
| **DockQ_wave** | an assembly with several interfaces: per-interface DockQ averaged with each interface weighted by its number of native contacts, so a large interface counts for more than a small one. CASP's own `dockq_wave` field, and what "mean DockQ" means in these docs | 0–1 | as DockQ |
| **QS-score** | how much of the reference's set of inter-chain contacts the model reproduces | 0–1 | Bertoni et al. 2017, *Sci Rep* 7:10480, [doi:10.1038/s41598-017-09654-8](https://doi.org/10.1038/s41598-017-09654-8) |
| **ICS, IPS** | interface contact similarity and interface patch similarity, the CASP assembly measures | 0–1 | defined by the CASP assembly assessment; computed with OpenStructure |
| **TM-score** | global fold similarity, length-normalised | 0–1; > 0.5 is the same fold | computed with USalign |

The weighting was read back out of the assessment output rather than assumed: on one CASP17 model with seven
interfaces, weighting the seven per-interface DockQ values by `nnat` reproduces `dockq_wave` exactly (0.687, against 0.534
for the unweighted `dockq_ave`).

The scores come from OpenStructure's `compare-structures` (Biasini et al. 2013, *Acta Cryst D* 69(5):701,
[doi:10.1107/S0907444913007051](https://doi.org/10.1107/S0907444913007051); first described in *Bioinformatics*
2010, [doi:10.1093/bioinformatics/btq481](https://doi.org/10.1093/bioinformatics/btq481)).

**A trap worth knowing.** CASP TS files carry header records that OpenStructure does not expect; left in, ICS and
IPS come out as 0 with no error, which looks like a failed prediction rather than a failed parse. Strip the
headers first.

## Predicted by lantern-qa

`xgb_pred` is a predicted DockQ: a gradient-boosted model fitted on features of the model itself against DockQ
measured on released structures, so unlike a confidence it is on the same scale as the thing it predicts, and the
same scale for every predictor. Held out by target (leave-one-protein-out) it reaches Pearson *r* 0.79 against
measured DockQ ([lantern-qa](../../lantern-qa/docs/interface/README.md), which also gives the targets where it
failed: ρ = −0.28 on H2442). It is what the report's slate table shows beside each model, and what the clustering
page gives per cluster ("best predicted DockQ"), so the alternatives a slate hedges across can be compared on a
scale that means the same thing for every predictor.

A third limit, measured: it does not separate what confidence cannot. On T1417's three trimer architectures the
interface score's medians sit within 0.002 of each other for two of them on af3, and on of3 it ranks them in a
different order than ipTM does (measured on T1417's own pool; that analysis stays with the dossier, because it is
scored against a native under embargo). A score that is better at ordering models
is not therefore able to answer a question the pool does not contain.

Two limits worth stating. Correlation is not calibration: *r* 0.79 says the ordering is good on average, not that
0.6 means DockQ 0.6. And `lantern slate` currently blends it half and half with the predictor's own confidence;
which weighting is best has not been measured on the CASP17 pools, because the cached pool scores carry the
confidences and the native scores but no interface predictions.

## Written by the predictor

`ranking_score`, `ipTM`, `pTM` and `pLDDT` are each model's own estimate of itself. They are useful for ordering
models from one predictor and they are **not** measurements of accuracy.

They are also not comparable between predictors: over 588,291 CASP17 pool models the median `ranking_score` was
0.84 for Boltz-2, 0.67 for AlphaFold 3, 0.58 for OpenFold 3, 0.58 for Protenix v2 and 0.47 for ESMFold2. Ranking
one pool by raw confidence across predictors therefore mostly ranks by which predictor wrote the number, which is
why `lantern slate` compares within a predictor and alternates between them
([`casp17_cases.md`](casp17_cases.md)).

One more incomparability, within a single predictor: a templated run and a template-free run of the same target do
not share an ipTM scale, because a template raises the predicted interface confidence whether or not the template
was the right one. Judge a template variant by af3 models against af3 models, never across the two.
