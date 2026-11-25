# Motivation

A CASP season is not limited by how well any one predictor folds a sequence. Over 154 CASP17 targets the team
produced 1,337,148 models, and the gap between the best model in a pool and the model actually submitted was
larger than the gap between predictors: picking by raw confidence gave a mean DockQ of 0.564 over 37 complexes
([what the metrics mean](../residudes/docs/metrics.md)),
while the best model available in the same pools averaged 0.751. The work that closes that gap — identifying the
construct, reading what is known about the target, building an input that reflects it, and selecting against that
knowledge rather than against confidence alone — is slow, has to be repeated per target under a deadline, and is
rarely written down in a form that can be checked afterwards.

These three tools do that work and record it. The audience is a prediction team running a campaign with deadlines:
people who need the analysis behind each submitted model to be inspectable, repeatable, and attributable to the
evidence it rested on.

## Relation to other work

AlphaFold 3, Boltz-2, OpenFold 3, Protenix and ESMFold2 are the predictors; residudes-engine runs them behind one input format
rather than replacing them. Automated CASP pipelines usually select by the predictors' own confidence
(ranking_score, ipTM, pLDDT); residudes instead applies structural gates derived from what the analysis found about
the target and uses confidence only to order the models that pass, which on CASP17's released natives moved model 1
from a mean DockQ of 0.564 to 0.651 over the same pools (paired difference +0.088 over 37 complexes, Wilcoxon
p = 0.001; against af3's confidence alone, +0.059). Agent frameworks for science generally optimise for
autonomy; residudes is built the other way, stopping at two fixed checkpoints and refusing to approve or submit
anything itself.
