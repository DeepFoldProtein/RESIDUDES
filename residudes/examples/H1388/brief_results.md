# H1388 results review (A1B1: LARP1 pS548 11-mer + 14-3-3 gamma)

Deck: <WORK>/targets/H1388/report/report.pdf (last agent slide: peptide placement per predictor).

## What the pool shows (300 models, 0 empty, 0 clashes, both chains in every run)
- Confidence by arm (ipTM median / max): af3+pS7 0.917/0.939; pt2 0.905/0.92; bt2 0.883/0.908; af3 unmodified 0.552/0.69; of3 0.473/0.67 (evidence qa, pool).
- Gates (evidence gates, gates_breakdown): phospho pocket (A7 within 6 A of >=2 of B49/56/131/132) 235/300: af3+pS7 100/100, pt2 50/50, bt2 46/50, of3 33/50, af3 unmodified 6/50. Motif and assembly gates 300/300.
- Peptide placement (evidence peptide_agreement): on receptor superposition to A1B1__sep__af3__s20_0, the peptide core A4-A10 of af3+pS7 (median 0.56 A, 99% < 2 A) and pt2 (0.57 A, 100% < 2 A) coincide. bt2 is 3.05 A off at median and its most confident model 10.5 A away; of3 6.8 A; unmodified af3 7.9 A.
- Only af3 carried SEP into its models; pt2/bt2/of3 wrote SER7 (evidence pool). So pt2's agreement with af3+pS7 comes without the phosphate and without the same inputs, which is independent support for the register. ipTM between af3+pS7 and the others is not comparable.

## Triggers
- missing_models: a false positive. The check expects 100 per `sep` cell because af3 ran 20 seeds there; D6 planned 10 seeds x5 for pt2/bt2/of3 and all 50 are present.
- peptide_register_split (added): settled by the pocket gate and by the af3+pS7 / pt2 agreement, the register the 6S9K caspase-2 pS-L-P analog shows (evidence peptide_analog, motif, binding_site).

## Options
- (a) proceed to the slate [recommended]: gated models only; af3+pS7 leads (3 models from distinct seeds), pt2 2 models; bt2/of3/unmodified af3 left out unless they pass the pocket gate AND sit within 2 A core of the consensus. No GPU cost.
- (b) loop:RUNNING: more pt2/bt2 seeds. Does not help: the pool already has >100 gate-passing models in one register; the ranker, not sampling, is the question.
- (c) loop:HYPOTHESES: rerun pt2/bt2 with SEP wired through their own converters (~5 GPU-h). Would only add confidence to a register already agreed.

## Default
Under the mandate: (a).
