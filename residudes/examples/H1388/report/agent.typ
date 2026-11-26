== The closest public pose: 6S9K (14-3-3 gamma + caspase-2 pS139 peptide)

#grid(columns: (55%, 45%), gutter: 1em,
  image("figs/pocket_6s9k.png", height: 72%),
  [
  - Grey: 14-3-3 gamma (100 % identical to chain B), 1.60 Å.
  - Orange: the caspase-2 phosphopeptide, *DL-pS-L-P-F*.
  - Blue: the phosphate pocket, K50 R57 R132 Y133 (native) = B49 B56 B131 B132 (input).
  - LARP1 is *SA-pS-L-P-D*: it shares pS, +1 Leu and +2 Pro with caspase-2. This is the register prior.
  - Use: a pinned template for chain B (D4), and the pocket gate (pS7 phosphate near B56/B131/B132).
  ])

== First results: where each predictor puts the LARP1 peptide

#grid(columns: (58%, 42%), gutter: 1em,
  image("figs/pep_split.png", height: 72%),
  [
  Most confident model per arm, superposed on 14-3-3 gamma (grey; blue sticks = pocket B49/56/131/132; on the peptides, residue 7).
  - #text(fill: rgb("#D55E00"))[af3 + pS7] (s20_0) and #text(fill: rgb("#009E73"))[pt2] (s5_0): one register, core A4-A10 median 0.56 / 0.57 Å apart.
  - #text(fill: rgb("#CC79A7"))[bt2] (s1_0): 10.5 Å off; arm median 3.05 Å.
  - #text(fill: rgb("#E69F00"))[af3 unmodified] (s5_0): 7.9 Å median, register shifted.
  - #text(fill: rgb("#56B4E9"))[of3] (s6_0): 12 Å, ipTM 0.67.
  - Only af3 carried the phosphate; pt2 agrees without it.
  ])
