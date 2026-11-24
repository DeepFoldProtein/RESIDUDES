== Template: 14-3-3 gamma groove with a pS peptide (6A5S)

#grid(columns: (60%, 40%), gutter: 1em,
  image("figs/template_6a5s.jpg", height: 70%),
  [
  - 6A5S: human 14-3-3 gamma dimer (grey) with TFEB pS211 peptide (orange) in each groove, 2.1 Å.
  - Phosphate pocket (blue): K50, R57, R132, Y133 (native) = input B 49/56/131/132.
  - Plan gate: LARP1 peptide S7 (pS548, pS-L-P) phosphate within ~4 Å of the pocket, peptide in the groove, register N→C as mode I.
  - No 14-3-3:LARP1 structure is public; placement is by family prior.
  ])

== Plan at a glance

- Stoichiometry: A1B1 (given) submitted; A2B2 (native 14-3-3 dimer) run as a context hedge, cut to one B:A pair.
- Inputs: unmodified vs SEP at A7 (pS548); deep B MSA, peptide single-sequence; AF3 default templates.
- Pool: existing CASP17 run, ~1.4k models (af3 25 seeds x 5 x 3 variants x 2 sto; bt2_jax, of3). No new GPU now.
- Reserve ≤ 40 of 400 GPU-h: pt2 rerun (no model files) + of3/bt2 with SEP if registers differ.
- Selection: gate (pS7 in pocket) first, confidence second.
