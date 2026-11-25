#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#set text(font: "DejaVu Sans", size: 17pt)
#show raw: set text(font: "DejaVu Sans Mono", size: 11pt)
#set figure(numbering: none, gap: 4pt)
#show figure.caption: set text(size: 10pt)

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(title: [#"H1388: target report"], subtitle: [#"stage SLATE_PROPOSED"],
    author: [residudes · DeepFold], date: [#"2026-11-26"]),
  config-common(new-section-slide-fn: none),
)

#title-slide()

== In one page

#text(size: 14pt)[
- #text(fill: rgb("#D55E00"), weight: "bold")[#"What it is"] #"A (11 aa) subunit1 — Homo sapiens (Eukaryota; Metazoa; Chordata; Mammalia; Primates); Q6PKG0 (LARP1_HUMAN), residues 542-552; La-related protein 1 (LARP1); the peptide is an intrinsically disordered linker segment, n · B (233 aa) subunit2 — Homo sapiens (Eukaryota; Metazoa; Chordata; Mammalia; Primates); P61981 (1433G_HUMAN, YW…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"The chemistry the FASTA does not carry"] #"A7 phosphoserine (CCD SEP) — S7 chosen; S5 rejected"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"MSA depth"] #"deepest `B` 10,377 sequences, thinnest `A` 0, over 2 measured alignments"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"Structures that exist"] #"24 PDB entries named in the record: `2b05`, `3uzd`, `4e2e`, `4j6s`, `4o46`, `5d3e`, `6a5s`, `6byj` …"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"The pool"] #"300 models in 12 clusters. cluster 6 is sep and nothing else (39 of the 250 models that variant has): that input reaches a shape the others do not"
  - #"cluster 2 is sep and nothing else (34 of the 250 models that variant has): that input reaches a shape the others do not"
  - #"cluster 7 is sep and nothing else (31 of the 250 models that variant has): that input reaches a shape the others do not"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"What nobody knows"] #"[\"No paper on the 14-3-3:LARP1 pS548 interaction or its structure was found (Europe PMC: '\\\"LARP1\\\" AND \\\"14-3-3\\\"', 'LARP1 AND Ser548', 'LARP1 AND (14-3-3 OR YWHA*)'); the organisers' description is not in the dossier, so whether the peptide was phosphorylated in the experiment is not confirmed here.\", \"Which kinase phosphoryl…"
]

== The target

#text(size: 12pt)[
- *construct*: #"A: length: 11; native region: LARP1 Q6PKG0 542-552; numbering: input = native - 541; input S…; B: length: 233; native region: P61981 3-234 (native length 247; C-terminal 235-…"
- *identity*: #"A: organism: Homo sapiens (Eukaryota; Metazoa; Chordata; Mammalia; Primates); uniprot: Q6PKG…; B: organism: Homo sapiens (Eukaryota; Metazoa; Chordata; Mammalia; Primates); u…"
- *public structures*: #"23 PDB entries; near-identical 2b05 (B), 3uzd (B), 4e2e (B), 4j6s (B), 4o46 (B), 5d3e (B), 6a5s (B), 6byj (B), 6byl (B), 6fel (B), 6gkf (B), 6gkg (B), 6s9k (B), 6sad (B), 6y4k (B), 6y6b (B), 6zbt (B), 6zc9 (B), 7a6r (B), 7a6y (B), 8xi3 (B), 9ci3 (B), 6bzd (B)"
- *templates*: #"B colabfold: 6byj_A 100% id, qcov 2-233/233, E 1.5e-116; 6y6b_B 100%; 6bzd_B 99.5% (R57E); 6zc9_A 99.1…; A: none (peptide); hhsearch: not run this step: chain B already has 2…"
- *stoichiometry given*: #"A1B1"
]

== Further priors and checks

#text(size: 12pt)[#table(columns: (auto, auto, 1fr), stroke: 0.4pt + luma(200), inset: 5pt,
  [*prior*], [*found at*], [*what*],
  [#"literature"], [#"NEW"], [#"claims: claim: 14-3-3 proteins bind phosphoserine-containing protei…, claim: 14-3-3 acts as a dim…; not known: No paper on the 14-3-3:LARP1 pS548 interaction or its struc…, Which kinase phosphorylates…; search…"],
  [#"binding_site"], [#"NEW"], [#"chain: B; residues: 49, 56, 131, 132; what: 14-3-3 gamma phosphate pocket in the amphipathic groove (native K50, R57, R132, Y133): th…; source: UniProt P61981 sites R57/R132; checked in PDB 6S9K (14-3-3 gamma …"],
  [#"chemistry"], [#"NEW"], [#"modification: phosphoserine (CCD SEP) on chain A residue 7 (LARP1 Ser548); chain: A; residues: 7; candidates: S7: chosen: only serine followed by X-Pro (S7-L8-P9, pS-X-P), and UniProt Q6PKG0 records …; motif: …"],
  [#"motif"], [#"NEW"], [#"chain: A; residues: 7, 8, 9; pattern: pS-X-P; what: 14-3-3 binding core on the LARP1 peptide (S7-L8-P9); source: Yaffe et al. 1997, PMID 9428519"],
  [#"stoichiometry_evidence"], [#"IDENTIFIED"], [#"given: A1B1; given source: organisers' declared stoichiometry (dossier stoichiometry.given); family: 14-3-3 is a constitutive homodimer (PMID 9428519); each protomer has its own amphipathic …; homolog assembli…"],
  [#"peptide_analog"], [#"PLAN_REVIEW"], [#"entry: 6S9K; chain receptor: A (14-3-3 gamma); chain peptide: B (caspase-2); peptide seqres start: DYDL(SEP)LPFPVCESCPLYKKLRLSTDT; core match: 6S9K caspase-2 pS139-L-P and LARP1 pS548-L-P (input A7-L8-P9) shar…"],
  [#"inputs"], [#"INPUTS_READY"], [#"built: 2026-11-26; base: file: <WORK>/targets/H1388/inputs/base/input.json; sto: A1B1; msa:…; sep: file: <WORK>/targets/H1388/inputs/sep/input.json; same as: base;…"],
  [#"gates_breakdown"], [#"POOL_READY"], [#"spec: <WORK>/targets/H1388/gates/H1388_A1B1.json; csv: <WORK>/targets/H1388/gates/pocket_A1B1.csv; job: 348951; pass all: 235/300; phospho pocket pass: sep/af3: 10…"],
  [#"qa"], [#"POOL_READY"], [#"sto: A1B1; interface: job: 348948; file: <WORK>/targets/H1388/qa/interface_A1B1/xgb_dock…; cluster: job: 348952; dir: <WORK>/targets/H1388/qa/cluster_A1B1/A1B1/poo…"],
)]

== The chains, and the residues a gate would use

#align(center)[#image("figs/chain_map.svg", width: 92%)]
#text(size: 11pt)[#"A: subunit1 11 aa · B: subunit2 233 aa"]

== What the literature settles, and what it does not

#align(center)[#image("figs/literature.svg", height: 76%)]
#text(size: 11pt)[#"7 claims, 7 with a source · not settled: [\"No paper on the 14-3-3:LARP1 pS548 interaction or its structure was found (Europe PMC: '\\\"LARP1\\\" AND \\\"14-3-3\\\"', 'LARP1 AND Ser548', 'LARP1 AND (14-3-3 OR YWHA*)'); the organisers' description is not in the dossier, so whethe…"]

== The sites the analysis found, residue by residue

#align(center)[#image("figs/sites.svg", height: 76%)]
#text(size: 11pt)[#"binding_site B: K49, R56, R131, Y132 · motif A: S7, L8, P9 · phosphoserine (CCD SEP) on A7"]

== The motifs, and every place they can sit

#align(center)[#image("figs/motif.svg", width: 92%)]
#text(size: 11pt)[#"pS-X-P on A: S7"]

== MSA depth and coverage

#align(center)[#image("figs/msa_coverage.svg", width: 92%)]
#text(size: 12pt)[#"A": 0 sequences, 0 ≥50% coverage · #"B": 10,377 sequences, 8,105 ≥50% coverage]

== The pool: confidence per stoichiometry and predictor

Best (and median of the top 10) ipTM for complexes, pTM for monomers. Darker is more confident.

#text(size: 12pt)[#table(columns: 2, stroke: 0.4pt + white, inset: 6pt, align: center,
  [], [*#"A1B1"*],
  [*#"af3"*], table.cell(fill: rgb(15, 122, 182))[#text(fill: white)[#"0.94 (0.93)"]],
  [*#"bt2"*], table.cell(fill: rgb(23, 126, 185))[#text(fill: white)[#"0.91 (0.90)"]],
  [*#"of3"*], table.cell(fill: rgb(84, 160, 203))[#text(fill: white)[#"0.67 (0.61)"]],
  [*#"pt2"*], table.cell(fill: rgb(20, 125, 184))[#text(fill: white)[#"0.92 (0.92)"]],
)]

== The pool, clustered: the alternatives a slate hedges across

#align(center)[#image("figs/clusters.svg", height: 76%)]
#text(size: 11pt)[#"300 models, 12 clusters in 2 input variants · best predicted DockQ per cluster: sep:6 0.81, sep:10 0.81, sep:9 0.80"]

== Decisions and the priors they rest on

#text(size: 10pt)[#table(columns: (auto, auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 4pt,
  [*id*], [*type*], [*chose*], [*rests on*],
  [#"D1"], [#"variant"], [#"phosphoserine (SEP) on chain A residue 7 (LARP1 S548), af3 only, run next to the unmodified input"], [#"construct, literature, binding_site, chemistry, motif"],
  [#"D2"], [#"stoichiometry"], [#"A1B1 (declared): one LARP1 phosphopeptide in the groove of one 14-3-3 gamma protomer; all five slate models A1B1"], [#"public_structures, literature, binding_site, motif, stoichiometry_evi…"],
  [#"D3"], [#"variant"], [#"MSA: ColabFold MSA for chain B (14-3-3 gamma, 10,377 seqs, deep); chain A (11-mer LARP1 peptide) single-sequence (…"], [#"identity, construct, msa_depth"],
  [#"D4"], [#"variant"], [#"Templates: chain B search templates (top 4) plus 6S9K chain A pinned at rank 1 (1.60 A, 100% id, phosphopeptide-bo…"], [#"public_structures, literature, binding_site, templates"],
  [#"D5"], [#"variant"], [#"PTM wiring: phosphoserine as an AF3/Protenix modification {ptmType: SEP, ptmPosition: 7} on chain A, added to the …"], [#"construct, chemistry, motif"],
  [#"D6"], [#"sampling"], [#"Sampling, all A1B1 (244 residues): af3 SEP 20 seeds x5 samples (lead arm); af3 unmodified 10 seeds x5 (control); p…"], [#"binding_site, chemistry, motif, stoichiometry_evidence, msa_depth, te…"],
  [#"D7"], [#"selection"], [#"Slate v2: A1B1__sep__af3__s20_0, A1B1__sep__pt2__s5_0, A1B1__sep__af3__s9_0, A1B1__sep__pt2__s10_0, A1B1__sep__af3…"], [#"binding_site, motif, peptide_analog, gates, qa, peptide_agreement, sl…"],
)]

== Checkpoint C1 (plan) at PLAN REVIEW

#text(size: 12pt)[
+ #"Q1 pocket gate: (a) pS7 phosphate within 4 A of B56/B131/B132 [recommended: 6S9K contacts these, not K49] (b) require B49 too"
+ #"Q2 dimer: (a) A1B1 only as declared [recommended] (b) add 5 af3 A2B2 seeds as a geometry check, ~+1 GPU-h, not submitted"
+ #"Q3 template: (a) pin 6S9K chain A in af3 lead arm, unpinned control alongside [recommended: pS-L-P analog, 1.6 A] (b) no pinned template"
+ #"Q4 unmodified arm: (a) eligible for the slate only if it passes the pocket gate [recommended] (b) slate from SEP models only"
]
#text(size: 11pt)[
- #text(fill: rgb("#2a2a2e"), weight: "bold")[agent]: #"Answer under deepfold's mandate. Q1 (a): gate on the pS7 phosphate within 4 A of B56/B131/B132; pass = at least 2 of the 3 (tolerates arginine rotamer variance; 6S9K has all three at <=3.5 A); the census still reports B49. B49 (K50) is not a contact in 6S9K (binding_site, peptide_analog), so requiring it would reject native-lik…"
- #text(fill: rgb("#D55E00"), weight: "bold")[resolved by a person]: #"proceed"
]

== Checkpoint C2 (results) at RESULTS REVIEW

#text(size: 12pt)[
+ #"Slate route: (a) proceed with gated af3+pS7 (3) and pt2 (2) in the consensus register [recommended: two predictors agree at 0.56 A core and the pocket gate passes]; (b) loop:RUNNING more seeds; (c) loop:HYPOTHESES rewire SEP into…"
+ #"missing_models trigger: treat as false positive (pt2/bt2/of3 planned at 10 seeds, 50/50 present) [recommended: yes]"
]
#text(size: 11pt)[
- #text(fill: rgb("#2a2a2e"), weight: "bold")[agent]: #"Answer under the mandate. Q1: (a) proceed. The pool has one register held by two predictors with different inputs (af3 with pS7 and the pinned 6S9K template; pt2 with SER7): core A4-A10 median 0.56/0.57 A to A1B1__sep__af3__s20_0, and every af3+pS7 and pt2 model passes the phospho-pocket gate (150/150). That register is the one…"
- #text(fill: rgb("#D55E00"), weight: "bold")[resolved by a person]: #"proceed"
]

== First results: the review

- #"missing_models: A1B1/sep/bt2 50/100, A1B1/sep/of3 50/100, A1B1/sep/pt2 50/100"
- #"missing_models_false_positive: the trigger expects 100 per sep cell because sep/af3 ran 20 seeds; D6 planned 10 seeds x5 for pt2/bt2/of3, and pool records expected 50 = n 50 for each, 0 empty, 0 clas…"
- #"peptide_register_split: on receptor superposition, af3-SEP and pt2 put the peptide core A4-A10 in one register (median 0.56/0.57 A to A1B1__sep__af3__s20_0, 99-100% under 2 A); bt2 median 3.05 A and …"
- #"ptm_not_carried: only af3 wrote SEP at A7; pt2/bt2/of3 sep runs wrote SER7 (evidence pool). pt2 still reproduces the af3-SEP register without the phosphate; ipTM is not comparable between af3-SEP and…"

== Structural gates #"pocket" at #"A1B1"

#text(size: 20pt)[*235* of 300 models pass every gate]

#text(size: 14pt)[#grid(columns: (1.5fr, 1fr), gutter: 18pt,
  table(columns: (auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 6pt, [*gate*], [*checks*], [*pass*],
    [#"complete"], [#"all chains present"], [#"300"],
    [#"phospho_pocket"], [#"A:7 within 6.0 Å of B:49,56,131,132 (≥ 2 site residues)"], [#"235"],
    [#"motif"], [#"B within 6.0 Å of A:7,8,9 (≥ 3 site residues)"], [#"300"],
    [#"assembled"], [#"interface A : B within 5.0 Å, ≥ 10 residue contacts"], [#"300"],
  ),
  table(columns: (auto, auto, auto), stroke: 0.4pt + luma(200), inset: 6pt, [*predictor*], [*models*], [*pass*],
    [#"af3"], [#"150"], [#"106"],
    [#"bt2"], [#"50"], [#"46"],
    [#"of3"], [#"50"], [#"33"],
    [#"pt2"], [#"50"], [#"50"],
  ),
)]

== The slate: structures

Pictures are being rendered (`residudes slides` again when the render job is done).

== Slate v2 (chosen by review)

#text(size: 12pt)[#table(columns: 6, stroke: 0.4pt + luma(200), inset: 5pt,
  [*slot*], [*model*], [*runner*], [*ipTM*], [*z (runner)*], [*interface*],
  [#"1"], [#"A1B1__sep__af3__s20_0"], [#"af3"], [#"0.939"], [#"1.1"], [#"0.768"],
  [#"2"], [#"A1B1__sep__pt2__s5_0"], [#"pt2"], [#"0.92"], [#"0.6"], [#"0.7037"],
  [#"3"], [#"A1B1__sep__af3__s9_0"], [#"af3"], [#"0.933"], [#"0.85"], [#"0.7617"],
  [#"4"], [#"A1B1__sep__pt2__s10_0"], [#"pt2"], [#"0.912"], [#"0.3"], [#"0.6947"],
  [#"5"], [#"A1B1__sep__af3__s7_0"], [#"af3"], [#"0.931"], [#"0.75"], [#"0.6996"],
)]

#include "agent.typ"

== What the analysis read

#text(size: 11pt)[#table(columns: (auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 5pt,
  [*\#*], [*source*], [*supports*],
  [#"1"], [#"UniProt P61981 sites R57/R132; checked in PDB 6S9K (14-3-3 gamma + caspase-2 phosphopeptide, 1.60 A): SEP139 phosphate oxygens within 3.5 A of R57, R…"], [#"literature, binding_site, chemistry, mo…"],
  [#"2"], [#"Yang et al. 2006 PNAS, PMID 17085597, PMC1859916 (abstract); the gamma entry from this work is 2B05 (inferred from release date and title, not checke…"], [#"literature"],
  [#"3"], [#"UniProt P61981 Site features"], [#"literature"],
  [#"4"], [#"UniProt Q6PKG0 Modified residue 548"], [#"literature"],
  [#"5"], [#"residudes structures H1388 (RCSB sequence search, 2026-11-26)"], [#"literature"],
  [#"6"], [#"organisers' declared stoichiometry (dossier stoichiometry.given)"], [#"stoichiometry_evidence"],
  [#"7"], [#"receptor (chain B) CA superposition onto the top-ranking_score af3 SEP model A1B1__sep__af3__s20_0, then peptide chain A CA RMSD (all 11 and core A4-…"], [#"peptide_agreement"],
  [#"8"], [#"A1B1__sep__af3__s20_0"], [#"peptide_agreement, slate_consensus"],
)]
#text(size: 11pt)[*No source recorded*: #"public_structures, identity, construct, templates, inputs, gates_breakdown, qa"]

== What this proposes, and what it is asking

#text(size: 13pt)[
- #text(fill: rgb("#D55E00"), weight: "bold")[#"variant"] #"phosphoserine (SEP) on chain A residue 7 (LARP1 S548), af3 only, run next to the unmodified input; MSA: ColabFold MSA for chain B (14-3-3 gamma, 10,377 seqs, deep); chain A (11-mer LARP1 peptide) single-sequence (empty unpaired MSA), no paired MSA …; Template…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"stoichiometry"] #"A1B1 (declared): one LARP1 phosphopeptide in the groove of one 14-3-3 gamma protomer; all five slate models A1B1"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"sampling"] #"Sampling, all A1B1 (244 residues): af3 SEP 20 seeds x5 samples (lead arm); af3 unmodified 10 seeds x5 (control); protenix SEP 10 seeds x5; boltz-2 10…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"selection"] #"Slate v2: A1B1__sep__af3__s20_0, A1B1__sep__pt2__s5_0, A1B1__sep__af3__s9_0, A1B1__sep__pt2__s10_0, A1B1__sep__af3__s7_0. All pass the pocket gate an…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"Best model in the pool so far"] #"by confidence `A1B1__sep__af3__s20_0` (0.963) · by predicted DockQ `A1B1__sep__af3__s4_1` (0.811) — not the same model, which is what the slate's two scales are for"
]

== What the agent spent

#text(size: 13pt)[#table(columns: 4, stroke: 0.4pt + luma(200), inset: 5pt,
  [*stage*], [*steps*], [*turns*], [*USD*],
  [#"NEW"], [#"1"], [#"22"], [#"0.96"],
  [#"IDENTIFIED"], [#"1"], [#"6"], [#"0.26"],
  [#"HYPOTHESES"], [#"1"], [#"11"], [#"0.35"],
  [#"PLAN_REVIEW"], [#"5"], [#"54"], [#"1.53"],
  [#"INPUTS_READY"], [#"1"], [#"19"], [#"0.54"],
  [#"RUNNING"], [#"1"], [#"19"], [#"0.49"],
  [#"POOL_READY"], [#"2"], [#"10"], [#"0.87"],
  [*total*], [#"12"], [#"141"], [#"5.00"],
)]

