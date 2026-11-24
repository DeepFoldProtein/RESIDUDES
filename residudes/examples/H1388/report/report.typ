#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#set text(font: "DejaVu Sans", size: 17pt)
#show raw: set text(font: "DejaVu Sans Mono", size: 11pt)
#set figure(numbering: none, gap: 4pt)
#show figure.caption: set text(size: 10pt)

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(title: [#"H1388: target report"], subtitle: [#"stage PLAN_REVIEW"],
    author: [lantern · DeepFold], date: [#"2026-11-23"]),
  config-common(new-section-slide-fn: none),
)

#title-slide()

== In one page

#text(size: 14pt)[
- #text(fill: rgb("#D55E00"), weight: "bold")[#"What it is"] #"A (11 aa) subunit1 — Q6PKG0; La-related protein 1 (LARP1), Homo sapiens, residues 542-552 · B (233 aa) subunit2 — P61981; 14-3-3 protein gamma (YWHAG), Homo sapiens; 14-3-3 protein family; Pfam PF00244; InterPro IPR000308, IPR023409, IPR023410, IPR036815. Given stoichiometry A1B1."
- #text(fill: rgb("#D55E00"), weight: "bold")[#"The chemistry the FASTA does not carry"] #"A7 phosphoserine (CCD SEP) — S7 chosen; S5 rejected"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"MSA depth"] #"deepest `af3/uniprot.a3m` 16,347 sequences, thinnest `af3/bfd.a3m` 1,607, over 4 measured alignments"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"Structures that exist"] #"33 PDB entries named in the record: `11hg`, `11jj`, `2b05`, `2br9`, `3iqu`, `3uzd`, `4e2e`, `4j6s` …"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"The pool"] #"875 models in 9 clusters. cluster 1 is af3_msa_pS548 and nothing else (125 of the 125 models that variant has): that input reaches a shape the others do not"
  - #"2 clusters hold models from three or more predictors; the largest is 7 (bt2_jax 59, of3 39, af3 24)"
  - #"one predictor only: cluster 0 (bt2_jax, 133), cluster 1 (af3, 125), cluster 6 (of3, 58), cluster 4 (of3, 31) — a shape no other predictor found"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"What nobody knows"] #"No paper or structure found on the 14-3-3:LARP1 complex itself (Europe PMC 'LARP1 14-3-3' 50 hits, none on the complex; one web search). The FASTA names no modification; the CASP description text was not available to this step - if it says phosphopeptide, the pSer is S7. Whether one or two peptides bind the 14-3-3 dimer in the …"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"Open, and waiting on a person (C1)"] #"(1) Phosphorylation; (2) Stoichiometry; (3) GPU; (4) Selection gate"
]

== The target

#text(size: 12pt)[
- *construct*: #"chains: A (subunit1, 11 aa): seq: PKGLSASLPDL; maps to: human LARP1 (UniProt Q6PKG0) residues 542…; given stoichiometry: A1B1"
- *identity*: #"B: uniprot: P61981; name: 14-3-3 protein gamma (YWHAG), Homo sapiens; family: 14-3-3 protein…; A: uniprot: Q6PKG0; name: La-related protein 1 (LARP1), Homo sapiens, residues …"
- *public structures*: #"exact sequence sweep: method: RCSB sequence search, chain B, identity>=95%, e<1e-10, 2026-11-23; n entities: 23…; peptide: method: RCSB seqmotif GLSASLPD + full-text 'LARP1 1…"
- *templates*: #"B: 23 exact human 14-3-3 gamma PDB structures (see public_structures); best peptide-bound: 6…; A: no template; peptide placement comes from the 14-3-3 groove (binding_site/mo…"
- *stoichiometry given*: #"A1B1"
]

== Further priors and checks

#text(size: 12pt)[#table(columns: (auto, auto, 1fr), stroke: 0.4pt + luma(200), inset: 5pt,
  [*prior*], [*found at*], [*what*],
  [#"literature"], [#"NEW"], [#"claims: claim: 14-3-3 proteins bind phosphoserine motifs; mode I co…, claim: 14-3-3 is a cup-shap…; not known: No paper or structure found on the 14-3-3:LARP1 complex itself (Europe PMC 'LARP1 14-3-3'…; search…"],
  [#"binding_site"], [#"NEW"], [#"chain: B; residues: 49, 56, 131, 132; what: phosphate pocket of 14-3-3 gamma (input numbering; native K50 R57 R132 Y133); peptide cha…; source: Yaffe et al. 1997 PMID 9428519; conserved in 14-3-3 gamma:phospho…"],
  [#"motif"], [#"NEW"], [#"chain: A; residues: 7; what: LARP1 S548 = peptide S7, 14-3-3 mode-I-like pS-X-P (S7-L8-P9); S5 rejected (S-A-S, no Pro…; source: UniProt Q6PKG0 (S548 phosphoserine); PMID 9428519"],
  [#"chemistry"], [#"NEW"], [#"modification: phosphoserine (CCD SEP) on chain A residue 7 (LARP1 S548); candidates: S7: chosen: pS-X-P motif, UniProt-annotated phosphosite; S5: rejected: S-A-S has no Pro a…; basis: 14-3-3 binds phosphorylat…"],
  [#"stoichiometry_evidence"], [#"IDENTIFIED"], [#"given: A1B1 (CASP; chain A = LARP1 peptide 11 aa, chain B = 14-3-3 gamma 3-233); homolog assemblies: method: RCSB assembly 1 + rcsb_struct_symmetry, 2026-11-23; entries: 2B05: tetrameric, A2…; family prior: 14…"],
  [#"existing_pool"], [#"HYPOTHESES"], [#"root: /gpfs/deepfold/casp/casp17/targets/H1388/results; found: existing pool (CASP17 run, 2026-06) for A1B1 and A2B2; input variants af3_msa, af3_msa_pS…; per combo: af3: 25 seeds x 5 samples per sto/variant (…"],
)]

== The chains, and the residues a gate would use

#align(center)[#image("figs/chain_map.svg", width: 92%)]
#text(size: 11pt)[#"A: subunit1 11 aa · B: subunit2 233 aa"]

== What the literature settles, and what it does not

#align(center)[#image("figs/literature.svg", height: 76%)]
#text(size: 11pt)[#"5 claims, 5 with a source · not settled: No paper or structure found on the 14-3-3:LARP1 complex itself (Europe PMC 'LARP1 14-3-3' 50 hits, none on the complex; one web search). The FASTA names no modification; the CASP description text was not available to this step - …"]

== The sites the analysis found, residue by residue

#align(center)[#image("figs/sites.svg", height: 76%)]
#text(size: 11pt)[#"binding_site B: K49, R56, R131, Y132 · motif A: S7 · phosphoserine (CCD SEP) on A7"]

== The motifs, and every place they can sit

#align(center)[#image("figs/motif.svg", width: 92%)]
#text(size: 11pt)[#"pS-X-P on A: S7 · S-A-S on A: S5"]

== MSA depth and coverage

#align(center)[#image("figs/msa_coverage.svg", width: 92%)]
#text(size: 12pt)[#"af3/uniref90.a3m": 6,536 sequences, 4,692 ≥50% coverage · #"af3/uniprot.a3m": 16,347 sequences, 13,544 ≥50% coverage · #"af3/mgnify.a3m": 4,332 sequences, 1,860 ≥50% coverage · #"af3/bfd.a3m": 1,607 sequences, 957 ≥50% coverage]

== The pool: confidence per stoichiometry and predictor

Best (and median of the top 10) ipTM for complexes, pTM for monomers. Darker is more confident.

#text(size: 12pt)[#table(columns: 3, stroke: 0.4pt + white, inset: 6pt, align: center,
  [], [*#"A1B1"*], [*#"A2B2"*],
  [*#"af3"*], table.cell(fill: rgb(15, 122, 182))[#text(fill: white)[#"0.94 (0.94)"]], table.cell(fill: rgb(17, 123, 183))[#text(fill: white)[#"0.93 (0.92)"]],
  [*#"bt2_jax"*], table.cell(fill: rgb(21, 126, 184))[#text(fill: white)[#"0.91 (0.91)"]], table.cell(fill: rgb(14, 122, 182))[#text(fill: white)[#"0.94 (0.94)"]],
  [*#"of3"*], table.cell(fill: rgb(68, 151, 198))[#text(fill: white)[#"0.73 (0.62)"]], table.cell(fill: rgb(26, 128, 185))[#text(fill: white)[#"0.90 (0.90)"]],
)]

== The pool, clustered: the alternatives a slate hedges across

#align(center)[#image("figs/clusters.svg", height: 76%)]
#text(size: 11pt)[#"875 models, 9 clusters in 3 input variants · best predicted DockQ per cluster: af3_msa_pS548:1 0.72, af3_msa:0 0.71, mmseqs2_gpu:0 0.71"]

== What each cluster is: the model at the centre of it

#grid(columns: 4, gutter: 8pt,
  figure(image("figs/C0_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C0 · 133 models · conf 0.93 · DockQ 0.61 · bt2_jax"]]),
  figure(image("figs/C1_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C1 · 125 models · conf 0.96 · DockQ 0.66 · af3"]]),
  figure(image("figs/C8_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C8 · 123 models · conf 0.92 · DockQ 0.46 · af3/bt2_jax"]]),
  figure(image("figs/C7_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C7 · 122 models · conf 0.93 · DockQ 0.46 · bt2_jax/of3/af3"]]),
  figure(image("figs/C5_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C5 · 101 models · conf 0.92 · DockQ 0.44 · af3/bt2_jax"]]),
  figure(image("figs/C3_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C3 · 95 models · conf 0.73 · DockQ 0.43 · of3/af3"]]),
  figure(image("figs/C2_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C2 · 87 models · conf 0.93 · DockQ 0.45 · of3/bt2_jax/af3"]]),
  figure(image("figs/C6_chains.jpg", height: 26%), caption: [#text(size: 9pt)[#"C6 · 58 models · conf 0.77 · DockQ 0.41 · of3"]]),
)
#text(size: 11pt)[#"Each is the model with the highest membership in its cluster, superposed on the first and coloured by chain. Clusters are ordered by size; the numbers are the cluster's, not the model's."]

== Decisions and the priors they rest on

#text(size: 10pt)[#table(columns: (auto, auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 4pt,
  [*id*], [*type*], [*chose*], [*rests on*],
  [#"D1"], [#"variant"], [#"af3 input with SEP at chain A position 7 (LARP1 pS548), run next to the unmodified input"], [#"construct, literature, motif, chemistry"],
  [#"D2"], [#"stoichiometry"], [#"A1B1 submitted (given; one 14-3-3 gamma groove + LARP1 pS548 peptide). Run A1B1 and A2B2 (native 14-3-3 dimer, a p…"], [#"construct, public_structures, literature, stoichiometry_evidence"],
  [#"D3"], [#"variant"], [#"MSA: use the existing deep chain-B MSAs (af3_msa: uniref90 N=6536, uniprot N=16347; plus mmseqs2_gpu variant). No …"], [#"construct, msa_depth, existing_pool"],
  [#"D4"], [#"variant"], [#"Templates: AF3 default template search (91 of 1578 hits are 14-3-3 gamma, peptide-bound gamma included); no custom…"], [#"public_structures, templates, existing_pool"],
  [#"D5"], [#"variant"], [#"No ligand or metal: 14-3-3 binds the phosphopeptide without cofactors. Only chemistry variant is pS548 SEP (D1)"], [#"literature, chemistry"],
  [#"D6"], [#"sampling"], [#"No new GPU now: CASP17 pool already holds af3 25 seeds x5 for A1B1 and A2B2 x (af3_msa, af3_msa_pS548, mmseqs2_gpu…"], [#"binding_site, motif, stoichiometry_evidence, existing_pool"],
)]

== Checkpoint C1 (plan) at PLAN REVIEW

#text(size: 12pt)[
+ #"Phosphorylation: (a) A7 = pS548 (SEP), lead with SEP models, unmodified as control [recommended: pS-L-P motif, 14-3-3 needs a phosphate], (b) unmodified only, (c) equal weight"
+ #"Stoichiometry: (a) submit A1B1, run A2B2 as dimer-context hedge cut to one B:A pair [recommended], (b) A1B1 runs only, (c) submit the A2B2 dimer"
+ #"GPU: (a) only <=40 of 400 GPU-h (pt2 rerun, of3/bt2 with SEP) on top of the existing ~1.4k-model pool [recommended], (b) none, (c) +100 GPU-h af3 seeds with a custom 6A5S template"
+ #"Selection gate: (a) A7 phosphate within 4 A of B K49/R56/R131/Y132 first, then confidence [recommended], (b) confidence only"
]
#text(size: 11pt)[
- #text(fill: rgb("#2a2a2e"), weight: "bold")[agent]: #"Correction to my question 4. Since writing it I measured 6A5S (notes/target.md section 4): nearest side-chain atom of B K49/R56/R131/Y132 to the phosphoserine is 2.4-2.8 A from the phosphate oxygens, 3.6-3.9 A from the phosphorus, and 4.2-5.1 A from the serine OG. So a 4 A cut-off passes the reference only when measured from th…"
]

== Best model per stoichiometry: structures (by chain)

#grid(columns: 2, gutter: 6pt,
  figure(image("figs/A1B1_chains.jpg", height: 33%), caption: [#"A1B1: A1B1__af3_msa_pS548__af3__s1_0"]),
  figure(image("figs/A2B2_chains.jpg", height: 33%), caption: [#"A2B2: A2B2__af3_msa_pS548__af3__s1_0"]),
)

== Best model per stoichiometry: confidence (pLDDT)

#grid(columns: 2, gutter: 6pt,
  figure(image("figs/A1B1_plddt.jpg", height: 33%), caption: [#"A1B1"]),
  figure(image("figs/A2B2_plddt.jpg", height: 33%), caption: [#"A2B2"]),
)
#text(size: 11pt)[pLDDT: #text(fill: rgb("#0053D6"))[■] > 90 · #text(fill: rgb("#65CBF3"))[■] 70–90 · #text(fill: rgb("#FFDB13"))[■] 50–70 · #text(fill: rgb("#FF7D45"))[■] < 50]

#include "agent.typ"

== What the analysis read

#text(size: 11pt)[#table(columns: (auto, 1fr, auto), stroke: 0.4pt + luma(200), inset: 5pt,
  [*\#*], [*source*], [*supports*],
  [#"1"], [#"UniProt Q6PKG0 REST json, /store/deepfold/lantern/targets/H1388/dl/Q6PKG0.json"], [#"construct"],
  [#"2"], [#"UniProt P61981 REST json, /store/deepfold/lantern/targets/H1388/dl/P61981.json"], [#"construct"],
  [#"3"], [#"UniProt P61981; /gpfs/deepfold/casp/casp17/targets/H1388/B/hhpred.hhr"], [#"identity"],
  [#"4"], [#"UniProt Q6PKG0"], [#"identity"],
  [#"5"], [#"RCSB sequence search, chain B, identity>=95%, e<1e-10, 2026-11-23"], [#"public_structures"],
  [#"6"], [#"RCSB search API; entry titles from data.rcsb.org"], [#"public_structures"],
  [#"7"], [#"RCSB seqmotif GLSASLPD + full-text 'LARP1 14-3-3'"], [#"public_structures"],
  [#"8"], [#"RCSB search API 2026-11-23"], [#"public_structures"],
  [#"9"], [#"Yaffe et al. 1997 PMID 9428519; conserved in 14-3-3 gamma:phosphopeptide crystals 2B05, 6A5S"], [#"literature, binding_site, motif, chemis…"],
  [#"10"], [#"14-3-3 obligate homodimer via N-terminal helices (Liu 1995 PMID 7603574); one peptide per monomer groove (Yaffe 1997 PMID 9428519); the dimer interfa…"], [#"literature, stoichiometry_evidence"],
  [#"11"], [#"RCSB 9T76-9T7B titles, 'mTORC1 recruits substrates via an expanded TOS (XTOS) motif' (2026)"], [#"literature"],
  [#"12"], [#"RCSB assembly 1 + rcsb_struct_symmetry, 2026-11-23"], [#"stoichiometry_evidence"],
  [#"13"], [#"RCSB search 2026-11-23"], [#"templates"],
)]
#text(size: 11pt)[*No source recorded*: #"existing_pool"]

== What this proposes, and what it is asking

#text(size: 13pt)[
- #text(fill: rgb("#D55E00"), weight: "bold")[#"variant"] #"af3 input with SEP at chain A position 7 (LARP1 pS548), run next to the unmodified input; MSA: use the existing deep chain-B MSAs (af3_msa: uniref90 N=6536, uniprot N=16347; plus mmseqs2_gpu variant). No crawl/plmMSA. Peptide A runs single…; Templates: AF3 de…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"stoichiometry"] #"A1B1 submitted (given; one 14-3-3 gamma groove + LARP1 pS548 peptide). Run A1B1 and A2B2 (native 14-3-3 dimer, a peptide in each groove); A2B2 models…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"sampling"] #"No new GPU now: CASP17 pool already holds af3 25 seeds x5 for A1B1 and A2B2 x (af3_msa, af3_msa_pS548, mmseqs2_gpu) plus bt2_jax and of3 (unmodified)…"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"Best model in the pool so far"] #"by confidence `A1B1__af3_msa_pS548__af3__s9_0` (0.960) · by predicted DockQ `A1B1__af3_msa_pS548__af3__s11_4` (0.717) — not the same model, which is what the slate's two scales are for"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"C1 question 1"] #"Phosphorylation: (a) A7 = pS548 (SEP), lead with SEP models, unmodified as control [recommended: pS-L-P motif, 14-3-3 needs a phosphate], (b) unmodified only, (c) equal weight"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"C1 question 2"] #"Stoichiometry: (a) submit A1B1, run A2B2 as dimer-context hedge cut to one B:A pair [recommended], (b) A1B1 runs only, (c) submit the A2B2 dimer"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"C1 question 3"] #"GPU: (a) only <=40 of 400 GPU-h (pt2 rerun, of3/bt2 with SEP) on top of the existing ~1.4k-model pool [recommended], (b) none, (c) +100 GPU-h af3 seeds with a custom 6A5S template"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"C1 question 4"] #"Selection gate: (a) A7 phosphate within 4 A of B K49/R56/R131/Y132 first, then confidence [recommended], (b) confidence only"
- #text(fill: rgb("#D55E00"), weight: "bold")[#"C1 thread, the agent"] #"Correction to my question 4. Since writing it I measured 6A5S (notes/target.md section 4): nearest side-chain atom of B K49/R56/R131/Y132 to the phosphoserine is 2.4-2.8 A from the phosphate oxygens, 3.6-3.9 A from the phosphorus, and 4.2-5.1 A from the serin…"
]

== What the agent spent

#text(size: 13pt)[#table(columns: 4, stroke: 0.4pt + luma(200), inset: 5pt,
  [*stage*], [*steps*], [*turns*], [*USD*],
  [#"NEW"], [#"1"], [#"24"], [#"0.89"],
  [#"IDENTIFIED"], [#"1"], [#"11"], [#"0.29"],
  [#"HYPOTHESES"], [#"1"], [#"12"], [#"0.35"],
  [#"PLAN_REVIEW"], [#"1"], [#"24"], [#"0.63"],
  [*total*], [#"4"], [#"71"], [#"2.15"],
)]

