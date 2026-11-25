# Findings of the four agent steps

The run itself is described in [`../README.md`](../README.md); this document records its results.

## Results of the four steps

[`analysis.md`](../analysis.md) is the document; these are the parts worth looking at.

**It identified both chains from the sequence alone.** Chain B is 14-3-3 gamma (YWHAG, UniProt P61981, *Homo
sapiens*), Pfam PF00244, InterPro IPR000308 and three more, "all-alpha, 9 antiparallel helices; amphipathic groove
binds phospho-Ser/Thr peptides; obligate dimer via helices A–D". Target residues 2–232 are native 3–233, so
`input = native − 1`, and the acidic C-terminal tail 234–247 is not in the construct. Chain A is LARP1 542–552
(Q6PKG0).

**The chemistry the FASTA does not carry is the whole target.** The peptide has two serines, S5 and S7 (LARP1 S546
and S548). 14-3-3 binds only phosphorylated motifs, so one of them should be phosphoserine. The agent chose S7:
it sits in a pS-X-P motif (S7-L8-P9) and UniProt annotates S548 as a phosphosite, while S5 is S-A-S with no proline
at +2 and no annotation. That became `residudes decide --type variant`: SEP at chain A position 7, run beside the
unmodified input.

**MSA depth per chain, measured not assumed.** B is deep (uniprot 16,347, uniref90 6,536, mgnify 4,332, bfd 1,607;
no thin regions). A is an 11-residue peptide whose a3m files are query-only, so it runs single-sequence — which is
expected, not a failure, and the note says so.

**The structures that exist.** 23 exact-sequence human 14-3-3 gamma entries in the PDB, the best peptide-bound one
6A5S at 2.10 Å — rendered in [`report/figs/template_6a5s.jpg`](../report/figs/template_6a5s.jpg).

## Figures generated from the dossier

One per kind of evidence, drawn from the dossier by `residudes report` with no job to run, and each its own page in
[`report/report.pdf`](../report/report.pdf). They are separate on purpose: the species, the site and the motif are
different questions, and one picture of all three answers none of them.

| figure | what it says here |
|---|---|
| [`chain_map.svg`](../report/figs/chain_map.svg) | both chains to scale with the residues a gate would use: S7 on the peptide, K49/R56/R131/Y132 in the phosphate pocket |
| [`literature.svg`](../report/figs/literature.svg) | the five claims, their sources, and which priors and decisions share an identifier with each — plus the line that nothing read settles the 14-3-3:LARP1 complex itself |
| [`sites.svg`](../report/figs/sites.svg) | the pocket and the motif residue by residue in their own sequences, with the ✱ where the phosphoserine goes, and the native numbering quoted beside the input numbering |
| [`motif.svg`](../report/figs/motif.svg) | the arithmetic behind the target's whole decision: pS-X-P fits the peptide **only** at S7, S-A-S fits at S5, and the paper's full mode-I consensus (R-S-X-pS-X-P) fits nowhere — the peptide is a partial match |
| [`msa_coverage.svg`](../report/figs/msa_coverage.svg) | coverage per residue for every MSA |
| [`clusters.svg`](../report/figs/clusters.svg) | the existing pool four ways — clusters, predicted DockQ, confidence, and where each predictor sits (below) |
