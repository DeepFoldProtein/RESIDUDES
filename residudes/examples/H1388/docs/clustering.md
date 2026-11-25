# Pool clustering and the target write-up

Computed after the four agent steps, using the standard commands and without generating new predictions.

## Clustering of the existing pool

The four agent steps stop at the plan checkpoint, but the CASP17 pool for this target already exists, so the
clustering page was computed afterwards, with the ordinary commands and no new predictions:

```sh
bin/residudes pool H1388                                                   # 1,750 models -> pool.csv
bin/residudes qa H1388 --sto A1B1 --what cluster --variants af3_msa,af3_msa_pS548,mmseqs2_gpu
bin/residudes qa H1388 --sto A1B1 --what interface                         # predicted DockQ per model
```

875 A1B1 models, nine clusters. The figure is the same UMAP four times — coloured by cluster, by predicted
DockQ, by the predictor's own confidence, and one small panel per predictor — then the clusters ranked by mean
predicted DockQ with what is in each, and four sentences the numbers support. The deck adds a page of **the
medoid of each cluster**, rendered from one view with the peptide as orange spheres, so the shape each cluster
claims is shown directly.

What it shows on this target:

- **cluster 1 holds all 125 `af3_msa_pS548` models and nothing else** — no unmodified model lands in it, no
  phosphoserine model lands anywhere else, and it has the highest confidence (0.960) and the highest mean
  predicted DockQ (0.66) of any cluster. The decision the agent made at HYPOTHESES separates the pool in two,
  which is what the person is deciding about at the checkpoint;
- the biggest cluster is not the best one: cluster 0 has 133 models and a mean predicted DockQ of 0.61;
- four clusters are one predictor's alone (bt2_jax 133, af3 125, of3 58, of3 31) — shapes no other predictor
  found, which is the case for hedging the slate across them rather than taking the top of one list.

Everything that page is drawn from ships with it — [`pool.csv`](../pool.csv) (1,750 models),
[`qa/cluster_A1B1_pool-befc310a_clustering.csv`](../qa/cluster_A1B1_pool-befc310a_clustering.csv) (the cluster and the
UMAP coordinates of each model) and [`qa/interface_A1B1/xgb_dockq.csv`](../qa/interface_A1B1/xgb_dockq.csv) (predicted
DockQ for all 875) — so the picture can be checked against its numbers, and `residudes report H1388` redraws it
from them. A test runs exactly that and compares the result with the `analysis.md` in this folder.

## The target write-up

[`notes/target.md`](../notes/target.md) is the document a person reads, written from the dossier in a separate
60-turn run once the analysis was done: identification, a subsection per chain, the biology that bears on the
structure, the sites and chemistry in input numbering with the native numbering beside them, what is uncertain, a
table of every structure that exists, the figures, and 20-odd numbered references. `residudes report` embeds it, so
[`analysis.md`](../analysis.md) opens with it.

One passage is of particular note. The plan recommends a gate of 4 Å between the A7 phosphate and
the B pocket, so the agent measured the distances in 6A5S itself: 2.4–2.8 Å from the phosphate oxygens, 3.6–3.9 Å
from the phosphorus, 4.2–5.1 Å from the serine OG. The unmodified runs have no phosphate, so for those the only
atom to measure from is OG — and measured from OG the reference structure itself fails a 4 Å gate. The note says
the spec has to name the atom or use about 5.5 Å. The agent therefore identified a defect in its own recommendation
before the checkpoint was answered.
