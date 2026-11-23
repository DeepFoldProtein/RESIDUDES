# MassiveClustering — CASP17 wrapper

Click-click clustering entry point. One person sets up the venv once;
everybody else just runs the wrapper to submit Slurm clustering jobs.
Same shape as `inferences/<model>/submit_inference.sh` and
`visualization/analyze_target.sh`.

> **Quick orientation.** This doc is the long-form wrapper reference.
> If you want the short tour, start with [`README.md`](README.md) +
> [`docs/quickstart.md`](notes/quickstart.md). Newer subcommands
> (`iface-matrix`, `cluster-analysis`, `submission-picks`,
> `repair-terminal`, `canonicalize-chains`) and the **interface
> distance metrics** (`iface-jaccard-residue/-contact`, `iface-chamfer`,
> `iface-wasserstein`) are documented there + in
> [`docs/advanced.md`](notes/advanced.md). This file focuses on the
> Slurm wrapper (`run_clustering.sh`) and the original
> tm/itm/drmsd/anchored-\*/com/tcr-angle workflows.

---

## One-time setup

Creates `analysis/.venv/` (uv-managed) and installs all dependencies. The
venv lives on the shared `/gpfs/deepfold/casp/casp17/` mount, so it's a
one-time cost shared by everybody.

```bash
# base install (massivecluster + GPU + comprehensive plots)
bash analysis/setup.sh

# optional: also install ANARCI for CDR detection (separate .venv-cdr/)
bash analysis/setup.sh --with-cdr
```

If `uv` isn't on PATH the script downloads it into `analysis/.bin/uv`.
HMMER is symlinked from `/gpfs/deepfold/anaconda3/envs/AlphaFold/bin/`.

ANARCI lives in a sibling `analysis/.venv-cdr/` (Python 3.10 +
biopython 1.81 — known-good combo for HMMER 3.3.2 output) so the main
venv stays clean. The clustering driver shells out to it when `--cdr`
is requested.

---

## Running

### The three commands that cover most runs

```bash
# 0) Sanity check — confirm structures are discovered + share chain composition
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 \
    --method tm --list

# 1) The "do everything sensible" command — submits 3 Slurm jobs (tm + itm + drmsd)
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 \
    --method all

# 2) Anchored mode — adds 2 more Slurm jobs (anchored-rmsd + anchored-tm)
#    which give MUCH sharper signal than BASE TM for variable-region analysis.
#    Same anchor=AB-framework / probe=AB-cdr works for any antibody/TCR target.
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1 \
    --method all \
    --anchor-chains AB --anchor-mode framework \
    --probe-chains AB --probe-mode cdr
```

Each command line submits one or more Slurm jobs (`normal` partition,
1 GPU, 8 h walltime by default) and returns immediately. Monitor with
`squeue -u $USER`. Outputs land under
`targets/<T>/cluster_results/<sto>/<variant>/<method>[_extras]/`.

### Step-by-step recipe for a new target

1. **Discover.** Use `--list` to verify what's there:
   ```bash
   analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method tm --list
   ```
   Expect output like:
   ```
   target=H1311  base=complex  sto=A1B1  variant=af3_msa  method=tm
     pooled N=500 structures (af3:125, bt2:125, of3:125, pt2:125)
     shape ok: every structure has chains A=204, B=236
   ```
   If you see `! sequence shape MISMATCH`, the inferences for this sto
   are not consistent — re-run inference for the offending models
   before clustering, or use `--allow-mismatch` to keep just the
   dominant shape.

2. **Submit the standard 3.** `--method all` expands to one Slurm job
   per metric (auto-skips `itm` for monomers):
   ```bash
   analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all
   ```

3. **Optional: add CDR-excluded variants.** Useful when you want to
   confirm whether CDR variability is what's driving cluster
   structure:
   ```bash
   analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all --cdr
   ```

4. **For antibody / TCR / pMHC targets: also run anchored modes.**
   These give an order of magnitude wider dynamic range than BASE TM
   on multimers (see § "Anchor / probe analysis" below):
   ```bash
   # framework-aligned CDR loop study (works on any Ig/TCR sub-stoichiometry):
   analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all \
       --anchor-chains <Ig_chains> --anchor-mode framework \
       --probe-chains <Ig_chains> --probe-mode cdr

   # peptide-MHC pose study (when stoichiometry contains a peptide chain):
   analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all \
       --anchor-chains <MHC_chains> --probe-chains <peptide_chain>
   ```

5. **Wait + inspect.** The wrapper prints a `squeue` watch command
   for each submission. Once jobs complete, the most useful files
   per run are:
   ```
   <output_dir>/results/<tag>_cluster_summary.png    ← top-level overview
   <output_dir>/results/<tag>_native_scores.csv      ← per-cluster confidence (mean/min/max)
   <output_dir>/results/<tag>_provenance.csv         ← which model dominates each cluster
   <output_dir>/results/<tag>_*_comprehensive.png    ← t-SNE / UMAP visualisation
   <output_dir>/results/<tag>_*_k_diagnostic.png     ← K-selection sanity (see § How K is chosen)
   <output_dir>/results/<tag>_representatives/       ← medoid PDB per cluster (downstream input)
   ```

### Common invocations

```bash
# single metric
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 --method tm

# multimer + interface-TM (more sensitive to chain-chain interfaces)
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 --method itm

# drop IMGT CDR1/2/3 residues per chain (ANARCI). Chains where ANARCI
# finds no Ig/TCR domain are kept intact.
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 --method tm --cdr

# all metrics in one go — auto-expands to tm+itm+drmsd for multimers,
# tm+drmsd for monomers (each becomes its own Slurm job in parallel)
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 --method all
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 --method all --cdr

# dry-run discovery only (no Slurm, no GPU work)
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1 --method tm --list
```

### Full flag list

```bash
analysis/MassiveClustering_tool/run_clustering.sh --help
```

| Flag | Meaning |
|---|---|
| `--sto SPEC` | required. `A1B1`, `A1B1C1D1E1`, `C1D1`, …, follows the same convention as `inferences/`. |
| `--method tm\|itm\|drmsd\|anchored-rmsd\|anchored-tm\|com\|tcr-angle\|all` | required. `tm/itm/drmsd` align and measure on the same residues. `anchored-rmsd / anchored-tm` Kabsch-align on `--anchor-chains` then measure RMSD or TM-score on `--probe-chains` (decoupled alignment frame). `com` = pairwise L2 of probe-chain COMs in anchor frame. `tcr-angle` = Pierce crossing/incident angle. `all` auto-expands; `itm` is skipped for monomers; `anchored-*` is added when both `--anchor-chains` and `--probe-chains` are supplied. **Interface metrics** (`iface-jaccard-residue/-contact`, `iface-chamfer`, `iface-wasserstein`) are built via the `iface-matrix` subcommand and fed back as a `.npy`; see [docs/advanced.md §1](notes/advanced.md). |
| `--anchor-chains SPEC` | Chain set for Kabsch alignment in anchored modes (e.g. `CD`). Use this **or** `--anchor-spec`, not both. |
| `--probe-chains SPEC` | Chain set whose RMSD or TM-score is measured after alignment (e.g. `E`). Use this **or** `--probe-spec`. |
| `--anchor-mode all\|framework\|cdr\|cdr1\|cdr2\|cdr3` | Residue subset within `--anchor-chains`. `all` (default), `framework` (chain − IMGT CDR), `cdr` (all IMGT CDR), `cdr1`/`cdr2`/`cdr3` (individual loops). Non-`all` requires `bash analysis/setup.sh --with-cdr`. |
| `--probe-mode ...` | Same options, applied to `--probe-chains`. |
| `--anchor-spec SPEC` | Mixed-mode replacement for `--anchor-chains`/`--anchor-mode`. Format: `'chains[:mode](+chains[:mode])*'`. Examples: `'CD'`, `'AB:cdr+E'`, `'A:cdr1+B:cdr3+E'`. Lets a single side mix multiple chain sets and modes (e.g. CDR of antibody chains plus full peptide chain). |
| `--probe-spec SPEC` | Same syntax, applied to probe. Mutually exclusive with `--probe-chains`. |
| `--variant NAME` | Single MSA / template variant (default `af3_msa`). |
| `--variants LIST` | Pool predictions from **multiple** variants. Comma-list (`af3_msa,mmseqs2_gpu`) or `all` (every subdir under `<sto>/`). Each prediction's name in the matrix gets a `__<variant>` suffix so provenance can attribute it. Mutually exclusive with `--variant`. |
| `--cdr` | exclude IMGT CDR1/2/3 residues per chain. Non-Ig/TCR chains are unaffected. |
| `--variant NAME` | MSA variant subdir. Default `af3_msa`. |
| `--cluster-method NAME` | `fuzzy` (default), `kmeans`, `hierarchical`, `leiden`. |
| `--allow-mismatch` | accept inputs with inconsistent chain composition; keep only the dominant shape. Common case: AF3 emits monomer chains as `A` regardless of stoichiometry letter, while bt2/of3/pt2 keep the original label. |
| `--skip-existing` | reuse cached distance matrix on re-run. |
| `--no-medoid-pdbs` | skip CIF→PDB export used for consensus / medoid output (faster). |
| `--keep-pdb-view` | retain the intermediate `_pdb_view/` directory. Default: deleted at end of run after `representatives/` is written, since this dir can be many GB on large pools. Pass this flag if you need it for downstream tools (e.g. medoid_faces re-runs). |
| `--local` | run in-place instead of submitting to Slurm (debug). |
| `--partition NAME` | Slurm partition. Default `normal`. |
| `--time HH:MM:SS` | Slurm walltime. Default `8:00:00`. |
| `--list` | print discovered structures and exit. No Slurm. |
| `--output-base PATH` | redirect output root (default `<target>/cluster_results/`). Used to write to user-scoped scratch (e.g. `playground/<user>/<target>_clustering/`). |
| `--target-name NAME` | override the target label in output filenames (default: `target_dir.name`). |
| `--gpu N` | GPU device id for the matrix computation (default: 0). |
| `--flat-dir PATH` | cluster a flat directory of structures (e.g. MassiveFold's `<T>_all_pdbs/`), bypassing the in-house `<sto>/<variant>/<runner>/seed_*/sample_*/` tree. Files must be named `Model_<idx>_<engine>_..._pred_<i>.{pdb,cif}`. Mutually exclusive with `--variant`/`--variants`. |
| `--flat-ranking PATH` | ranking CSV with `mapped_name` column (e.g. MassiveFold `ranking_<TARGET>.csv`). Auto-detected as `<flat-dir>/ranking_<TARGET>.csv` if exactly one match exists. |
| `--flat-tag NAME` | tag used as the `<variant>` segment in the output path (e.g. `massivefold` → `cluster_results/<sto>/massivefold/<method>/`). Default: basename of `--flat-dir`. |
| `--tcr-angle-csv PATH` | precomputed Pierce angle CSV (used with `--method tcr-angle`). |
| `--tcr-angle-recompute` | force re-running Pierce even if `--tcr-angle-csv` exists. |
| `--tcr-angle-group-filter NAME` | restrict CSV rows to a single `group_dir` (e.g. `af3_msa`). |
| `--tcr-angle-chain-layout NAME` | chain-layout passed to `tcr3d_pierce.compute_angles` (default `ours_5chain`). |
| `--tcr-angle-target NAME` | optional HLA-offset key (e.g. `9RU5`) for the Pierce computation. |
| `--tcr-angle-workers N` | parallel workers for Pierce auto-compute (default 16). |

---

## Output layout

Original CIFs in `<target>/results/.../sample_*/model.cif` are read-only
— every artefact lands under `<target>/cluster_results/`. Each
metric (and each anchor/probe combo) gets its own sibling directory,
so multiple analyses on the same `(sto, variant)` coexist without
overwriting:

```
<target>/cluster_results/<sto>/<variant>/
├── tm                        # BASE: TM
├── itm                       # BASE: interface TM (multimer only)
├── drmsd                     # BASE: distance-vector RMSD
├── tm_cdr                    # --cdr variants (CDR residues excluded from input)
├── itm_cdr
├── drmsd_cdr
├── anchored-rmsd_anchor-CD_probe-E             # anchored: pMHC peptide pose
├── anchored-tm_anchor-CD_probe-E
├── anchored-rmsd_anchor-CD_probe-AB            # anchored: TCR docking on MHC
├── anchored-tm_anchor-CD_probe-AB
├── anchored-rmsd_anchor-AB-framework_probe-AB-cdr   # anchored: CDR loops on TCR framework
└── anchored-tm_anchor-AB-framework_probe-AB-cdr
```

Each leaf directory has the same internal structure:

```
<output_dir>/
├── matrices/
│   ├── <method>_matrix.npy       # (N, N) float32 distance matrix
│   ├── structure_names.npy       # row/col → filename mapping
│   ├── cdr_residues.json         # only with --cdr: per-chain excluded residue indices
│   └── anchor_probe.json         # only with anchored-*: chains, modes, residue counts, CDR set
├── ranking.csv                    # per-prediction confidence collected from summary.json
├── _pdb_view/                     # CIF→PDB cache for consensus / medoid extraction (auto-removed at end of run; pass --keep-pdb-view to retain)
└── results/                       # massivecluster comprehensive output
    ├── *_clustering.csv           # structure → cluster + soft membership probabilities
    ├── *_cluster_summary.csv      # per-cluster size, intra-distance, medoid filename
    ├── *_native_scores.csv        # per-cluster confidence stats:
    │                              #   {ranking_score, iptm, ptm, plddt, gpde,
    │                              #    fraction_disordered}_{mean, min, max}
    │                              #   plus medoid_{...} for each
    ├── *_provenance.csv           # per-cluster model occupancy (af3 / bt2 / of3 / pt2)
    ├── *_method_summary.csv       # global model statistics across all clusters
    ├── *_k_scores.csv             # full per-K scoring table (see § How K is chosen)
    ├── *_consensus/               # consensus CIF per cluster (RMSF in B-factor)
    │   ├── cluster0_consensus.cif
    │   ├── cluster0_rmsf.csv
    │   └── ...
    ├── *_representatives/         # medoid PDB per cluster (downstream-ready)
    │   ├── cluster0_medoid.pdb
    │   └── ...
    ├── *_confidence_dist.png      # 6-panel violin: per-cluster ranking_score / iPTM /
    │                              #   pTM / pLDDT / gPDE / fraction_disordered distributions
    ├── *_variant_cluster.png      # multi-variant runs only: variant × cluster heatmap
    ├── *_variant_cluster_counts.csv   # multi-variant: cross-tab counts
    ├── *_variant_cluster_frac.csv     # multi-variant: column-normalised fractions
    └── *.png                      # diagnostic figures:
                                   #   *_comprehensive.png (t-SNE + UMAP + score colormap),
                                   #   *_k_diagnostic.png (K-selection),
                                   #   *_cluster_summary.png (table-as-figure),
                                   #   *_native_scores.png, *_provenance.png

<target>/cluster_results/_slurm_logs/
└── cl-<target>-<sto>-<method>[-extras].<jobid>.{out,err}
```

**The four files you'll open most often per run**:

1. `<tag>_cluster_summary.csv` — sizes, intra-distance, medoid filename per cluster
2. `<tag>_native_scores.csv` — per-cluster confidence (sort by `ranking_score_mean`)
3. `<tag>_provenance.csv` — which model dominates each cluster
4. `<tag>_*_comprehensive.png` — t-SNE / UMAP scatter coloured by cluster

The medoid PDB of each cluster is the prediction closest to its
cluster's centre. Large clusters with high mean confidence are the
strongest QA candidates — pick them up from `<tag>_representatives/`.

---

## Common workflows

### 1) New target, first pass

```bash
# verify discovered structures share the same chain composition.
# Mismatches are flagged with a per-model breakdown so you know which
# inferences need to be re-run.
analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method tm --list

# proceed if clean
analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all
```

### 2) Multimer including antibody / TCR chains

```bash
# baseline
analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all

# CDR-excluded — if top cluster shifts versus baseline, CDR variability
# is dominating the conformational signal in that target.
analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto <SPEC> --method all --cdr
```

### 2a) Anchor / probe analysis (decoupled alignment frame)

**When to use** — BASE TM saturates (everything ~0.99) on large complexes
because the rigid scaffold dominates the alignment. Interesting variation
in a small region (a peptide in the MHC groove, CDR loops on a TCR
framework) gets diluted out. Anchored modes Kabsch-align on a *reference*
chain set and measure on a *different* chain set, isolating the variable
region.

Empirical comparison on H1311 (TCR-pMHC):

| sto | metric | matrix range | dynamic range |
|---|---|---|---|
| A1B1 (TCR α+β) | BASE TM | 0.925 .. 1.000 | 0.075 (saturated) |
| A1B1 | **anchored-TM (framework→CDR)** | **0.311 .. 1.000** | **0.69 (9× wider)** |
| C1D1E1 (MHC+pep) | BASE TM | 0.954 .. 1.000 | 0.046 (super-saturated) |
| C1D1E1 | **anchored-TM (AB→C peptide)** | **0.187 .. 0.999** | **0.81 (17× wider)** |

The wider dynamic range translates directly into more meaningful clusters
— the top-confidence cluster jumps from "everyone agrees this is right"
in BASE to "*this specific region* is consistent across these N
predictions" in anchored.

**When you want** to fix one part of the complex as a reference frame and
study how a different part is positioned in that frame:

```bash
# pMHC peptide pose: align on MHC + β2m, measure peptide
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto C1D1E1 \
    --method anchored-rmsd --anchor-chains AB --probe-chains C

# 1 - TM (probe length d0) as a distance instead of Å
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto C1D1E1 \
    --method anchored-tm --anchor-chains AB --probe-chains C

# TCR-pMHC docking: align on MHC, measure TCR position
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1C1D1E1 \
    --method anchored-rmsd --anchor-chains CD --probe-chains AB

# antibody/TCR CDR-only clustering — align on framework, measure CDR loops
# (anchor and probe share the same chains but disjoint residue subsets)
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto A1B1 \
    --method anchored-rmsd \
    --anchor-chains AB --anchor-mode framework \
    --probe-chains AB --probe-mode cdr

# all-in-one: tm + itm + drmsd + anchored-rmsd + anchored-tm
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 --sto C1D1E1 \
    --method all --anchor-chains AB --probe-chains C
```

The metric output goes to a separate output dir per anchor/probe pair —
multiple anchor/probe selections coexist for the same `(sto, variant)`:

```
<target>/cluster_results/<sto>/<variant>/anchored-rmsd_anchor-AB_probe-C/
<target>/cluster_results/<sto>/<variant>/anchored-tm_anchor-AB_probe-C/
```

**Choosing the metric**:

| metric | output | best for | caveat |
|---|---|---|---|
| `anchored-rmsd` | distance, Å | absolute spatial difference; intuitive | not normalised — can't compare across targets with different probe sizes |
| `anchored-tm` | similarity, 0..1 (use `1 − x` as distance) | length-normalised comparison | very short probes (≲ 16 aa) hit d0 = 0.5 floor, becoming hyper-sensitive → over-clustering |

Run both — they correlate ~0.7 (Pearson) on the same data, so cluster
consistency between the two is itself a robustness signal. K is usually
much smaller for `anchored-rmsd` (better for picking dominant modes)
while `anchored-tm` discriminates fine micro-states.

**Cross-correlation with existing metrics** (H1311 A1B1 framework→CDR
anchored-rmsd, distance space, 500 structures):

| anchored-rmsd vs | Pearson | interpretation |
|---|---|---|
| BASE iTM | +0.81 | iTM-interface and CDR overlap heavily for TCRs |
| BASE TM | +0.77 | overall similarity captures some CDR variation |
| `--cdr` TM (framework only) | +0.58 | weak — framework alone doesn't see CDR |
| `--cdr` iTM | +0.41 | weakest — interface minus CDR |

→ Anchored framework→CDR is a genuinely different signal from BASE/iTM/`--cdr`.
The right choice depends on the question.

### 2b) Mixed-region probe (`--probe-spec`)

`--probe-mode` applies one mode uniformly to all probe chains. Sometimes
you want to combine **different modes for different chains** in the same
probe — e.g. "the CDR loops of TCR chains A,B PLUS the full peptide
chain E". The `--probe-spec` (and `--anchor-spec`) syntax lets you list
multiple groups in one string:

```
spec  ::= group ('+' group)*
group ::= chain_chars [':' mode]
mode  ::= all | framework | cdr | cdr1 | cdr2 | cdr3   (default 'all')
```

Examples on H1311 A1B1C1D1E1 (A=TCRα, B=TCRβ, C=MHC, D=β2m, E=peptide):

```bash
# pMHC-anchored TCR binding interface — CDR loops + peptide together:
--anchor-spec CD  --probe-spec 'AB:cdr+E'

# Same but only CDR3 of TCR (the most variable loop) + peptide:
--anchor-spec CD  --probe-spec 'AB:cdr3+E'

# Per-chain CDR mixing — A's CDR1 + B's CDR3 + peptide:
--anchor-spec CD  --probe-spec 'A:cdr1+B:cdr3+E'
```

Path / job names auto-encode the spec (`+` → `_` in the dir name, e.g.
`anchored-rmsd_anchor-CD_probe-AB-cdr_E/`).

When the same chain appears in multiple groups (e.g. `'A:cdr1+A:cdr3'`),
the residue masks are unioned. Disjoint anchor and probe is enforced at
the residue level — overlap → argparse error.

### 2c) Multi-variant pool (`--variants`)

Each `<target>/results/<sto>/<variant>/` directory is one MSA / template
build. By default `--variant af3_msa` clusters predictions from one
build. With `--variants` you can pool multiple builds into a single
matrix:

```bash
# specific subset:
analysis/MassiveClustering_tool/run_clustering.sh targets/H1311 \
    --sto A1B1C1D1E1 --method anchored-rmsd \
    --anchor-spec CD --probe-spec 'AB:cdr+E' \
    --variants 'af3_msa,mmseqs2_gpu,tcrm_msa'

# every variant directory under <sto>/:
... --variants all
```

Each prediction in the pooled matrix is named
`<model>_seed<S>_sample<I>__<variant>.pdb` so the `__<variant>` suffix
is recoverable. The variant suffix is placed at the **end** so the
provenance regex (`^af3_`/`^bt2_`/…) still attributes the model
correctly at the start of the name.

Two extra artefacts for multi-variant runs:

- **`<tag>_variant_cluster_counts.csv`** + `_variant_cluster_frac.csv`:
  cross-tab of variant rows × cluster columns (counts and
  column-normalised fractions).
- **`<tag>_variant_cluster.png`**: 2-panel heatmap of the same.
  Tells you at a glance which variants contribute to which clusters
  (e.g. "cluster 5 is 92% from `tcrm_msa`; cluster 0 is split evenly
  across 4 template variants").

The output goes to a directory whose path encodes the variant set, so
multiple variant subsets can coexist:
`<target>/cluster_results/<sto>/<v1>_AND_<v2>_AND_…/<method>[_extras]/`.

`ranking.csv` gains a `variant` column for downstream slicing.

> **When to use multi-variant pooling**: study how MSA / template
> choice influences the conformational landscape — e.g. "do
> custom-template builds find a different binding pose than the
> default af3_msa build?". **When to keep per-variant** (default):
> each build is a self-contained baseline you want to compare
> independently; pooling can swamp a smaller variant's signal under
> the larger one.

### 2d) MassiveFold flat-directory mode (`--flat-dir`)

When the structures don't live in the in-house
`<target>/results/<sto>/<variant>/<runner>/seed_*/sample_*/` tree —
e.g. you ran MassiveFold and got a single flat directory of
`Model_*_pred_*.{pdb,cif}` — use `--flat-dir` to bypass discovery and
cluster the directory directly:

```bash
analysis/MassiveClustering_tool/run_clustering.sh /unused/target_path \
    --sto A1B1   --method tm \
    --flat-dir   /path/to/T1234_all_pdbs \
    --flat-tag   massivefold \
    --output-base /path/to/playground/T1234_clustering
```

Filename convention required: `Model_<idx>_<engine>_..._pred_<i>.{pdb,cif}`.
The `engine` slot becomes the provenance label (af3/bt2/of3/pt2 attribution).

- **`--flat-ranking PATH`** — point at MassiveFold's `ranking_<TARGET>.csv`
  (it has the `mapped_name` column the wrapper expects). Auto-detected as
  `<flat-dir>/ranking_<TARGET>.csv` if exactly one match exists in the dir.
- **`--flat-tag`** — replaces the `<variant>` segment in the output path.
  Default is the basename of `--flat-dir`. Pick something short and
  human-readable (`massivefold`, `mf_v1`) — it lands in
  `cluster_results/<sto>/<flat-tag>/<method>/`.

`--flat-dir` is mutually exclusive with `--variant`/`--variants`; the
in-house variant tree is skipped entirely. Everything else (matrix,
clustering, network, representatives, consensus) works the same way.

### 3) Monomers (single chain)

`--method all` auto-skips `itm` for monomers (chain count ≤ 1). AF3
monomer outputs label the chain as `A`; other runners keep the
stoichiometry letter (`B`, `C`, …) — same protein, different label.
Use `--allow-mismatch` so the wrapper drops the AF3 set instead of
aborting:

```bash
analysis/MassiveClustering_tool/run_clustering.sh targets/<T> --sto A1 --method all --allow-mismatch
```

(Long-term fix is to change AF3 inference to emit the original chain id
even for single-chain runs.)

### 4) Job monitoring

```bash
squeue -u $USER -o '%.8i %.32j %.2t %.6M %R'
sacct -X --starttime now-1day --format=JobID,JobName,State,Elapsed
ls <target>/cluster_results/<sto>/<variant>/<method>/results/
```

---

## How K (number of clusters) is chosen

The wrapper does *not* pick K manually — `massivecluster comprehensive`
runs an adaptive search and combines four independent signals into a
composite score. This section explains every term, the artefacts that
record the decision, and how to override / debug.

### Why a composite score (vs. plain silhouette / elbow)

A single criterion is brittle in this domain:

- **Silhouette alone** tends to flatten on saturated similarity
  matrices (BASE TM on multimers where every pair > 0.99) — it picks
  small K because nothing looks separated, and we miss the real
  multi-mode structure.
- **Elbow on WCSS / BIC alone** tends to pick K too high on
  high-dimensional CA-coord data — the "elbow" gets blurry and the
  curve keeps decreasing.
- **t-SNE / UMAP-derived K** (e.g. running k-means in 2D) is
  notoriously unstable: the embedding creates artificial micro-clusters
  and silhouette monotonically rises, so K runs to the upper bound.

The wrapper instead runs the K probe on the **raw distance matrix
with hierarchical clustering** (deterministic, no embedding), records
several mutually-disagreeing signals, and combines them so the failure
modes cancel.

### The search loop

For every `k ∈ [k_min, k_max]` (defaults: `k_min=3`,
`k_max=ceil(sqrt(n))`):

1. Run **agglomerative hierarchical clustering** with **average
   linkage** on the precomputed distance matrix. (This is just a K
   probe — the *final* labels are produced separately by whatever
   `--cluster-method` is set, default fuzzy c-means.)
2. Record per-K:
   | quantity | formula | meaning |
   |---|---|---|
   | `silhouette` (sil) | `sklearn.silhouette_score(D, labels)` | cluster compactness vs. separation, in `[-1, 1]` |
   | `bic` | `n·log(WCSS/n) + k·log(n)`<br>`WCSS = Σ_c Σ_{i<j∈c} dᵢⱼ² / (2 nc)` | model-complexity penalty. WCSS is approximated from pairwise distances since we don't have a centroid (no embedding) |
   | `balance` | `(1 − max_fraction)²` | how evenly clusters are sized |
   | `max_frac` | `max(cluster sizes) / n` | what fraction of structures the dominant cluster swallowed |

**Early stopping**: if silhouette declines for **30 consecutive k** in
a row *after* at least 20 values were seen, the search breaks. With
`n=6715` the naive search would go to `k_max ≈ 82` (sqrt) or `n−1`;
early stopping caps wall-time to whatever's actually informative for
that distance matrix.

### The composite score

All raw arrays are min-max normalised to `[0, 1]` so the weights mean
what they say. Then:

```
                     ┌─ PEAK-AWARE PENALISED SILHOUETTE ────────────────┐
pen_score[k]      = sil[k] − 0.5 · (cumulative_drop_from_peak[k] / sil_range)

                     ↑ Up to the silhouette's peak, pen_score == sil.
                       After the peak, every drop accumulates a debt;
                       larger drops make later K's pay more. This stops
                       the score from quietly climbing into K-region
                       where silhouette is *technically* still high but
                       only because we're past the meaningful elbow.

                     ┌─ SILHOUETTE TRIPLET (65 %) ──────────────────────┐
sil_component[k]  = 0.35 · pen_n[k]
                  + 0.20 · sil_n[k]                  ← raw silhouette
                  + 0.10 · delta_sil_n[k]            ← Δ silhouette vs k-1
                                                       (rewards K where
                                                        silhouette JUMPS,
                                                        not just stays high)

                     ┌─ BALANCE GATE ───────────────────────────────────┐
balance_gate[k]   = 1 / (1 + exp(10 · (max_frac − 0.6)))

                     ↑ Multiplicative sigmoid. Roughly:
                        max_frac ≤ 0.40  →  gate ≈ 1.00 (don't penalise)
                        max_frac  = 0.60 →  gate = 0.50 (heavy penalty)
                        max_frac ≥ 0.80  →  gate ≈ 0.00 (kill this K)
                       Suppresses degenerate "1 huge cluster + rest noise"
                       solutions that would otherwise score well on
                       silhouette alone.

                     ┌─ FINAL ──────────────────────────────────────────┐
combined[k]       = sil_component[k] · balance_gate[k]   ← 65 % gated
                  + 0.25 · bic_n[k]                       ← 25 % model selection
                  + 0.10 · balance_n[k]                   ← 10 % size sanity
```

**Chosen K = argmax(combined)**.

### Why the three signals deliberately disagree

By construction the score is robust to any single signal failing:

| signal | failure mode | what catches it |
|---|---|---|
| silhouette saturated (BASE TM on multimers) | always low / flat → uninformative | BIC term picks up real multi-mode structure |
| BIC keeps decreasing (high-dim noise) | wants K → ∞ | silhouette + balance pull it back |
| silhouette peaks at trivially imbalanced K (e.g. 1 huge + rest tiny) | high silhouette but degenerate | balance_gate zeroes it → BIC alone has to carry, which it usually doesn't for low K |

When all three terms agree on the same K → very confident pick. When
they disagree → read `<tag>_k_diagnostic.png` (it has separate panels
for each term) and override if needed.

### What gets written to disk per K-selection run

Every `comprehensive` run with auto-K writes two artefacts beside the
clustering output:

**`<tag>_<metric>_k_scores.csv`** — full per-K table:
```
# Target: H1311_C1D1E1_tm
# Metric: tm
# Selected K: 23
# BIC optimal K: 32
# Plateau K: 17
# Ascending at boundary: False
# k_max requested: 28
# k_max effective: 28
k,silhouette,penalized_sil,delta_sil,bic,balance,combined_score
3,0.421,0.421,0.000,-1234.5,0.21,0.18
4,0.467,0.467,0.046,-1456.7,0.34,0.33
…
```
Header lines tell you:
- `Selected K`: what the chosen K was
- `BIC optimal K`: K that minimises BIC (often higher than selected)
- `Plateau K`: first K where silhouette has plateaued (lower bound suggestion)
- `Ascending at boundary`: true if silhouette is still climbing at
  `k_max` → bump `--k-max`
- `k_max effective` < `k_max requested` if early stopping fired

**`<tag>_<metric>_k_diagnostic.png`** — 4-panel figure:
1. **Silhouette vs K** — should show a clear peak; flat means saturated input
2. **BIC vs K** — should show a clear minimum; if monotone-decreasing, BIC alone would over-cluster
3. **Balance + max_frac vs K** — sanity check that the chosen K isn't dominated by one cluster
4. **Combined score vs K** — the actual decision curve, with the chosen K marked

If panel 4's peak isn't sharp (broad plateau at the top), the
selection is ambiguous — try a smaller fixed `--k`.

### Adaptive plateau detection

Beyond the early-stop, the search post-processes the silhouette curve
to find a "plateau K" — the first K where silhouette has effectively
levelled off:

```python
for window of size `plateau_window` (default 3) sliding over sil_arr:
    if max(window) − min(window) < `plateau_threshold` (default 0.01):
        plateau_k = first K in window
        break
```

This is reported as an alternative (often-tighter) K. The
`is_ascending_at_boundary` flag is set if the whole `[k_max−window,
k_max]` window is still climbing — that's the "you should have
searched further" warning.

`--no-adaptive-k` turns this whole post-processing off (you still get
the chosen K, but no plateau / ascending warnings).

### Override knobs (most → least invasive)

```bash
# Force a specific K outright (skips selection entirely):
--k 5

# Constrain the search range:
--k-min 3   # default 3
--k-max 30  # default ceil(sqrt(n)); raise if "Ascending at boundary"

# Disable plateau detection (still does early-stopping):
--no-adaptive-k

# Tune the plateau detector:
--plateau-window 5      # default 3 — wider = stricter "plateau"
--plateau-threshold 0.005  # default 0.01 — smaller = stricter

# Switch the algorithm used for the FINAL cluster labels.
# K-selection always uses hierarchical (deterministic, no init issues);
# this flag only changes how labels are produced once K is chosen:
--cluster-method fuzzy         # default — soft membership probabilities
--cluster-method leiden        # graph-based, ignores --k entirely
--cluster-method kmeans        # standard hard
--cluster-method hierarchical  # same algo as K-selection (cheapest)
```

**`leiden` is special** — it has its own K via modularity, ignores
`--k` / `--k-min` / `--k-max`. Use it when:
- silhouette landscape is degenerate (saturated matrices),
- the data has clear graph-community structure (KNN connections form
  obvious blocks), or
- you want a sanity check on the hierarchical K.

### Reading K-selection in our actual H1311 runs

| run | matrix dynamic range | chosen K | BIC opt K | plateau K | reading |
|---|---|---|---|---|---|
| A1B1 BASE TM | 0.075 (saturated) | 19 | 26 | 8 | silhouette flat → BIC dragging K up; 8 (plateau) is the more honest pick |
| A1B1 anchored-rmsd (fw→CDR) | 1.32 Å | **10** | 13 | 7 | three signals agreeing within ±3 → solid |
| A1B1 anchored-tm (fw→CDR) | 0.69 (d0≈11) | 31 | 31+ | — | TM-term sharp on 49-residue probe; over-clusters. Use RMSD twin's K=10. |
| C1D1E1 BASE TM | 0.046 (super-saturated) | 23 | 32 | 17 | flat sil, BIC picks high K; trust plateau K=17 as alternative |
| C1D1E1 anchored-rmsd (AB→C) | 3.2 Å | **4** | 6 | — | clean K=4 dominant peptide modes |
| C1D1E1 anchored-tm (AB→C) | 0.81 (d0=0.5 floor) | 16 | 18 | — | peptide is 10aa → d0 floor → micro-clusters |
| A1B1C1D1E1 BASE TM | 0.05 | 19 | — | — | saturated multimer pattern again |

**Patterns to internalise**:

- **Saturated matrices** (small dynamic range) → silhouette can't
  differentiate, **BIC drives the chosen K up artificially**.
  Cross-check with the plateau K (much lower) and consider using
  Leiden or anchored-RMSD instead.
- **Anchored-RMSD with broad Å range** (clear distance signal) →
  K = 4-10, all three signals agree, biologically interpretable.
  **This is what to trust.**
- **Anchored-TM with short probe** (d0 hits the 0.5 Å floor) →
  hyper-sensitive, K = 16-31. Treat as fine-grained micro-states
  inside the macro-modes that the RMSD twin found.

> **Recommendation in practice**: when you have RMSD and TM twins of
> the same anchor/probe selection, **trust the RMSD twin's K for
> picking dominant modes**. Use the TM twin only when you want to
> resolve fine-grained structure *inside* an RMSD cluster.

### Debug recipe when K looks wrong

1. **Open `<tag>_<metric>_k_diagnostic.png`** — does the combined
   score peak sharply, or is the top a broad plateau? Broad plateau
   ⇒ ambiguous, consider forcing a smaller `--k`.
2. **Check `<tag>_<metric>_k_scores.csv` header**:
   - `Selected K` >> `Plateau K` ⇒ BIC dragged it up. Try
     `--k <plateau_K>`.
   - `Ascending at boundary: True` ⇒ search stopped too early. Bump
     `--k-max`.
   - `k_max effective` << `k_max requested` ⇒ early stopping fired.
     Probably fine; just means silhouette stopped improving long
     before the requested upper bound.
3. **Check the `max_frac` column** — if the chosen K has
   `max_frac > 0.6` the balance gate suppressed it
   (`combined_score` near 0). That shouldn't normally happen because
   the gate kills those, but worth confirming.
4. **Try `--cluster-method leiden`** — same input, totally different
   K. If Leiden's K is similar to plateau-K but very different from
   the chosen K, there's something weird about the silhouette
   landscape and the plateau is more trustworthy.

---

## Picking QA candidates

For each `(sto, variant, method)` run:

1. Open `<tag>_native_scores.csv`, sort by `ranking_score_mean` descending.
2. Top cluster qualifies if its `size` is ≥ ~5 % of the total
   prediction pool **and** `ranking_score_min` is also reasonable
   (e.g. > 0.5) — i.e. the whole cluster is high-confidence, not just
   the medoid.
3. Check `<tag>_provenance.csv` `dominant_method` — a single dominant
   model means that runner's preferred conformational mode.
4. Pick `<tag>_representatives/cluster<i>_medoid.pdb` for downstream
   work (template, refine, etc.).

If the same cluster comes out on top across multiple metrics
(`tm`, `itm`, `drmsd`), the choice is robust.

For complexes with antibody / TCR / pMHC sub-regions, **also run an
`anchored-rmsd` view of the variable region** (CDR loops, peptide
pose, TCR docking) and check whether the top cluster is consistent
with the BASE-metric top cluster. Disagreement means BASE is dominated
by the rigid scaffold and the variable region's mode is *different*
from the overall structure's mode — pick by the question you're asking.

> Note: `ranking_score` is the model's *self-assessed* confidence
> (AF3: `0.8·iPTM + 0.2·pTM` for complexes, `pTM` for monomers). It is
> not a distance to native — external validation (USAlign vs native,
> assessor scores, etc.) still applies. The wrapper records the full
> set (mean/min/max + medoid) for every metric and every confidence
> field in `<tag>_native_scores.csv`.

---

## Patches landed in [`massiveclustering/`](../../src/lantern_qa/cluster)

All backward compatible — none change existing call signatures except by adding optional kwargs.

| File | Change |
|---|---|
| `core/gpu/_loader.py` (new) | unified CIF/PDB loader (gemmi dispatch + per-chain residue exclude mask) |
| `core/gpu/{tmscore, drmsd, itm}.py` | `parse_pdb_ca` / `parse_pdb_with_chains` now use the new loader and accept an `exclude_indices` kwarg |
| `core/gpu/tmscore.py` | cache key changed from filename basename to list-index; previously every `model.cif` collapsed onto one entry. SVD core extracted to a shared `_kabsch_R(P, Q, weights)` utility used by both TM and anchored kernels |
| `core/gpu/drmsd.py` | broadcast diff replaced with the gram-matrix identity `‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩` to avoid 422 GiB allocations on large N |
| `core/gpu/anchored.py` (new) | anchored RMSD / TM kernel. Kabsch on `anchor_residues`, transform applied to `probe_residues`, RMSD or TM-score on probe. Reuses `_kabsch_R` and `compute_d0` from tmscore.py |
| `analysis/provenance.py` | `_METHOD_PATTERNS` extended with `^af3_`, `^bt2_`, `^of3_`, `^pt2_` for CASP17 unified-output filenames |
| `analysis/native_comparison.py` | added `ranking_score`, `plddt`, `gpde`, `fraction_disordered` to the recognised confidence columns; added `_min` alongside every existing `_mean` / `_max` |
| `visualization/comprehensive.py` | suptitle `y=1.01` removed and savefig pad added so long target names are not clipped |

---

## Known issues / design notes

- **Small-N monomers over-cluster.** Fuzzy c-means picks K ≈ N/2 when
  N ≲ 30 (e.g. 25 clusters from 30 structures = essentially
  unclustered). Workarounds: increase the seed budget so N ≥ 100, or
  switch to `--cluster-method leiden`. Forcing a fixed K via the
  driver is on the TODO.
- **AF3 monomer chain-id collapse.** AF3 always emits monomer chain as
  `A`; other runners keep the stoichiometry letter. The wrapper
  flags this as a "shape mismatch" and aborts unless
  `--allow-mismatch` is given.
- **dRMSD memory pressure on large N.** The gram-matrix path keeps the
  output `(N, N)` matrix on the GPU, but the input distance vector
  table is still `N × L²/2 × 4 bytes` (≈ 9 GB for L=822, N=6715).
  Should still fit on a 24 GB GPU.
- **`--cdr` impact is small when the chain pool is large.** CDRs are
  ~3 % of total residues in the H1311 test target — top clusters
  shift only marginally relative to baseline. The flag is more useful
  when isolating the antibody Fv on its own. For pulling out CDR
  signal in a large complex, `--method anchored-rmsd
  --anchor-chains AB --anchor-mode framework --probe-chains AB
  --probe-mode cdr` is a much cleaner choice (see section 2a).
- **`anchored-tm` over-clusters for short probes.** TM-score's `d0`
  has a hard floor of 0.5 Å (used when `L < 16`). On a 10 aa peptide,
  this means a 0.5 Å displacement halves the TM-term, producing very
  sharp distinguishability and K = 16-31 in our tests. Use
  `anchored-rmsd` (intuitive Å scale, K = 4-10) when the probe is
  short, and reserve `anchored-tm` for longer probes (≳ 50 aa, e.g.
  full TCR or CDR loop pool) or for cross-target comparison where
  length normalisation matters.
- **anchor must be rigid for the result to be meaningful.** If the
  anchor itself varies between predictions, the "anchor-aligned" frame
  is itself variable and any probe RMSD inherits anchor noise.
  H1311 MHC scaffold is well-converged (BASE TM > 0.99 on full
  complex) so this is fine; in cases where the anchor is flexible,
  prefer a smaller, more rigid sub-region as anchor.

---

## Direct invocation (skip Slurm)

```bash
# via wrapper (validates inputs, builds job name, etc.)
analysis/MassiveClustering_tool/run_clustering.sh --local targets/H1311 --sto A1B1C1D1E1 --method tm

# raw driver
analysis/.venv/bin/python analysis/MassiveClustering_tool/run_clustering.py \
    targets/H1311 --sto A1B1C1D1E1 --method tm --gpu 0
```
