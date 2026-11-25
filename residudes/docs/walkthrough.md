# A target, from the first command to the retrospective

```
NEW → IDENTIFIED → HYPOTHESES → PLAN_REVIEW ⇄ person → INPUTS_READY → RUNNING → POOL_READY
    → RESULTS_REVIEW ⇄ person → SLATE_PROPOSED → [person: residudes approve] → APPROVED
    → [residudes package] → PACKAGED → [person submits, then `residudes stage T SUBMITTED`] → SUBMITTED → RETROSPECTIVE
  loops: PLAN_REVIEW → HYPOTHESES (rework the plan) · POOL_READY → RUNNING · RESULTS_REVIEW → RUNNING (more sampling)
         or HYPOTHESES (new hypotheses) · SUBMITTED → HYPOTHESES (a stage-2 reissue)
```

The same thing, one picture per stage ([`mascot/`](../../mascot/), which also has them with a transparent background
for slides):

| | | | |
|:--:|:--:|:--:|:--:|
| <img src="../../mascot/thumb/new.png" width="150"><br>**NEW**<br>the sequence and one line | <img src="../../mascot/thumb/identified.png" width="150"><br>**IDENTIFIED**<br>construct, identity, the papers | <img src="../../mascot/thumb/hypotheses.png" width="150"><br>**HYPOTHESES**<br>the input variants | <img src="../../mascot/thumb/plan_review.png" width="150"><br>**PLAN_REVIEW**<br>*a person holds the pen* |
| <img src="../../mascot/thumb/inputs_ready.png" width="150"><br>**INPUTS_READY**<br>MSAs and templates built | <img src="../../mascot/thumb/running.png" width="150"><br>**RUNNING**<br>seeds through sbatch | <img src="../../mascot/thumb/pool_ready.png" width="150"><br>**POOL_READY**<br>every model imported | <img src="../../mascot/thumb/results_review.png" width="150"><br>**RESULTS_REVIEW**<br>*a person decides the loop* |
| <img src="../../mascot/thumb/slate_proposed.png" width="150"><br>**SLATE_PROPOSED**<br>gates first, then confidence | <img src="../../mascot/thumb/approved.png" width="150"><br>**APPROVED**<br>*the stamp is theirs* | <img src="../../mascot/thumb/packaged.png" width="150"><br>**PACKAGED**<br>geometry checks, CASP files | <img src="../../mascot/thumb/submitted.png" width="150"><br>**SUBMITTED**<br>*a person submits* |
| <img src="../../mascot/thumb/retrospective.png" width="150"><br>**RETROSPECTIVE**<br>scored against the native | <img src="../../mascot/thumb/loop_back.png" width="150"><br>**a loop**<br>more sampling, or new hypotheses | <img src="../../mascot/thumb/safety_refusal.png" width="150"><br>**a refusal**<br>the target is parked, not retried | <img src="../../mascot/thumb/human_only.png" width="150"><br>**a person's move**<br>approve, resolve, submit |

The four with a human hand in them are the rule, not decoration: residudes proposes, a person decides.

```sh
bin/residudes target add T1234 --fasta T1234.fasta [--stoich A3]   # dossier: <work_root>/targets/T1234/dossier.json
bin/residudes next T1234                                            # the next step and the skill that covers it
bin/residudes agent step T1234                                      # the agent does that step, then stops
bin/residudes daemon --target T1234                                 # or: step the named targets whenever they can move
bin/residudes target show T1234                                     # the dossier (or `residudes report T1234`: the analysis)
```

- An agent step does the work of the current stage, records findings (`evidence`) and choices (`decide`, with the
  evidence keys and the rejected alternatives), moves the stage when the goal is met, and stops.
- The next step starts from the dossier alone, so steps can be repeated, interrupted, or done by a person.

### The analysis document, and the figures it draws

`residudes report T` reads the dossier and writes `analysis.md`; `residudes slides T` puts the same material in a deck
(`report/report.pdf`). Both lead with the target rather than with the record: what it is, the sequence counted per
chain, what the literature says, the sites, the motifs, MSA depth, the structures that exist, and the pool once
there is one. The whole raw record follows, so nothing is only summarised.

Each kind of evidence gets **its own figure**, drawn from that evidence alone (`residudes/src/residudes/figures.py`,
stdlib SVG, no job to run) — one picture that mixes four questions answers none of them:

| figure | drawn from | what it shows |
|---|---|---|
| `chain_map` | the FASTA, `identity`, `construct`, the site keys | each chain to scale, its domains, and the residues a gate would use |
| `literature` | `literature` | every claim with its source, which priors and decisions share an identifier with it, and what the reading says is **not** known |
| `sites` | `binding_site`, `epitope`, `motif`, … and `chemistry` | each site in its own sequence, residue by residue, with what the record says it does and where that came from |
| `motif` | any motif written anywhere in the evidence | every place the pattern fits the sequence, which match carries the recorded residue, and the consensus patterns that fit nowhere |
| `clusters` | `residudes qa T --what cluster`, `pool.csv`, the interface scores | the same models four ways — by cluster, by predicted DockQ, by the predictor's own confidence, and one small panel per predictor showing where its models sit — then the clusters ranked by mean predicted DockQ with what is in each, and what that says in sentences |

Two rules hold for all of them: everything drawn comes from a number that was recorded or computed here, and
anything the parser could not place is named on the figure instead of being dropped. A figure with nothing to draw
is not drawn, and an older one is deleted rather than left to be linked.

Beside them the deck renders **the medoid of each cluster** with PyMOL — one view for all of them, superposed, with
any chain short enough to vanish (a peptide, a ligand-sized subunit) drawn as spheres — because a cluster is a
claim that these models are the same shape, and the medoid is what that shape looks like. And both documents open
with **one page of findings**: what the target is, the chemistry the FASTA does not carry, MSA depth, the
structures that exist, what the pool does, what nobody knows, and what is waiting on a person — each line read out
of the dossier, with its numbers, and left out when there is no evidence for it.

### The two checkpoints

- **Plan review (PLAN_REVIEW).** After the analysis the agent writes a brief and opens a checkpoint. The brief covers:
  - what it found, and which priors each decision rests on;
  - the prediction plan: stoichiometries, variants, seeds, GPU hours;
  - its questions, each with options and a recommendation;
  - the figures above, redrawn as the checkpoint opens and linked from the brief, so the person decides from the
    picture of the sites and the motifs rather than from a sentence about them;
  - what the analysis never recorded (no organism, no family, no MSA measurement, no public-structure search, no
    write-up, a site with no residues field) — each may be right for the target, so it is a list, not a block.

  The person and the agent talk it through (`residudes reply`). When the person writes, the daemon gives the agent a
  step to answer. Nothing is submitted until the person runs `residudes resolve T C1 --outcome proceed`.
- **Results review (RESULTS_REVIEW).** `residudes review T` checks the first round:
  - low confidence;
  - a split or flat n-sweep;
  - low pLDDT;
  - clashes;
  - missing models;
  - predictors in different architectures;
  - weak interfaces;
  - runs that lost a chain, and models whose file is empty;
  - the most confident models at different sites (from a gate census);
  - a site in the evidence that no gate uses, or one whose source is not recorded.

  The agent adds what it sees itself. Any trigger opens a results checkpoint, and the person decides: proceed to the
  slate, `loop:RUNNING` or `loop:HYPOTHESES`.

```sh
bin/residudes consult T --kind plan --brief brief.md --ask "Copy number: (a) A3 [rec.], (b) A1, (c) A4"   # the agent
bin/residudes checkpoints T C1                       # read the conversation
bin/residudes reply T C1 "why not A6? the family has hexamers"                                        # either side
bin/residudes resolve T C1 --outcome proceed         # people only (also loop:RUNNING / loop:HYPOTHESES)
bin/residudes review T [--add "name: why"]           # first-results checks -> triggers
```

### Pool, QA, slate, retrospective

**Before `residudes pool`.** It reads `<dir>/results/<sto>/<variant>/<runner>/seed_<N>/sample_<M>/` (`summary.json`
and `model.cif`). Per-chain runs go under `<dir>/<chain>/results/...`. A pool of this campaign is public if you
want one to try these commands on: [residudes-decoy](https://residudes-decoy.deepfold.org), 726,097 CASP17 models
with the input of each ([`docs/residudes-decoy.md`](../../docs/residudes-decoy.md)).
- `<dir>` is `--dir`, or `<targets_root>/<T>` from `residudes.toml`.
- The folder names become the names everything else uses: `--sto A3`, and models like `A3__af3_msa__af3__s97_3`.
- residudes-engine writes the seed part, so point it there:
  - `residudes-engine predict --seed 1 --out <dir>/results/A3/af3_msa/pt2/seed_1`, or
  - `residudes-engine submit --seeds 1-20 --out <dir>/results/A3/af3_msa/pt2`.

```sh
bin/residudes pool T1234                       # read the pool -> pool.csv, per-runner confidence (--dir, --threads)
bin/residudes qa T1234 --what interface        # residudes-qa through sbatch (interface | cluster); antibody: --mode ab
bin/residudes jobs T1234                       # refresh the recorded jobs (wait for COMPLETED)
bin/residudes evidence T1234 epitope "A:611,615 (escape mutants, J Virol 2024)"   # a site and its source
bin/residudes gate T1234 --init                # draft -> <work_root>/targets/T1234/gates/draft_<sto>.json (a new file each run)
bin/residudes gate T1234 --spec <work_root>/targets/T1234/gates/draft_A3.json   # evaluate it (sbatch; --local up to 100)
bin/residudes gate T1234 --show                # after the job: pass counts per gate and predictor -> evidence gates
bin/residudes slate T1234                      # propose 5 from the models that pass: af3's best leads, runners alternate,
                                             # one per cluster/seed in slots 1-3 (--site S; antibodies: --no-interface)
bin/residudes slate T1234 --sto A2:3,A1:2      # a hedge across stoichiometries, slots in that order
bin/residudes slate T1234 --models M1,...,M5   # the five chosen after review
bin/residudes approve T1234 --what "slate v2"  # people only: moves it to APPROVED; names models that fail its gates
bin/residudes package T1234 --server           # geometry checks, polymer-only TS files via casp17-utils `prepare`
bin/residudes stage T1234 SUBMITTED            # people only, after submitting the files yourself
bin/residudes retro T1234 --native native.pdb  # once the native is out: score the slate models (sbatch, OpenStructure)
bin/residudes retro T1234 --scores scored.csv  # with pool models scored against the native (lddt/dockq_wave columns):
                                             # every slate version, the baselines, and whether the gates kept the good models
bin/residudes export T1234 examples/T1234      # dossier, analysis.md, checkpoints, step costs, logs, slates, QA and gate summaries
```

`--sto` defaults to the stoichiometry given at `target add --stoich`.

## The worked example

[`examples/H1388`](../examples/H1388/) is one live run on a CASP17 target: human 14-3-3 gamma with an 11-residue LARP1
peptide, declared A1B1, from `target add` to a packaged slate under the default mandate. Fifteen agent steps, $6.81,
45 minutes, with nobody answering anything: NEW → IDENTIFIED → HYPOTHESES → PLAN_REVIEW → INPUTS_READY → RUNNING →
POOL_READY → RESULTS_REVIEW → SLATE_PROPOSED → APPROVED → PACKAGED, where it stops, because submission is a person's.

It identifies both chains (14-3-3 gamma P61981; LARP1 542–552, Q6PKG0), picks the phosphosite out of two serines by
motif and annotation, finds 23 exact-sequence PDB entries and pins the best peptide-bound one (6S9K, 1.60 Å), builds
two inputs - the peptide with phosphoserine and without - and submits 60 prediction jobs across AF3, Protenix,
Boltz-2 and OpenFold3. Both checkpoints are written as for a person and answered by the agent: at C1 it closes two
inconsistencies in its own plan; at C2 it finds that the predictors split on the peptide's register and that only
AF3 carried the phosphate, and builds the slate from the two that agree.

The folder holds:
- the analysis (`analysis.md`: priors, decisions and the prior → decision map);
- the write-up the agent wrote from the dossier, with the figures it cites (`notes/target.md`, `report/figs/`);
- both checkpoints with the agent's answers (`checkpoints/`), the slate (`slate_v2.md`) and the packaged files
  with their geometry checks (`package/`);
- every step's transcript and cost (`agent_logs/`, `steps.json`).

It does not ship the 300 models or the MSAs (1.2 GB); `pool.csv` lists every model with its scores. Paths are
written as placeholders (`<WORK>`, `<AF3_DB>`, ...), listed in
[`docs/provenance.md`](../examples/H1388/docs/provenance.md), which also records the one fix made mid-run.

Against the native, released after CASP17, all five packaged models are high-quality by DockQ, model 1 at 0.870
([`docs/retrospective.md`](../examples/H1388/docs/retrospective.md)). That page also says why this is not a blind
result: the intake skill was written after CASP17 and quotes this target's lesson, and the agent said so itself.

**The reading.** The run records sourced claims under `literature` - the two 14-3-3 consensus motifs and the groove
that holds them (Yaffe 1997, PMID 9428519), the arginines that grip the phosphate in this construct's own numbering
(UniProt P61981), UniProt's annotation of LARP1 S548, and why only S7 of the peptide fits the pS-X-P core - beside a
`not_known` field saying that no paper on the 14-3-3:LARP1 complex itself was found, with the three Europe PMC
queries it tried. That is what the source requirement in [docs/literature.md](literature.md) asks for; the earlier
replays (T1416, T1331) recorded no `literature` key at all and their site priors carry no source field, which, with
the H2319 case, is why it was written down.
