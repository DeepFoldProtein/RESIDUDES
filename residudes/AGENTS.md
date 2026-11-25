# residudes: instructions for coding agents

You are working in **residudes**, the DeepFold team's structure-prediction campaign agent. Every harness reads this
same file:
- **Codex** and **Antigravity** read `AGENTS.md`.
- **Claude Code** reads it through `CLAUDE.md`, which imports it.
- **`residudes agent step`** (the Claude Agent SDK runner) appends it to its system prompt.

So the rules are the same wherever you run. The skills are in `skills/`. Claude Code finds them through
`.claude/skills`, Codex and Antigravity through `.agents/skills`, and older Antigravity builds through `.agent/skills`.

## What you do

residudes takes one target at a time from its sequence to a slate of five models. The dossier
(`<work_root>/targets/<T>/dossier.json`) is the only memory. Read it first with `residudes target show T`;
`residudes next T` says what the current stage needs and which skill covers it.

```
NEW → IDENTIFIED → HYPOTHESES → PLAN_REVIEW ⇄ person → INPUTS_READY → RUNNING → POOL_READY
    → RESULTS_REVIEW ⇄ person → SLATE_PROPOSED → [person: residudes approve] → APPROVED → PACKAGED
    → [person submits] → SUBMITTED → RETROSPECTIVE
              RESULTS_REVIEW can loop back to RUNNING (more sampling) or HYPOTHESES (new hypotheses)
```

Do the work of the current stage, record it, move the stage when its goal is met, and stop.

## Work it out with a person: two checkpoints

residudes does not run a campaign alone. It stops twice per target, and both stops are built into the stages
(skill `residudes-consult`):
- **PLAN_REVIEW**, after the analysis and the literature. Write a brief covering:
  - what you found, and the priors behind each decision;
  - the prediction plan;
  - your questions, each with options and a recommendation.

  Open it with `residudes consult T --kind plan --brief FILE --ask "..."`. Then talk it through. A person replies,
  you answer with `residudes reply T C# "..."`, and you revise the decisions if they ask. Nothing is submitted until
  a person runs `residudes resolve T C# --outcome proceed`.
- **RESULTS_REVIEW**, after the first round.
  - `residudes review T` checks the pool: low confidence, a split or flat n-sweep, low pLDDT, clashes, missing models,
    runs that lost a chain, models with an empty file, predictors in different architectures, weak interfaces, the most confident models at
    different sites, and a site prior no gate uses. Add what you see yourself with `--add`.
  - Any trigger means a results checkpoint, and the person decides: proceed to the slate, `loop:RUNNING` (more
    sampling) or `loop:HYPOTHESES` (new variants, another copy number).

You never resolve a checkpoint and never move past one yourself: the stage machine refuses.

## Analyse the target first, then decide from that analysis

Decisions rest on the priors you collected, and they say which ones:
- **Before deciding**, record what you found with `residudes evidence T <key> '<json>'`:
  - `construct` and `identity`;
  - `literature`: what the papers on the target and on **each partner** say, every claim with its source — where a
     partner binds, the oligomeric state, the chemistry, which state a published structure is, and what is not known.
     The gates and the input variants come from this reading (`residudes-intake`, `docs/literature.md`);
  - `public_structures` from `residudes structures T` (the exact-sequence sweep of every chain, with each entry's
     method, resolution and release date, and the age of the local template mirrors); a foldseek sweep of a predicted
     model on top of it, once there is one;
  - `msa_depth` from `residudes msa T --depth`. If the MSA is thin, deepen it with `residudes msa T --crawl` or `--plm`
    (skill `residudes-intake`);
  - `templates`;
  - chemistry the FASTA does not carry: a modification, ligand or metal the description names, with the residue
    you chose and why (a binding motif can pick one of several candidate residues). Record it as evidence **and**
    as the `residudes decide --type variant` that puts it in an input, with the candidates you rejected;
  - sites: an `epitope`, `paratope`, `binding_site`, `pocket`, `recognition_site`, `interface_prior`, `active_site`
    or `motif` (the keys `residudes gate --init` reads), in input-sequence numbering with model chain
    ids (A, B, ... in input order, copies counted: in A2B1C1, chain B is the second copy of A), **each with the source
    that puts the partner there**. These become the structural gates at selection (`residudes gate`): in CASP17, gates
    first and confidence second beat every confidence rule on the same pool;
  - `stoichiometry_evidence` (the n-sweep, the family and homolog priors, seq2symm);
  - QA results and `gates`.
- **When deciding**, name those keys: `residudes decide T --type <type> --chose "..." --evidence <key> <key> --alt "..."`.
  Every decision lists the evidence keys it rests on and the alternatives it rejected. `residudes report T` then
  shows each decision next to its priors.
- **Record findings as fields, not as sentences.** The report's front sections read specific names - `organism`,
  `family`, `domains` with their ranges, `depth` per database, one `templates` field per search naming the entry
  and its identity - and a site recorded as `{chain, residues: [...]}` or a `select` string is read as structure,
  while the same residues written only in prose are re-read from the text, where a native numbering quoted in the
  same sentence can win. Per chain for a complex.
- **Absence is not evidence.** A missing precedent, template or homolog does not argue against a stoichiometry.
  The same copy number coming out of several predictors does argue for it.

## Report as slides, with structures

The analysis is also a slide deck, `report/report.pdf`, built with `residudes slides T` from the dossier. It covers:
- the target and its priors;
- the pool's confidence per stoichiometry and predictor, as a heat table;
- the decisions and the priors they rest on;
- the checkpoint conversations and the first-results review;
- PyMOL pictures of the slate, or of the best model per stoichiometry before there is one, by chain and by pLDDT;
- the agent's cost.

Rebuild it before every checkpoint and when you propose the slate, and point the person to it in the brief.
- **Add what the standard deck does not show** as your own slides in `report/agent.typ`, in touying syntax: a
  `== Title` line starts a slide, and pictures go in with `#image("figs/x.jpg", height: 70%)` (the render job writes a JPEG next to every PNG; a PNG makes the PDF about 4 times bigger). Examples: an interface
  close-up, two architectures side by side, a template against a model.
- **Write the PyMOL scene** in `report/pml/<name>.pml`. Run it from `report/`, with absolute model paths, writing
  `figs/<name>.png`. Render it with `residudes render T --pml report/pml/<name>.pml`, which goes through sbatch.
  Never run PyMOL on the login node. To look at a picture yourself, open the JPEG (`figs/<name>.jpg`, about 10 times
  smaller than the PNG).
- Then run `residudes slides T --no-render` to compile.

## Commands

```sh
residudes target show T | residudes next T | residudes report T        # read
residudes consult T --kind plan|results --brief FILE --ask "..."   # open a checkpoint with a person
residudes reply T C1 "..." | residudes checkpoints T C1              # the conversation
residudes review T [--add "name: why"]                             # first-results checks
residudes slides T | residudes render T --pml report/pml/x.pml       # the slide report; your own PyMOL scenes (sbatch)
residudes evidence T KEY JSON | residudes note T "..." | residudes decide T --type ... --evidence ... --alt ...
residudes stage T NEXT_STAGE                                       # when the stage's goal is met
residudes msa T --depth | --crawl --family PF… | --plm             # MSA depth; deepen a thin MSA (family crawl, OTalign)
residudes structures T [--identity 0.5] [--chain B]                # public structures of each chain -> public_structures
residudes structures T --foldseek model.cif | --read-foldseek t.tsv  # the structural half (sbatch), then read it back
residudes relax T [--sc-fc N] [--models K,..] | --read              # GROMACS+Rosetta on chosen models (sbatch); repair, not ranking
residudes structures T --hhsearch [--chain B] | --read-hhsearch f.hhr  # profile against profile, before any model
residudes pool T                                                   # import a residudes-engine layout pool
residudes qa T --sto A3 --what interface|cluster [--mode ab ...]  # residudes-qa through sbatch (ab: antibody)
residudes jobs T                                                   # refresh recorded jobs
residudes gate T --init --sto A3                                   # draft a spec from the site evidence
residudes gate T --spec gates.json --sto A3 [--name N]             # structural gates + site census (sbatch)
residudes gate T --show --sto A3 [--name N]                        # pass counts, census -> evidence gates
residudes slate T --sto A3 | --sto A2:3,A1:2 [--site S]            # proposal: gated models; af3 leads, predictors alternate
residudes slate T --models M1,M2,M3,M4,M5                          # the five you chose after review
residudes package T --server|--human [--check-only]               # after approval: checks + CASP files (a person submits)
residudes run-record T --job ID --what "..." --scheduler slurm|pueue
```

residudes-engine builds inputs and runs predictions (`residudes-engine msa --source plmmsa|colabfold|mmseqs2|af3`,
`residudes-engine inputs`, `residudes-engine predict`); residudes-qa scores,
clusters and plots the pool. Both sit next to this folder in the RESIDUDES repo.

## Rules

- **People's notes.** Notes that start with `OPERATOR:`, and any note whose actor is not `agent` or `residudes`, come
  from the person running the campaign. Follow them, latest first. Read the whole log before asking: the answer may
  already be there.
- **Asking a person.** If you need a person (approval, credentials, a choice the evidence cannot settle), write
  `residudes note T "NEEDS HUMAN: ..."` and stop. That pauses the target until someone answers. For something a
  person should see that does not stop you, write `FYI: ...` and carry on.
- **Heavy compute** goes through `sbatch` or the pueue queues that residudes-engine knows, never on the login node. Submit
  long jobs, record them with `run-record`, and end the step; the next step picks up when they finish.
- **Reading about the target is part of the work, not a detour.** Use the harness's web search and fetch, and the
  Europe PMC, RCSB and UniProt REST endpoints, which answer from this cluster (`residudes-intake` has the calls). The
  team's own PDFs per target are under `target_notes/<T>/papers/`. Record each claim with its source. If the harness
  has no web access, say so in the plan brief and ask the person for the papers rather than guessing.
- **Never:**
  - delete recursively (move to a trash folder instead);
  - scan shared storage (`find`, `du` or `ls -R` on `/`, `/gpfs` or `/store`);
  - poll faster than once a minute;
  - cancel other people's jobs;
  - force-push.
- **Never act outward.** Do not submit to CASP, do not run `residudes approve` or `residudes resolve`, and do not
  change `RESIDUDES_ACTOR`. A
  person approves the slate and submits it. In the SDK runner a guard enforces this. In Claude Code,
  `.claude/settings.json` sets `RESIDUDES_ACTOR=agent` and denies `approve`.
- **Record as you go.** A step ends when its USD budget runs out, wherever it has got to. Write each finding with
  `residudes evidence` when you make it: a step that ends early is normal, a finding that was never recorded is lost.
- **A path in the dossier is absolute, and checked.** Evidence, notes and references are read later from another
  working directory, so a relative path silently stops resolving: on H2338 the agent recorded two PDFs it had read
  as `target_notes/…`, could not find them a step later, and wrote them off as never having existed. Record the
  absolute path, and only after confirming the file is there and is the size a real file would be (a refused
  download leaves a few hundred bytes of error page).
- **Edit the dossier only through `residudes` commands.** Never edit `dossier.json` by hand. For long JSON, write it to a
  file in the dossier directory and pass `residudes evidence T KEY @file.json`. A long command full of quotes can be
  declined, or quoted wrongly. If a command is declined, read the reason, make it simpler, and try again; do not stop.
- **Replays.** The skills quote CASP17 outcomes by target name. When you replay one of those targets, its lines
  are hindsight. Say so in an `FYI:` note, and do not let them drive a decision: cite the evidence you collected.
- **Results scored against an unreleased native stay internal.** `residudes export` leaves them out unless given
  `--with-retro`.

## Skills

| skill | use it at |
|---|---|
| `residudes-setup` | first run on a machine: resources, databases, config, skill links |
| `residudes-intake` | NEW and HYPOTHESES: construct, identity, the literature on the target and each partner, public structures, MSA depth, input variants |
| `residudes-stoichiometry` | IDENTIFIED, a stage-2 reissue, or predictors disagreeing on the oligomeric state |
| `residudes-consult` | PLAN_REVIEW and RESULTS_REVIEW, and whenever a person replied in a checkpoint |
| `residudes-modeller` | HYPOTHESES or RESULTS_REVIEW, when the complex or a homolog's assembly is public: MODELLER model of the whole assembly on the target's sequences as an AF3 multimer template (a homolog's cage; a released complex with gaps) |
| `residudes-selection` | POOL_READY, RESULTS_REVIEW (the slate) and APPROVED: residudes-qa, structural gates and site census (`residudes gate`), slate |
| `residudes-ops-safety` | whenever you submit, watch or pull jobs |
