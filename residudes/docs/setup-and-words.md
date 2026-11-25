# Setting up, and the words used

## Words used

| word | meaning |
|---|---|
| **dossier** | a target's folder, `<work_root>/targets/<T>/`, around `dossier.json`: evidence, decisions, stage, log |
| **stage** | where the target is (NEW, IDENTIFIED, …, SLATE_PROPOSED, APPROVED, …); each stage has a next step |
| **sto** | a stoichiometry such as `A3` or `A2B1C1` (`--stoich` at `target add`, `--sto` after; it defaults to the given one) |
| **pool** | every prediction of the target, read from a residudes-engine layout into `pool.csv` |
| **runner** | the predictor: af3 = AlphaFold 3, pt2 = Protenix v2, of3 = OpenFold 3, bt2 = Boltz-2, ef2 = ESMFold2 (`_jax`: their JAX ports) |
| **variant** | the input a run used (`af3_msa`: AF3's MSA; `mmseqs2_gpu`; a template or ligand variant) |
| **gate** / **census** | a structural test a model must pass / a per-model label of which named site a partner touches |
| **slate** | the five models proposed for submission; a person approves it |
| **checkpoint** | a conversation with a person at PLAN_REVIEW or RESULTS_REVIEW; only a person resolves it |

`residudes setup` goes through six parts:
1. **Resources.** This machine, its GPUs, the Slurm partitions and queue, storage and quota, and the tools; then recommendations.
2. **Databases.** The databases, weights and tools in `catalog.toml`: present or missing, and how old.
3. **Components.** residudes-engine's and residudes-qa's own `doctor`.
4. **Config.** `residudes.toml`: where dossiers live, budgets, model.
5. **Skills.** Links into your project's `.claude/skills`, `.agents/skills` and `.agent/skills`.
6. **Tour.** The commands below.

Nothing is downloaded or overwritten without a yes.
