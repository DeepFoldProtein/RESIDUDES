---
name: lantern-setup
description: First-run onboarding for lantern - walk the user through resources, databases, config and skills. Use when someone sets lantern up on a new machine or account, or asks what this cluster can run.
---

# lantern setup, done with the user

Run `lantern setup` and go through its six parts with the user. Do not rush past the findings.

1. **Resources.** Read the report back in plain words: GPUs per partition, queue pressure, free space.
   - If a filesystem is ≥ 95 % full, say so. Propose a work root elsewhere (on the DeepFold cluster: `/store/deepfold/lantern`, since `/gpfs` has been 97 % full and the per-user GPFS quota has run out before).
2. **Databases / weights / tools.**
   - For every STALE or MISS line, explain what it breaks. In CASP17 the 2024 foldseek PDB hid homologs that were public, so "no homolog" from a stale DB is not evidence.
   - Propose the fix line. Ask before any download.
   - Downloads run as `sbatch` jobs on a compute node, never on the login node.
3. **Components.** lantern-engine and lantern-qa must both pass their own `doctor`. If not, show the failing lines.
4. **Config.** Confirm `work_root`, `targets_root`, the agent budget per step and the GPU-hour budget per target. Write `lantern.toml` only after a yes.
5. **Skills.** Offer to link the lantern skills into the user's project. Claude Code reads `.claude/skills`, Codex reads `.agents/skills`.
6. **Tour.** Create a test dossier with `lantern target add`, run `lantern next` and `lantern target show`, and explain the stages and that only a person runs `lantern approve`.

Never print secrets: AF3 weights path is fine, CASP author codes in `casp17-utils/.env` are not. Re-check later with `lantern doctor`.
