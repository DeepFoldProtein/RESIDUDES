# RESIDUDES: instructions for coding agents

This repository holds three tools from the DeepFold team's CASP17 entry:

| folder | what it is | its own instructions |
|---|---|---|
| `lantern/` | the campaign agent: target analysis, checkpoints with a person, slate | `lantern/AGENTS.md` (read it before any campaign work) |
| `lantern-engine/` | inputs and predictions: Protenix v2, OpenFold 3, Boltz-2, AlphaFold 3 on PyTorch, TensorRT or JAX | `lantern-engine/README.md` |
| `lantern-qa/` | QA of a prediction pool: interface scores, clustering, plots | `lantern-qa/README.md` |

Harnesses read this file (Codex, Antigravity) or `CLAUDE.md` (Claude Code, which imports it). Opened from here,
they find lantern's skills through `.agents/skills`, `.agent/skills` and `.claude/skills`, which link to
`lantern/skills`. For work on a target, run from `lantern/` (`cd lantern`), where `lantern/AGENTS.md` applies.

Rules that hold everywhere in this repository:
- **Heavy compute** (predictions, QA, scoring) goes through `sbatch` or the pueue queues, never on a login node.
- **Never:**
  - delete recursively, or scan shared storage (`/`, `/gpfs`, `/store`);
  - poll faster than once a minute;
  - force-push.
- **Never act outward.** Do not submit to CASP, do not run `lantern approve` or `lantern resolve`, and do not change
  `LANTERN_ACTOR`. A person does those.
- **Unreleased natives.** Results scored against an unreleased native structure do not go into git.
