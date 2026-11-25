# RESIDUDES: instructions for coding agents

This repository holds three tools from the DeepFold team's CASP17 entry:

| folder | what it is | its own instructions |
|---|---|---|
| `residudes/` | the campaign agent: target analysis, checkpoints with a person, slate | `residudes/AGENTS.md` (read it before any campaign work) |
| `residudes-engine/` | inputs and predictions: Protenix v2, OpenFold 3, Boltz-2, AlphaFold 3, ESMFold2 on PyTorch, TensorRT or JAX | `residudes-engine/README.md` |
| `residudes-qa/` | QA of a prediction pool: interface scores, clustering, plots | `residudes-qa/README.md` |

Harnesses read this file (Codex, Antigravity) or `CLAUDE.md` (Claude Code, which imports it). Opened from here,
they find residudes's skills through `.agents/skills`, `.agent/skills` and `.claude/skills`, which link to
`residudes/skills`. For work on a target, run from `residudes/` (`cd residudes`), where `residudes/AGENTS.md` applies.

Rules that hold everywhere in this repository:
- **Heavy compute** (predictions, QA, scoring) goes through `sbatch` or the pueue queues, never on a login node.
- **Never:**
  - delete recursively, or scan shared storage (`/`, `/gpfs`, `/store`);
  - poll faster than once a minute;
  - force-push.
- **Never act outward.** Do not submit to CASP, do not run `residudes approve` or `residudes resolve`, and do not change
  `RESIDUDES_ACTOR`. A person does those.
- **Unreleased natives.** Results scored against an unreleased native structure do not go into git.
