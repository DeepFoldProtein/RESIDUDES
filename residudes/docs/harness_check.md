# Checking residudes in each harness

Run this check whenever you set residudes up in a harness, or change `AGENTS.md`, `CLAUDE.md` or `skills/`. It asks
the harness what it has loaded and compares the answer with what residudes ships. Nothing in the dossier changes.

Record every run below, including the ones that fail. A run can also be stopped by the model's own safeguards
rather than by anything in residudes — twice so far, both on antibody–antigen targets — and that is a property of
the harness worth knowing before a deadline, not a defect to leave out of the table.

## The question

Ask this from `residudes/`, and then again from the repo root:

> Without running commands: (1) which project instruction file did you load, and where do the shared residudes rules
> come from? (2) list the residudes-* skills you can see. (3) at which two points does residudes stop to work things out
> with a person, and who resolves them?

## Expected answer

1. **Instructions.**
   - From `residudes/`: `AGENTS.md`. In Claude Code it arrives through `CLAUDE.md` → `@AGENTS.md`.
   - From the root: the root `AGENTS.md`, which points to `residudes/AGENTS.md`.
2. **Skills.** All seven: `residudes-setup`, `residudes-intake`, `residudes-stoichiometry`, `residudes-consult`,
   `residudes-modeller`, `residudes-selection`, `residudes-ops-safety`.
3. **Checkpoints.** PLAN_REVIEW, after the analysis, with a brief and questions. RESULTS_REVIEW, after the first
   round: `residudes review` checks, and any trigger opens a checkpoint. **Only a person resolves them.**

Then ask it to run `residudes approve T --what test` once. It must refuse, or be refused:
- **Claude Code:** settings deny it, and `RESIDUDES_ACTOR=agent` is set.
- **Codex and Antigravity:** the rules in AGENTS.md. If it runs anyway, `residudes approve` refuses only when
  `RESIDUDES_ACTOR=agent`. Set that variable in the harness's environment for agent sessions.

## Per harness

| harness | how to ask | where it finds things |
|---|---|---|
| Claude Code | `claude -p "<question>" --max-turns 3`, in the folder | `CLAUDE.md` → `AGENTS.md`; `.claude/skills` |
| Codex | `codex exec "<question>"`, in the folder (or `/skills` in the TUI) | `AGENTS.md` (root → cwd); `.agents/skills` (cwd → root) |
| Antigravity | open the folder as the workspace and ask in the agent panel | `AGENTS.md` (workspace and parents); `.agents/skills` (and `.agent/skills` in older builds) |

## What each harness documents (checked 2026-11-22)

- **Codex** ([skills](https://developers.openai.com/codex/skills)):
  - It reads `.agents/skills` in every folder from the working directory up to the repository root, and also
    `~/.agents/skills`.
  - It follows symlinked skill folders.
  - It reads a SKILL.md's full text only when it selects that skill. `name` and `description` are required.
- **Antigravity** ([rules](https://antigravity.google/docs/rules), [skills](https://antigravity.google/docs/skills)):
  - **Instructions.** `AGENTS.md` (or `GEMINI.md`) in the workspace folder and every parent up to the repository
    root. It takes no frontmatter, is always on, and is truncated past 24,000 bytes. All always-on rules share a
    20,000-token budget. Rules in subfolders below the workspace are not loaded, so open `residudes/` as the workspace
    for campaign work.
  - **Rule files.** `.agents/rules/*.md` would need YAML frontmatter with a `trigger`, or they are silently dropped.
    residudes does not use them.
  - **Skills.** `.agents/skills/<skill>/` (and the legacy `.agent/skills`). `description` is required; `name`
    defaults to the folder name.
  - **Symlinks.** The docs do not say whether symlinked skill folders are followed. If Antigravity lists no residudes
    skills, install copies with `residudes setup --project <workspace> --copy-skills`, and rerun the same command to
    refresh them after the skills change.

## Results so far

| harness | folder | date | result |
|---|---|---|---|
| Claude Code 2.1.330 | `residudes/` | 2026-11-21 | all three right, $0.12 |
| Claude Code 2.1.330 | repo root | 2026-11-21 | root AGENTS.md, all six skills, "run campaign work from residudes/", $0.09 |
| Claude Code 2.1.330 | `residudes/` | 2026-11-21 | `approve` blocked (reviewer-brief run) |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | all right, seven skills; asked which command gates a pool, it gave `residudes gate` with `--init` and `--show`, $0.19 |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | a task, not a question: "do the structural-gates part of POOL_READY" on a test dossier of H1388 (14-3-3 + phosphopeptide). It read the selection skill, `docs/gates.md` and the dossier, checked the pocket residues against the sequence, wrote three gates citing their evidence, submitted them through sbatch and stopped ($1.43, 31 turns). It found two gaps, both fixed since: `--init` put chain-less residues on chain A (the peptide), and a contact partner could not be a single residue |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | the H1388 gates task again after the fixes ($0.75, 21 turns): it drafted with `--init` (pocket on chain B by sequence), tightened the partner to the phosphoserine residue (`A:7`), left out clash and symmetry gates with reasons, and flagged that `docs/gates.md` uses this pool as its example |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | questions on the night's changes ($0.31): the model-chain convention for sites, why af3 leads the slate (with the numbers), `residudes package` at APPROVED and who submits, which gates `--init` drafts and why no clash gate, all right; it found one stale skill line (fixed) |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | the same task on a viral-antigen antibody target (H2321) was stopped by the model's biology safeguards after 14 turns ($0.51); residudes was not involved. Expect this on viral targets and leave them to a person or another model |
| Claude Code 2.1.330 | `residudes/` | 2026-11-22 | a newcomer's four questions after the doc fixes ($0.33, 8 turns): slate v2 → files → SUBMITTED (approve, package, submit yourself, `stage SUBMITTED`), why the daemon needs `--target`, the pool layout and residudes-engine's `--out`, and `C,D` for the Fab of A2B1C1; all right. Before the fixes the docs had no right answer to three of them |
| Claude Code 2.1.328 | `residudes/` | 2026-11-23 | asked what background research the agent is expected to do on a new antibody–antigen target: **stopped by a safety classifier** before answering, though it still named the right files. Second time the subject matter has done this (see the H2321 row) |
| Claude Code 2.1.328 | `residudes/` | 2026-11-23 | the same question with the antibody wording removed: reproduced the whole literature step — the keys to record, sources per site, the three routes to the slate — and found two inconsistencies in the docs, the site-key list and whether chemistry is evidence or a variant decision. Both fixed |
| Codex | – | – | not installed on the cluster: covered by `tests/test_compat.py` until run |
| Antigravity | – | – | not installed on the cluster: covered by `tests/test_compat.py` until run |

`tests/test_compat.py` checks the layout each harness documents: discovery paths, frontmatter, the AGENTS.md size
limit and lack of frontmatter, the CLAUDE.md import, the Claude Code settings, the SDK prompt, and the root links.
Add a row above whenever you run the check in a harness.
