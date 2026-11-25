# Built for Claude Code, Codex and Antigravity; run in Claude Code

One instruction file, [`AGENTS.md`](AGENTS.md), and one skill set, [`skills/`](../skills/), laid out so that each
harness finds them where it looks. That much is tested. Whether the other two harnesses behave as this table says
is not: residudes has been run in Claude Code and in the Agent SDK, and Codex and Antigravity are covered only by
tests against their documented layouts, because neither is installed on this cluster. The claim in this section is
therefore about the layout, not about observed behaviour, until someone runs
[`docs/harness_check.md`](harness_check.md) in them.

| harness | instructions | skills | how it is kept an agent |
|---|---|---|---|
| **Claude Code** | `CLAUDE.md` → `@AGENTS.md` | `.claude/skills` | `.claude/settings.json`: `RESIDUDES_ACTOR=agent`, read-only commands allowed, `approve`/`resolve` denied |
| **Codex** | `AGENTS.md` | `.agents/skills` | `residudes approve`/`resolve` refuse `RESIDUDES_ACTOR=agent`; the stage machine refuses an agent's move past a checkpoint |
| **Antigravity** | `AGENTS.md` | `.agents/skills` (and `.agent/skills` for older builds) | as for Codex |
| **`residudes agent step` / `daemon`** (Claude Agent SDK) | `AGENTS.md` appended to the system prompt | `.claude/skills` | plus a PreToolUse guard and writes confined to the dossier directory |

`residudes setup` links the skills into a project for all three harnesses. With `--copy-skills` it installs copies
instead of symlinks: Codex documents that it follows symlinks, and Antigravity's docs do not say (see
[`docs/harness_check.md`](harness_check.md)). [`tests/test_compat.py`](../tests/test_compat.py)
checks each harness's documented layout:
- skill discovery paths;
- frontmatter: `name` equals the folder, a `description` of at most 1,024 characters;
- `AGENTS.md` without frontmatter and under Antigravity's 24,000-byte limit;
- the `CLAUDE.md` import;
- the Claude Code settings;
- the SDK prompt.

Live checks so far:
- **Claude Code** (headless `claude -p` in this folder):
  - It loaded `CLAUDE.md` → `AGENTS.md`, listed all seven skills, explained both checkpoints, and named `residudes gate`
    (with `--init` and `--show`) for gating a pool (2026-11-22, $0.19).
  - Given a stage to do, it wrote and submitted the structural gates for a 14-3-3–phosphopeptide pool from the
    dossier's evidence ($1.43; $0.75 on the second attempt, after the fixes it prompted).
  - Earlier: a reviewer brief that flagged the unchecked gates, with the `approve` attempt blocked ($0.33); the
    read-only CLI with `--allowedTools` ($0.12).
- **The Agent SDK runner** took five CASP17 replays through the stages. Two reached a proposed slate and three wait at
  a checkpoint; [`examples/H1388`](../examples/H1388/) is the one that ships (below).
- **Codex and Antigravity** are not installed on the cluster. They are covered by the spec tests above until someone
  runs [`docs/harness_check.md`](harness_check.md) in them: the same question and expected answer, for every
  harness.
