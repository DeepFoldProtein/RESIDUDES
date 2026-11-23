@AGENTS.md

## Claude Code

- `.claude/skills` links to `skills/`.
- `.claude/settings.json` treats this session as the agent:
  - `LANTERN_ACTOR=agent`, so `lantern approve` refuses;
  - the read-only `lantern` commands are allowed;
  - `approve` is denied.
- Claude Code applies a project's allow list only in a trusted workspace. In a headless run, pass the tools with
  `--allowedTools` instead.
