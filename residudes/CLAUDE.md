@AGENTS.md

## Claude Code

- `.claude/skills` links to `skills/`.
- `.claude/settings.json` treats this session as the agent:
  - `RESIDUDES_ACTOR=agent`, so `residudes approve` refuses;
  - the read-only `residudes` commands are allowed;
  - `approve` is denied.
- Claude Code applies a project's allow list only in a trusted workspace. In a headless run, pass the tools with
  `--allowedTools` instead.
