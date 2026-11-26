"""`residudes agent step T`: one agent step on one target, through the Claude Agent SDK.

A step reads the dossier, does the work of the current stage (see dossier.PLAYBOOK) with the residudes,
residudes-engine and residudes-qa command lines, records findings and decisions in the dossier through `residudes`
commands, and moves the stage when the stage's goal is met. Then it stops. A scheduler or a person
calls the next step; nothing carries over between steps except the dossier.

Limits on every step:
  - shell commands pass residudes.guard (no recursive deletes, broad scans, mass cancels, submissions)
  - files may only be written under this target's dossier directory
  - max_turns and max_budget_usd from residudes.toml (or the command line)
  - outward actions (CASP submission) are never available; APPROVED needs a person's `residudes approve`
"""
from __future__ import annotations

import asyncio
import datetime as dt
import fcntl
import os
from pathlib import Path

from residudes import guard
from residudes.config import RESIDUDES_DIR, Config
from residudes.dossier import Dossier

ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Skill", "Bash", "Write", "Edit", "WebSearch", "WebFetch", "TodoWrite"]
HUMAN_STAGES = ("SLATE_PROPOSED", "PACKAGED", "SUBMITTED")     # a person acts here (approve, submit, wait)


AGENTS_MD = RESIDUDES_DIR / "AGENTS.md"                       # the same instructions Claude Code, Codex and Antigravity read


def _prompt(d: Dossier, cfg: Config) -> tuple[str, str]:
    goal, skill = d.next_step()
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    shared = AGENTS_MD.read_text() if AGENTS_MD.exists() else ""
    system = f"""You are residudes, the structure-prediction campaign agent of the DeepFold team, running one step on {d.target}
through the Claude Agent SDK. Your memory is the dossier; read it first (`residudes target show {d.target}`).

This run:
  dossier directory (your working directory; write files only here): {ddir}
  residudes-engine: {cfg.residudes_engine}
  residudes-qa: {cfg.residudes_qa}
  existing predictions (residudes-engine layout), if any: {cfg.targets_root or '(none configured)'}
    (import only these, your own runs, or a directory a person names; predictions you come across elsewhere -
    beside the FASTA, in a team folder - are a finding to record, not a pool to import)
  `residudes` is on PATH; RESIDUDES_ACTOR=agent.

The instructions below are residudes/AGENTS.md, shared with Claude Code, Codex and Antigravity.

{shared}"""
    user = (f"Target {d.target} is at stage {d.stage}. This step: {goal}."
            + (f" Use the `{skill}` skill." if skill else " Nothing for the agent to do at this stage unless the dossier says otherwise.")
            + f" Budget: {cfg.agent_usd_per_step} USD of agent time; GPU budget {cfg.gpu_hours_per_target} h for the target."
            + " Record each finding with `residudes evidence` as you make it, not at the end: the step stops when the"
            " budget runs out, and a step that ends early is normal, while a finding that was never recorded is lost."
            + " Do not wait for jobs inside the step (no background watchers, no sleep loops): submit, record, and end"
            " it; the daemon runs the next step when the jobs finish.")
    return system, user


def _write_hook(allowed_root: Path):
    async def hook(input_data: dict, tool_use_id, context) -> dict:
        if input_data.get("tool_name") not in ("Write", "Edit"):
            return {}
        fp = Path(str(input_data.get("tool_input", {}).get("file_path", ""))).resolve()
        if fp.name in ("dossier.json", ".dossier.lock", "dossier.json.tmp"):
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": "residudes guard: change the dossier only through `residudes` commands"}}
        if str(fp).startswith(str(allowed_root.resolve()) + os.sep):
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": f"residudes guard: write only under {allowed_root}"}}
    return hook


async def _step(d: Dossier, cfg: Config, budget: float, max_turns: int, model: str, verbose: bool) -> dict:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
    try:
        from claude_agent_sdk import ToolResultBlock, UserMessage
    except ImportError:                                        # older SDKs
        ToolResultBlock = UserMessage = None

    tdir = Dossier.path(cfg.targets_dir, d.target).parent
    logs = tdir / "agent_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{d.stage}.log"
    system, user = _prompt(d, cfg)
    bins = [str(RESIDUDES_DIR / "bin"), str(Path(cfg.residudes_engine) / "bin"), str(Path(cfg.residudes_qa) / "bin")]
    from residudes.setup import CLAUDE_SETTINGS
    options = ClaudeAgentOptions(
        settings=CLAUDE_SETTINGS,                          # a flagged message switches models instead of failing the step
        system_prompt={"type": "preset", "preset": "claude_code", "append": system},
        cwd=str(tdir),
        add_dirs=[str(RESIDUDES_DIR), cfg.residudes_engine, cfg.residudes_qa] + ([cfg.targets_root] if cfg.targets_root else []),
        setting_sources=["project"],
        skills="all",
        allowed_tools=ALLOWED_TOOLS,
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[guard.hook(d.target if d.automation_allows(d.stage) else None)]),
                              HookMatcher(matcher="Write|Edit", hooks=[_write_hook(tdir)])]},
        max_turns=max_turns,
        max_budget_usd=budget,
        max_buffer_size=64 * 1024 * 1024,                  # a Read of a rendered picture can exceed the 1 MB default
        model=model,
        env={"PATH": os.pathsep.join(bins + [os.environ.get("PATH", "")]),
             "RESIDUDES_CONFIG": cfg.path, "RESIDUDES_ACTOR": "agent",
             "CLAUDE_CODE_AUTO_CONNECT_IDE": "false"},          # never the person's editor: see chat._agent_env
    )
    result: dict = {}
    with open(log, "w", buffering=1) as fh:            # line-buffered: tail -f works
        fh.write(f"# system (appended)\n{system}\n\n# prompt\n{user}\n\n")
        async for msg in query(prompt=user, options=options):
            if isinstance(msg, AssistantMessage):
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        line = b.text.strip()
                    elif isinstance(b, ToolUseBlock):
                        line = f"[{b.name}] {str(b.input)[:800]}"
                    else:
                        continue
                    fh.write(line + "\n")
                    if verbose:
                        print(line, flush=True)
            elif UserMessage is not None and isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for b in msg.content:                          # refusals and errors go into the log (with their reason)
                    if isinstance(b, ToolResultBlock) and getattr(b, "is_error", False):
                        text = b.content if isinstance(b.content, str) else str(b.content)
                        fh.write(f"[tool error] {text[:600]}\n")
            elif isinstance(msg, ResultMessage):
                result = {"subtype": msg.subtype, "cost_usd": getattr(msg, "total_cost_usd", None),
                          "is_error": bool(getattr(msg, "is_error", False)),
                          "turns": getattr(msg, "num_turns", None), "session": getattr(msg, "session_id", None),
                          "result": (getattr(msg, "result", "") or "")[-2000:], "log": str(log)}
                fh.write(f"\n# result\n{result}\n")
    return result


async def _run(d: Dossier, cfg: Config, budget: float, max_turns: int, model: str, verbose: bool) -> dict:
    """`_step`, but a budget or turn limit ends the step instead of raising: everything the agent recorded through
    `residudes` is already in the dossier, so the step is unfinished, not failed, and the next one continues."""
    try:
        return await _step(d, cfg, budget, max_turns, model, verbose)
    except Exception as e:                                 # claude_agent_sdk.ResultError, among others
        err = str(e)
        spent = "budget" in err.lower()
        if not spent and "max turns" not in err.lower():
            raise
        return {"subtype": "budget" if spent else "turns", "cost_usd": budget if spent else None,
                "is_error": False, "turns": max_turns if not spent else None, "session": None, "result": err[:500]}


def step(cfg: Config, target: str, budget: float | None = None, max_turns: int | None = None,
         model: str | None = None, verbose: bool = True) -> dict:
    d = Dossier.load(cfg.targets_dir, target)
    goal, _ = d.next_step()
    if not d.agent_may_act():
        print(f"{target} is at {d.stage}: {goal}. Nothing for the agent to do.")
        return {"subtype": "skipped"}
    lock = Dossier.path(cfg.targets_dir, target).parent / ".step.lock"
    with open(lock, "w") as fh:                              # one agent step per target at a time
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{target}: another agent step is running on this target; not starting a second one")
            return {"subtype": "busy"}
        fh.write(f"{os.getpid()} {dt.datetime.now().isoformat(timespec='seconds')}\n")
        fh.flush()
        res = asyncio.run(_run(d, cfg, budget or cfg.agent_usd_per_step, max_turns or cfg.agent_max_turns, model or cfg.model, verbose))
    with Dossier.locked(cfg.targets_dir, target) as d:      # the agent changed it through `residudes` commands
        d.note("residudes", f"agent step: {res.get('subtype')}, ${res.get('cost_usd')}, turns {res.get('turns')}, log {res.get('log')}")
    if res.get("subtype") in ("budget", "turns"):
        print(f"{target}: the step ran out of {'its USD budget' if res['subtype'] == 'budget' else 'turns'} and "
              f"stopped early. What it recorded through `residudes` stands (`residudes target show {target}`); run another "
              f"step to carry on, or raise the limit for one step with `--budget`.")
    return res
