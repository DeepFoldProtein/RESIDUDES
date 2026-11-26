"""Guardrails for every shell command the agent runs (used as a PreToolUse hook on Bash).

Each rule comes from an incident on the DeepFold cluster during CASP17 or from the approval policy:
  broad find/du on shared storage      a `find /` walked NFS mounts for hours; GPFS metadata storms
  recursive deletes                     deletions are never undone; move to a trash directory instead
  mass scancel / pueue clean|reset      kills other people's work or wipes the queue state
  force push                            rewrites shared history
  outward submission                    CASP submissions need a person's `residudes approve`
  approval / identity                   the agent never changes RESIDUDES_ACTOR, and never grants itself
                                        the automation mandate (`residudes automate --on`)
  per-job watcher loops                 2,131 watchers once froze GPFS; use one scheduler instead

This is a seatbelt, not a sandbox: the agent runs as the same Unix user as the person, and a determined
command (or a script the agent writes) can get past any pattern. The Write/Edit hook confines file tools to the
dossier directory and keeps dossier.json behind the `residudes` CLI; shell writes are not confined.
"""
from __future__ import annotations

import os
import re
import shlex

# Roots that belong to everyone on a cluster: a recursive delete or a full scan here hits other
# people's work. A site can add its own with RESIDUDES_SHARED_ROOTS (colon separated).
SHARED_ROOTS = {"/", "/gpfs", "/store", "/data", "/home", "/scratch", "/work", "/project"} | {
    d for d in os.environ.get("RESIDUDES_SHARED_ROOTS", "").split(":") if d}
WRAPPERS = {"sudo", "env", "nohup", "time", "nice", "timeout", "xargs", "command", "exec", "stdbuf", "srun"}
WRAPPER_OPTS_WITH_VALUE = {"-u", "-s", "-k", "-n", "-C", "-S", "-I", "-P"}
GLOB = re.compile(r"[*?\[]")
RM_WHY = "recursive delete is not allowed; move the directory to a trash folder (mv) and tell the user"

RULES: list[tuple[str, re.Pattern, str]] = [
    ("scancel-mass", re.compile(r"\bscancel\s+(.*\s)?(-u|--user|--state|-t)\b"),
     "mass scancel is not allowed; cancel your own job ids one by one"),
    ("pueue-wipe", re.compile(r"\bpueue\s+(clean|reset|kill\s+(-a|--all))\b"),
     "pueue clean/reset/kill --all wipes shared queue state"),
    ("force-push", re.compile(r"\bgit\s+push\b.*\s(-f|--force|--force-with-lease)\b"),
     "force push is not allowed"),
    ("casp-submit", re.compile(r"\b(casp17\s+submit|residudes\s+submit|casp17_utils(\.cli)?\s+submit|casp17_utils\.net)\b"
                               r"|predictioncenter\.org"),
     "submitting to CASP is an outward action: a person runs it after `residudes approve`"),
    ("actor", re.compile(r"\bRESIDUDES_ACTOR\s*=|\b(unset|export)\s+RESIDUDES_ACTOR\b|-u\s*RESIDUDES_ACTOR\b"),
     "the agent does not change RESIDUDES_ACTOR"),
    # a mandate the agent could grant itself would not be a mandate
    ("mandate", re.compile(r"\bresidudes\s+automate\b"),
     "only a person turns automation on or off (`residudes automate <target> --on`)"),
    ("approve", re.compile(r"\bresidudes(\.cli)?\b.*\b(approve|resolve)\b"),
     "`residudes approve` and `residudes resolve` are for people only"),
    ("rmtree", re.compile(r"\b(rmtree|removedirs)\b"), RM_WHY),
]


def _segments(cmd: str) -> list[list[str]]:
    """Split a shell line into simple commands (on ; & | newlines and command substitution), tokenise each,
    and drop leading VAR=value assignments and wrappers (sudo, env, timeout, xargs, ...)."""
    out = []
    for part in re.split(r"[;&|\n]+|\$\(|`|\)", cmd):
        part = part.strip()
        if not part:
            continue
        try:
            toks = shlex.split(part, posix=True)
        except ValueError:
            toks = part.split()
        while toks:
            head = os.path.basename(toks[0])
            if "=" in toks[0] and not toks[0].startswith("-"):
                toks = toks[1:]
                continue
            if head in WRAPPERS:
                toks = toks[1:]
                while toks and toks[0].startswith("-"):
                    opt = toks.pop(0)
                    if opt in WRAPPER_OPTS_WITH_VALUE and toks:
                        toks.pop(0)
                if head == "timeout" and toks and re.fullmatch(r"[0-9.]+[smhd]?", toks[0]):
                    toks = toks[1:]
                continue
            break
        if toks:
            out.append(toks)
    return out


def _shared(arg: str) -> bool:
    """A path argument that names a shared root, or globs directly under one (/gpfs/*)."""
    if not arg.startswith("/"):
        return False
    m = GLOB.search(arg)
    if m:
        arg = arg[:m.start()].rsplit("/", 1)[0] or "/"
    return arg == "/" or arg.rstrip("/") in SHARED_ROOTS


def _rm(toks: list[str]) -> str | None:
    for t in toks[1:]:
        if t == "--":
            break
        if t == "--recursive" or (t.startswith("-") and not t.startswith("--") and re.search(r"[rR]", t)):
            return RM_WHY
    return None


def _scan(prog: str, toks: list[str]) -> str | None:
    args = toks[1:]
    if prog == "find":
        if "-delete" in args or any(a in ("-exec", "-execdir", "-ok") and i + 1 < len(args) and os.path.basename(args[i + 1]) == "rm"
                                    for i, a in enumerate(args)):
            return "find -delete / -exec rm is a recursive delete; move files to a trash folder instead"
        paths = []
        for a in args:                                     # find's paths come before the first expression token
            if a.startswith(("-", "(", "!")):
                break
            paths.append(a)
    elif prog == "ls":
        if not any((a.startswith("-") and not a.startswith("--") and "R" in a) or a == "--recursive" for a in args):
            return None
        paths = [a for a in args if not a.startswith("-")]
    else:                                                  # du, bfs, fd, tree
        paths, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a in ("-d", "--max-depth", "-B", "--block-size", "-t", "--threshold", "--exclude", "-X", "-L"):
                skip = True
                continue
            if a.startswith("-"):
                continue
            paths.append(a)
    for p in paths:
        if _shared(p):
            return f"`{prog} {p}` scans shared storage; scope it to the working directory"
    return None


def _watch(cmd: str) -> str | None:
    if re.search(r"(^|[;&|\s])watch\s", cmd):
        m = re.search(r"\bwatch\b.*?(-n|--interval)\s*=?\s*([0-9.]+)", cmd)
        if not m or float(m.group(2)) < 60:
            return "`watch` faster than a minute overloads the cluster; ask the scheduler or poll >= 60 s"
    if re.search(r"\b(while|until)\b", cmd) and re.search(r"\b(squeue|sacct|pueue|ls|stat|nvidia-smi|tail)\b", cmd):
        sleeps = [int(n) * {"m": 60, "h": 3600}.get(u, 1) for n, u in re.findall(r"\bsleep\s+([0-9]+)([smh]?)\b", cmd)]
        if not sleeps or min(sleeps) < 60:
            return "polling loops faster than a minute overload the cluster; ask the scheduler or poll >= 60 s"
    return None


def check(cmd: str, mandate_for: str | None = None) -> str | None:
    """Return a reason to block the command, or None to let it run.

    `mandate_for` names the target whose automation mandate covers the current step. The approve/resolve rule
    is a seatbelt for a target without one, where the CLI would refuse anyway; for that target, under its
    mandate, approving the slate and resolving a checkpoint is the job, and blocking it here stopped full
    automation at the first checkpoint (H1388: the agent answered C1, `resolve` was refused, NEEDS HUMAN)."""
    for name, rx, why in RULES:
        if rx.search(cmd):
            if name == "approve" and mandate_for and re.search(rf"\b{re.escape(mandate_for)}\b", cmd):
                continue
            if name == "approve" and re.search(r"(^|\s)(-h|--help)(\s|$)", cmd):
                continue                       # usage only: refusing it left M1370's agent unable to see the flags
            return why
    for toks in _segments(cmd):
        prog = os.path.basename(toks[0])
        if prog == "rm" and (why := _rm(toks)):
            return why
        if prog in ("find", "bfs", "fd", "du", "ls", "tree") and (why := _scan(prog, toks)):
            return why
        if prog in ("pymol", "pymol2") or (prog.startswith("python") and "pymol" in toks[1:3]):
            return "rendering runs through sbatch: `residudes render T --pml FILE` (never PyMOL on the login node)"
    return _watch(cmd)


def hook(mandate_for: str | None = None):
    """A Claude Agent SDK PreToolUse hook for Bash, for one step on one target."""
    async def pre_tool_use(input_data: dict, tool_use_id, context) -> dict:
        if input_data.get("tool_name") != "Bash":
            return {}
        if input_data.get("tool_input", {}).get("run_in_background"):
            # A step is one unattended agent run: nobody is left to hear a background watcher report, and while it
            # polls, the step's process stays alive and holds the daemon - every other target - until the job ends
            # (M1370's step "picked this up when the watcher reports" and the daemon sat behind a queued QA job).
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason": "residudes guard: no background commands in a step. Submit the job, "
                    "record it (`residudes run-record` or the command's own record), and end the step: the daemon "
                    "refreshes the jobs and runs the next step when they finish."}}
        why = check(str(input_data.get("tool_input", {}).get("command", "")), mandate_for)
        if not why:
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": f"residudes guard: {why}"}}
    return pre_tool_use


pre_tool_use = hook()                                   # no mandate: the seatbelt stays on
