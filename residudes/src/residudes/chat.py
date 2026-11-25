"""`residudes chat`: one long-running conversation per target, in its own tmux session.

Talking to the agent used to mean posting into a checkpoint thread with its id, which works for
a decision that has to be on the record but not for "what are you doing right now". A session
here is an agent CLI running in the target's dossier directory, so the conversation keeps its
context between visits: attach, ask, leave it running, come back.

Persistence comes from a terminal multiplexer when the machine has one - tmux, else screen - so a
session survives a closed terminal and can be re-attached over ssh from somewhere else. Neither is
required: with no multiplexer the conversation runs in the foreground instead, which costs the
detaching rather than the conversation.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

from residudes.config import Config
from residudes import session
from residudes.dossier import Dossier
from residudes.setup import available_harness, bold, dim, green, yellow

PREFIX = "residudes-"
LEAVE = "ctrl-\\"             # what detaches from one of our own sessions


def backend(prefer: str | None = None) -> str:
    """Which session mechanism to use: our own by default, a multiplexer when asked for one."""
    if prefer in ("tmux", "screen") and shutil.which(prefer):
        return prefer
    if prefer == "builtin" or prefer is None:
        return "builtin"
    return "builtin"


def _tmux(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=capture, text=True)


def _screen_sessions() -> list[dict]:
    r = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():            # "\t12345.residudes-T1234\t(Detached)"
        parts = line.strip().split("\t")
        if not parts or "." not in parts[0]:
            continue
        name = parts[0].split(".", 1)[1]
        if name.startswith(PREFIX):
            out.append({"target": name[len(PREFIX):], "name": name, "created": "",
                        "attached": "Attached" in line, "windows": "1"})
    return out


def builtin_sessions(cfg) -> list[dict]:
    """Our own sessions: one directory per target under the work root."""
    out = []
    root = Path(cfg.targets_dir)
    if not root.is_dir():
        return out
    for ddir in sorted(root.iterdir()):
        pid = session.running(ddir) if ddir.is_dir() else None
        if pid:
            p = session.paths(ddir)
            started = p["started"].read_text().strip() if p["started"].exists() else ""
            out.append({"target": ddir.name, "name": ddir.name, "created": started,
                        "attached": False, "windows": "1", "pid": pid})
    return out


def sessions(cfg=None, prefer: str | None = None) -> list[dict]:
    """The target sessions running on this machine."""
    mux = backend(prefer)
    if mux == "builtin":
        return builtin_sessions(cfg) if cfg else []
    if mux == "screen":
        return _screen_sessions()
    r = _tmux("list-sessions", "-F", "#{session_name}\t#{session_created}\t#{session_attached}\t#{session_windows}")
    if r.returncode != 0:
        return []                                  # no server running: no sessions, not an error
    out = []
    for line in r.stdout.splitlines():
        name, created, attached, windows = (line.split("\t") + ["", "", "", ""])[:4]
        if name.startswith(PREFIX):
            try:                                   # tmux gives the epoch; show it as a local time
                when = dt.datetime.fromtimestamp(int(created)).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                when = created
            out.append({"target": name[len(PREFIX):], "name": name, "created": when,
                        "attached": attached not in ("", "0"), "windows": windows})
    return out


def _state(cfg: Config, target: str) -> str:
    try:
        d = Dossier.load(cfg.targets_dir, target)
    except Exception:                               # a session for a target with no dossier yet
        return dim("no dossier")
    c = d.open_checkpoint()
    bits = [d.stage]
    if c:
        bits.append(yellow(f"{c['id']} open" + (" — your turn" if d.agents_turn(c) is False else " — agent's turn")))
    return " · ".join(bits)


def list_cmd(cfg: Config, prefer: str | None = None) -> int:
    rows = sessions(cfg, prefer)
    if not rows:
        print("no conversations running. " + dim("start one with `residudes chat <target>`"))
        return 0
    print(bold("conversations"))
    for s in rows:
        where = green("attached") if s.get("attached") else dim("running")
        print(f"  {s['target']:12s} {where:20s} {_state(cfg, s['target'])}")
        if s.get("created"):
            print(f"               {dim('since ' + s['created'])}")
    print("\n" + dim("join one with `residudes chat <target>`; " + LEAVE + " leaves it running"))
    return 0


BRIEF = """You are the RESIDUDES campaign agent, in a conversation about {target} with the person who
started this session. Your memory is the dossier: read it first with `residudes target show {target}`,
and `residudes next {target}` for what the stage machine expects next.

This is a conversation, not a batch step. Answer what they ask, explain what you are doing, and do
the work they agree to - recording it the usual way (`residudes note|evidence|decide|run-record`),
because the dossier is what survives this session. If a decision belongs on the record for a person
to answer, open a checkpoint (`residudes consult`) rather than deciding it here.

You never approve a slate, resolve a checkpoint or submit anything: those are the person's, and the
stage machine refuses them from you.

Dossier directory (your working directory): {ddir}
"""


def chat_cmd(cfg: Config, target: str | None, harness: str | None, prefer: str | None = None,
             stop: bool = False) -> int:
    if target is None:
        return list_cmd(cfg, prefer)
    mux = backend(prefer)
    ddir = Path(cfg.targets_dir) / target
    if stop:
        if mux == "builtin":
            ok_ = session.stop(ddir) == 0
        else:
            ok_ = _tmux("kill-session", "-t", PREFIX + target).returncode == 0
        print(f"{target}: conversation ended" if ok_ else f"{target}: no conversation to end")
        return 0 if ok_ else 1
    if not ddir.is_dir():
        print(f"no dossier for {target} at {ddir} (create it with `residudes target add {target}`)")
        return 1

    if mux == "builtin":
        if session.running(ddir) is None:
            found = available_harness(harness)
            if not found:
                print("no agent CLI on PATH (claude, codex, ag, gemini); `residudes chat` needs one")
                return 1
            cmd, agent_name, argv = found
            print(f"starting {green(agent_name)} for {bold(target)} {dim('(' + LEAVE + ' leaves it running)')}")
            session.start(ddir, argv(BRIEF.format(target=target, ddir=ddir)))
        else:
            print(f"joining {bold(target)}  {dim('(' + LEAVE + ' leaves it running)')}")
        return session.attach(ddir)

    name = PREFIX + target
    exists = any(s["name"] == name for s in sessions(cfg, prefer))
    inside = bool(os.environ.get("TMUX"))
    if not exists:
        found = available_harness(harness)
        if not found:
            print("no agent CLI on PATH (claude, codex, ag, gemini); `residudes chat` needs one")
            return 1
        cmd, agent_name, argv = found
        print(f"starting {green(agent_name)} for {bold(target)} in {dim(str(ddir))}")
        cmdline = argv(BRIEF.format(target=target, ddir=ddir))
        if mux == "tmux":
            _tmux("new-session", "-d", "-s", name, "-c", str(ddir), *cmdline, capture=False)
        else:
            subprocess.run(["screen", "-dmS", name, *cmdline], cwd=str(ddir))
    else:
        leave = "ctrl-b d" if mux == "tmux" else "ctrl-a d"
        print(f"attaching to {bold(target)}  {dim('(' + leave + ' to leave it running)')}")
    if mux == "tmux":
        return _tmux("switch-client" if inside else "attach-session", "-t", name, capture=False).returncode
    return subprocess.call(["screen", "-r", name])
