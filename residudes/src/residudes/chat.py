"""`residudes chat`: one long-running conversation per target, in its own tmux session.

Talking to the agent used to mean posting into a checkpoint thread with its id, which works for
a decision that has to be on the record but not for "what are you doing right now". A session
here is an agent CLI running in the target's dossier directory, so the conversation keeps its
context between visits: attach, ask, leave it running, come back.

Sessions are tmux sessions named residudes-<target>, so they survive a closed terminal, and the
same session can be attached from another machine over ssh.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

from residudes.config import Config
from residudes.dossier import Dossier
from residudes.setup import available_harness, bold, dim, green, yellow

PREFIX = "residudes-"


def _tmux(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=capture, text=True)


def sessions() -> list[dict]:
    """The target sessions tmux is holding, newest first."""
    if not shutil.which("tmux"):
        return []
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


def list_cmd(cfg: Config) -> int:
    rows = sessions()
    if not shutil.which("tmux"):
        print("tmux is not installed; `residudes chat <target>` needs it to keep a session running")
        return 1
    if not rows:
        print("no sessions. " + dim("start one with `residudes chat <target>`"))
        return 0
    print(bold("sessions"))
    for s in rows:
        mark = green("attached") if s["attached"] else dim("detached")
        print(f"  {s['target']:12s} {mark:20s} {_state(cfg, s['target'])}")
        print(f"               {dim('since ' + s['created'] + ', ' + s['windows'] + ' window(s)')}")
    print("\n" + dim("attach with `residudes chat <target>`; leave it running with ctrl-b d"))
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


def chat_cmd(cfg: Config, target: str | None, harness: str | None) -> int:
    if target is None:
        return list_cmd(cfg)
    if not shutil.which("tmux"):
        print("tmux is not installed; it is what keeps the conversation running between visits")
        return 1
    name = PREFIX + target
    exists = any(s["name"] == name for s in sessions())
    inside = bool(os.environ.get("TMUX"))
    if not exists:
        found = available_harness(harness)
        if not found:
            print("no agent CLI on PATH (claude, codex, ag, gemini); `residudes chat` needs one")
            return 1
        cmd, agent_name, argv = found
        ddir = Path(cfg.targets_dir) / target
        if not ddir.is_dir():
            print(f"no dossier for {target} at {ddir} (create it with `residudes target add {target}`)")
            return 1
        print(f"starting {green(agent_name)} for {bold(target)} in {dim(str(ddir))}")
        _tmux("new-session", "-d", "-s", name, "-c", str(ddir),
              *argv(BRIEF.format(target=target, ddir=ddir)), capture=False)
    else:
        print(f"attaching to {bold(target)}  {dim('(ctrl-b d to leave it running)')}")
    # attach in place: switch-client inside tmux, attach outside it
    return _tmux("switch-client" if inside else "attach-session", "-t", name, capture=False).returncode
