"""`residudes chat`: one long-running conversation per target, in its own tmux session.

Talking to the agent used to mean posting into a checkpoint thread with its id, which works for
a decision that has to be on the record but not for "what are you doing right now". A session
here is an agent CLI running in the target's dossier directory, so the conversation keeps its
context between visits: attach, ask, leave it running, come back.

Persistence comes from tmux: a session survives a closed terminal and can be re-attached over ssh
from another machine. tmux is required rather than reimplemented - see have_tmux().
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
LEAVE = "ctrl-b d"             # tmux's own detach key, not one we invented


def have_tmux() -> bool:
    """tmux is required rather than reimplemented: a terminal application needs resizing,
    scrollback and repainting handled correctly, and a half-written multiplexer gets those
    wrong in ways that show up as a garbled screen."""
    return bool(shutil.which("tmux"))


def _no_tmux() -> int:
    print("`residudes chat` needs tmux, which is not on this machine.")
    print(dim("  debian/ubuntu: sudo apt install tmux      rhel/rocky: sudo dnf install tmux"))
    print(dim("  without root:  conda install -c conda-forge tmux"))
    return 1


def _tmux(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=capture, text=True)


def sessions(cfg=None, prefer: str | None = None) -> list[dict]:
    """The target conversations tmux is holding."""
    if not have_tmux():
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


def _ago(stamp: str) -> str:
    """'2026-11-25 22:07' -> '18m', '2h', '3d'; a conversation's age is more use than its date."""
    try:
        then = dt.datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return ""
    mins = int((dt.datetime.now() - then).total_seconds() // 60)
    if mins < 60:
        return f"{max(mins, 0)}m"
    if mins < 60 * 48:
        return f"{mins // 60}h"
    return f"{mins // 1440}d"


def _rows(cfg: Config, prefer: str | None) -> list[dict]:
    """Every target with a dossier, and whether a conversation is running for it."""
    live = {s["target"]: s for s in sessions(cfg, prefer)}
    root = Path(cfg.targets_dir)
    out = []
    for ddir in sorted(root.iterdir()) if root.is_dir() else []:
        if not (ddir / "dossier.json").exists():
            continue
        try:
            d = Dossier.load(cfg.targets_dir, ddir.name)
        except Exception:
            continue
        c = d.open_checkpoint()
        waiting = ""
        if c and not d.agents_turn(c):
            waiting = f"you ({c['id']})"
        elif c:
            waiting = f"agent ({c['id']})"
        s = live.get(ddir.name)
        out.append({"target": ddir.name, "stage": d.stage, "waiting": waiting,
                    "live": bool(s), "age": _ago(s["created"]) if s else ""})
    return out


def list_cmd(cfg: Config, prefer: str | None = None) -> int:
    rows = _rows(cfg, prefer)
    if not rows:
        print(dim(f"no targets under {cfg.targets_dir}"))
        print("start one with " + bold("residudes target add T1234 --fasta T1234.fasta"))
        return 0
    live = sum(r["live"] for r in rows)
    yours = [r["target"] for r in rows if r["waiting"].startswith("you")]
    print()
    print("  " + bold("RESIDUDES") + dim(f" · {len(rows)} targets · {live} in conversation"))
    print()
    w = max(8, max(len(r["target"]) for r in rows))
    # pad first, colour second: an escape sequence has width in the format spec but not on screen
    print("  " + dim(f"{'':2}{'target':{w}}  {'stage':16}  {'waiting on':14}  since"))
    print("  " + dim("─" * (w + 42)))
    for r in rows:
        dot = green("●") if r["live"] else dim("·")
        name = bold(f"{r['target']:{w}}")
        stage = f"{r['stage']:16}"
        wait_text = f"{r['waiting'] or '—':14}"
        wait = yellow(wait_text) if r["waiting"].startswith("you") else dim(wait_text)
        print(f"  {dot} {name}  {stage}  {wait}  {dim(r['age'])}")
    print()
    if yours:
        print("  " + yellow("waiting on you: ") + ", ".join(yours))
    print("  " + dim("join: ") + "residudes chat <target>" + dim("    leave: ") + LEAVE
          + dim("    end: ") + "--stop")
    print()
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


def _banner(cfg: Config, target: str, new: bool) -> None:
    """What someone needs on arriving: which target, where it stands, what it is waiting for."""
    try:
        d = Dossier.load(cfg.targets_dir, target)
    except Exception:
        return
    nxt, skill = d.next_step()
    c = d.open_checkpoint()
    width = 78
    print(dim("─" * width))
    head = f" {bold(target)}  {d.stage}"
    if c:
        head += "  " + (yellow(f"{c['id']} waiting on you") if not d.agents_turn(c)
                        else dim(f"{c['id']} with the agent"))
    print(head + ("   " + dim("new conversation") if new else ""))
    if nxt:
        first = nxt.split(":")[0] if ":" in nxt[:60] else nxt
        print(dim(f" next: {first[:width - 8]}"))
    print(dim(f" {LEAVE} leaves it running · `residudes chat {target} --stop` ends it"))
    print(dim("─" * width))


def chat_cmd(cfg: Config, target: str | None, harness: str | None, prefer: str | None = None,
             stop: bool = False) -> int:
    if target is None:
        return list_cmd(cfg)
    if not have_tmux():
        return _no_tmux()
    name = PREFIX + target
    ddir = Path(cfg.targets_dir) / target
    if stop:
        ended = _tmux("kill-session", "-t", name).returncode == 0
        print(f"{target}: conversation ended" if ended else f"{target}: no conversation to end")
        return 0 if ended else 1
    if not ddir.is_dir():
        print(f"no dossier for {target} at {ddir} (create it with `residudes target add {target}`)")
        return 1

    exists = any(s["name"] == name for s in sessions())
    _banner(cfg, target, new=not exists)
    if not exists:
        found = available_harness(harness)
        if not found:
            print("no agent CLI on PATH (claude, codex, ag, gemini); `residudes chat` needs one")
            return 1
        _cmd, agent_name, argv = found
        print(f"starting {green(agent_name)} in {dim(str(ddir))}")
        _tmux("new-session", "-d", "-s", name, "-c", str(ddir),
              *argv(BRIEF.format(target=target, ddir=ddir)), capture=False)
    inside = bool(os.environ.get("TMUX"))          # switch-client inside tmux, attach outside it
    rc = _tmux("switch-client" if inside else "attach-session", "-t", name, capture=False).returncode
    try:                                           # on leaving, say where the target stands now
        d = Dossier.load(cfg.targets_dir, target)
        c = d.open_checkpoint()
        print(f"{bold(target)} is at {d.stage}" + (f", {yellow(c['id'] + ' open')}" if c else "")
              + dim(f" · rejoin with `residudes chat {target}`"))
    except Exception:
        pass
    return rc
