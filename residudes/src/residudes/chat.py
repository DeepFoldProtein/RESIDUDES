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
        _banner(cfg, target, new=session.running(ddir) is None)
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
        rc = session.attach(ddir)
        try:                                   # on leaving, say where the target stands now
            d = Dossier.load(cfg.targets_dir, target)
            print(f"{bold(target)} is at {d.stage}" +
                  (f", {yellow(d.open_checkpoint()['id'] + ' open')}" if d.open_checkpoint() else ""))
        except Exception:
            pass
        return rc

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
