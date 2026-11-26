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

SESSION = "residudes"          # one tmux session, one window per target: `tmux ls` stays one line
LEAVE = "ctrl-b d"             # tmux's own detach key, not one we invented
SWITCH = "ctrl-b w"            # and its own window chooser, which is the list of targets


def have_tmux() -> bool:
    """tmux is required rather than reimplemented: a terminal application needs resizing,
    scrollback and repainting handled correctly, and a half-written multiplexer gets those
    wrong in ways that show up as a garbled screen."""
    return bool(shutil.which(tmux_bin()))


def tmux_bin() -> str:
    """The tmux that runs the server this process is inside ($TMUX is socket,pid,session), else the one on PATH.
    Versions do not mix: a 3.8 server drops a 3.2a client ("server exited unexpectedly"), and a sixel-capable
    tmux beside an older system one is exactly that pair."""
    pid = os.environ.get("TMUX", "").rsplit(",", 2)[1:2]
    if pid and pid[0].isdigit():
        try:
            exe = os.readlink(f"/proc/{pid[0]}/exe")
        except OSError:
            exe = ""
        if exe and os.access(exe, os.X_OK):
            return exe
    return shutil.which("tmux") or "tmux"


def _no_tmux() -> int:
    print("`residudes chat` needs tmux, which is not on this machine.")
    print(dim("  debian/ubuntu: sudo apt install tmux      rhel/rocky: sudo dnf install tmux"))
    print(dim("  without root:  conda install -c conda-forge tmux"))
    return 1


def _tmux(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([tmux_bin(), *args], capture_output=capture, text=True)


def sessions(cfg=None, prefer: str | None = None) -> list[dict]:
    """The target conversations: one window each, inside the single residudes session."""
    if not have_tmux():
        return []
    r = _tmux("list-windows", "-t", SESSION, "-F",
              "#{window_name}\t#{window_activity}\t#{window_active}")
    if r.returncode != 0:
        return []                                  # no session yet: no conversations, not an error
    out = []
    for line in r.stdout.splitlines():
        name, activity, active = (line.split("\t") + ["", "", ""])[:3]
        if not name:
            continue
        try:                                       # tmux gives the epoch; show it as a local time
            when = dt.datetime.fromtimestamp(int(activity)).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            when = ""
        out.append({"target": name, "name": name, "created": when,
                    "attached": active not in ("", "0")})
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
        auto = bool(d.automation)
        # Who the target waits on, by the same rule the daemon follows (Dossier.agent_may_act), so the list
        # never shows a checkpoint the agent will answer under its mandate as one waiting on a person.
        from residudes import auto as autopilot
        from residudes.daemon import _needs_human
        if d.stage == "PACKAGED":
            waiting = "you (submit)"
        elif _needs_human(d) and not (c and d.agents_turn(c)):
            waiting = "you (NEEDS HUMAN)"
        elif not d.agent_may_act():
            waiting = f"you ({c['id']})" if c else ("you (approve)" if d.stage == "SLATE_PROPOSED" else "")
        elif c:
            waiting = f"agent ({c['id']})"
        else:
            waiting = ""
        s = live.get(ddir.name)
        out.append({"target": ddir.name, "stage": d.stage, "waiting": waiting, "auto": auto,
                    "moves": autopilot.status(cfg, ddir.name), "live": bool(s), "age": _ago(s["created"]) if s else ""})
    return out


def list_cmd(cfg: Config, prefer: str | None = None) -> int:
    rows = _rows(cfg, prefer)
    if not rows:
        print(dim(f"no targets under {cfg.targets_dir}"))
        fasta = Path(__file__).resolve().parents[2] / "examples" / "H1388" / "H1388.fasta"
        print("start one with " + bold(f"residudes target add H1388 --fasta {fasta} --stoich A1B1")
              + dim("   (the shipped example's input)"))
        return 0
    live = sum(r["live"] for r in rows)
    yours = [r["target"] for r in rows if r["waiting"].startswith("you")]
    moving = sum(bool(r["moves"]) and not r["waiting"].startswith("you") for r in rows)
    print()
    print("  " + bold("RESIDUDES") + dim(f" · {len(rows)} targets · {live} in conversation · ")
          + (green(f"{moving} moving on by themselves") if moving else dim("none moving on by themselves")))
    print()
    w = max(8, max(len(r["target"]) for r in rows))
    ww = max(14, max(len(r["waiting"]) for r in rows))
    # pad first, colour second: an escape sequence has width in the format spec but not on screen
    print("  " + dim(f"{'':2}{'target':{w}}  {'stage':16}  {'waiting on':{ww}}  {'moves on':14}  since"))
    print("  " + dim("─" * (w + ww + 56)))
    for r in rows:
        dot = green("●") if r["live"] else dim("·")
        name = bold(f"{r['target']:{w}}")
        stage = f"{r['stage']:16}"
        wait_text = f"{r['waiting'] or '—':{ww}}"
        wait = yellow(wait_text) if r["waiting"].startswith("you") else dim(wait_text)
        print(f"  {dot} {name}  {stage}  {wait}  {_moves(r)}  {dim(r['age'])}")
    print()
    if yours:
        print("  " + yellow("waiting on you: ") + ", ".join(yours))
    reviewed = [r["target"] for r in rows if not r.get("auto")]
    if reviewed:                         # the exception now, so name it: these stop at every checkpoint for a person
        print("  " + dim("with human checkpoints: ") + ", ".join(reviewed))
    print("  " + dim("join: ") + "residudes chat <target>" + dim("    leave: ") + LEAVE
          + dim("    end: ") + "--stop")
    print("  " + dim("▶ asks for each next step when the conversation goes quiet (on with chat)    by hand: ")
          + "residudes chat <target> --no-auto")
    print()
    return 0


def _moves(r: dict) -> str:
    """The 'moves on' cell: ▶ when something will take the next step without being asked, ‖ when that something
    is waiting on a person, ■ when nothing will."""
    if not r["moves"]:
        return dim(f"{'■ manual':14}")
    if r["waiting"].startswith("you"):
        return yellow(f"{'‖ ' + r['moves'] + ', paused':14}")
    return green(f"{'▶ ' + r['moves']:14}")


BRIEF = """You are the RESIDUDES campaign agent, in a conversation about {target} with the person who
started this session. Your memory is the dossier: read it first with `residudes target show {target}`,
and `residudes next {target}` for what the stage machine expects next.

This is a conversation, not a batch step. Answer what they ask, explain what you are doing, and do
the work they agree to - recording it the usual way (`residudes note|evidence|decide|run-record`),
because the dossier is what survives this session. If a decision belongs on the record for a person
to answer, open a checkpoint (`residudes consult`) rather than deciding it here.

You never submit anything. Approving the slate and resolving a checkpoint are the person's too, unless the
target's mandate gives them to you (`residudes automate {target} --show`); the stage machine refuses them from
you otherwise.

Lines that begin with [auto] are residudes keeping the target moving while nobody types: take the step they name
as if the person had asked. A question left only in this conversation gets the next [auto] step as its answer,
so when you need the person, write `residudes note {target} "NEEDS HUMAN: ..."` (or open a checkpoint) and stop;
that pauses auto until they answer.

The person is looking at this tmux window. When the conversation turns on a shape - they ask to see a model, a
site, a template, a comparison, or a structure would answer them better than a sentence - put it in the pane
beside this one, and keep driving it as they talk: `residudes view {target} show M1` (M1-M5 the slate, C0.. a
cluster, tpl_6S9K_A a template, `slate`, or a file; several are superposed), `residudes view {target} do "<molterm
command>"` (color plddt · color chain · focus chain A and resi 7 · show ballstick resi 49+56 · align M3 to M1 automap ·
turn y 90 · zoom chain A), `residudes view {target} snap` to see what they see (read the PNG it names), and
`residudes view {target} close` when they are done. Say what the picture shows; the pane is the evidence, the
sentence is the claim.

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
    from residudes import auto as autopilot
    moves = autopilot.status(cfg, target)
    if moves:
        print(green(f" ▶ moves on by itself ({moves}): the next step is asked for whenever this goes quiet")
              + dim(" · --no-auto to drive it by hand"))
    elif autopilot.by_hand(cfg, target):
        print(dim(f" ■ moves on when asked (set by hand) · `residudes chat {target} --auto` keeps it moving"))
    print(dim(f" {LEAVE} leaves it running · `residudes chat {target} --stop` ends it"))
    print(dim("─" * width))


MCP_CONFIG = {"mcpServers": {"molterm": {"command": "npx", "args": ["-y", "molterm-mcp"]}}}


def write_mcp_config(ddir: Path) -> Path | None:
    """Give the session molterm's tools: fetch, screenshot, select, measure, align, dssp.

    A conversation about a target is mostly about shapes, and an agent that can only describe
    one in prose is working with a hand tied. MCP-aware CLIs read .mcp.json from the directory
    they start in, which is this target's dossier. An existing file is left alone - someone may
    have put their own servers there.
    """
    import json
    path = Path(ddir) / ".mcp.json"
    if path.exists():
        return None
    path.write_text(json.dumps(MCP_CONFIG, indent=1) + "\n")
    return path


def ensure(cfg: Config, target: str, harness: str | None = None) -> bool:
    """The target's window, opened if it is not there. One window per target: the agent that works
    on it and the conversation a person joins are the same agent, with the same context."""
    if not have_tmux():
        return False
    if any(s["target"] == target for s in sessions()):
        return True
    ddir = Path(cfg.targets_dir) / target
    found = available_harness(harness or (cfg.extra.get("agent", {}) or {}).get("cli"))
    if not ddir.is_dir() or not found:
        return False
    _cmd, _name, argv = found
    write_mcp_config(ddir)
    cmdline = argv(BRIEF.format(target=target, ddir=ddir))
    # `env` in front, not tmux's -e: tmux replaces PATH with the PATH of the client that creates the session, so a
    # PATH given with -e never reaches the agent (measured: RESIDUDES_* arrived, PATH did not)
    cmdline = ["env", *_agent_env(cfg), *cmdline]
    if _tmux("has-session", "-t", SESSION).returncode != 0:
        ok = _tmux("new-session", "-d", "-s", SESSION, "-n", target, "-c", str(ddir), *cmdline, capture=False).returncode == 0
    else:
        ok = _tmux("new-window", "-d", "-t", SESSION, "-n", target, "-c", str(ddir), *cmdline, capture=False).returncode == 0
    if ok:                                         # the window's one pane is the agent's; a structure may join it
        _tmux("set-option", "-p", "-t", f"{SESSION}:{target}", AGENT_TAG, "1")
    return ok


AGENT_TAG = "@residudes_agent"


def _agent_env(cfg: Config) -> list[str]:
    """What `residudes agent step` gives its agent, given to the conversation's too: the three checkouts' bin/ on
    PATH, so `residudes ...` (and `residudes view`) resolves however chat was started - `bin/residudes chat` puts
    nothing on PATH - and this config, so a conversation reads the same dossiers the command that opened it did."""
    from residudes.config import RESIDUDES_DIR
    bins = [str(RESIDUDES_DIR / "bin"), str(Path(cfg.residudes_engine) / "bin"), str(Path(cfg.residudes_qa) / "bin")]
    out = ["PATH=" + os.pathsep.join(bins + [os.environ.get("PATH", "")]), "RESIDUDES_ACTOR=agent"]
    return out + ([f"RESIDUDES_CONFIG={cfg.path}"] if getattr(cfg, "path", None) else [])


def agent_pane(target: str) -> str:
    """The conversation's own pane. Addressing the window would reach whichever pane is active, and with a structure
    open beside the conversation (residudes view) that may be molterm: the daemon's next prompt would be typed into
    the viewer. A window opened before the tag existed falls back to its first pane that is not the viewer."""
    r = _tmux("list-panes", "-t", f"{SESSION}:{target}", "-F", "#{pane_id}\t#{" + AGENT_TAG + "}\t#{@residudes_view}")
    rows = [ln.split("\t") for ln in r.stdout.splitlines()] if r.returncode == 0 else []
    pick = next((x[0] for x in rows if len(x) > 1 and x[1]), None) or next(
        (x[0] for x in rows if len(x) < 3 or not x[2]), None)
    return pick or f"{SESSION}:{target}"


def _dress_session() -> None:
    """A status line that is the target list: the window you are in, and the others to page to."""
    for option, value in (("status-left", " #[bold]RESIDUDES#[default] "),
                          ("status-left-length", "16"),
                          ("status-right", f" {SWITCH} targets · {LEAVE} leave "),
                          ("status-right-length", "40"),
                          ("window-status-format", " #W#{?@residudes_auto, ▶,} "),
                          ("window-status-current-format", "#[reverse] #W#{?@residudes_auto, ▶,} #[default]"),
                          ("status-style", "bg=colour236 fg=colour250"),
                          ("base-index", "1")):
        _tmux("set-option", "-t", SESSION, option, value)


def archive_window(cfg: Config, target: str) -> Path | None:
    """Keep what was said. A conversation is the reasoning behind what the dossier records, so
    ending one without saving it throws away the only account of how the target got here."""
    if not any(s["target"] == target for s in sessions()):
        return None
    r = _tmux("capture-pane", "-p", "-S", "-", "-t", agent_pane(target))
    if r.returncode != 0 or not r.stdout.strip():
        return None
    out = Path(cfg.targets_dir) / target / "chat"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (dt.datetime.now().strftime("%Y%m%d-%H%M") + ".log")
    path.write_text(r.stdout)
    return path


def chat_cmd(cfg: Config, target: str | None, harness: str | None, prefer: str | None = None,
             stop: bool = False, archive: bool = False, auto: bool | None = None) -> int:
    if target is None:
        return list_cmd(cfg)
    if not have_tmux():
        return _no_tmux()
    from residudes import auto as autopilot
    where = f"{SESSION}:{target}"
    ddir = Path(cfg.targets_dir) / target
    if auto is False:                              # --no-auto: by hand, remembered; the conversation stays as it is
        print(autopilot.stop(cfg, target, "set to move by hand", remember=True))
        return 0
    if stop:
        autopilot.stop(cfg, target, "the conversation was ended")
    if stop or archive:
        saved = archive_window(cfg, target)
        if saved:
            print(f"{target}: conversation saved to {dim(str(saved))}")
        elif archive:
            print(f"{target}: no conversation to save")
            return 1
        if not stop:
            return 0
        ended = _tmux("kill-window", "-t", where).returncode == 0
        print(f"{target}: conversation ended" if ended else f"{target}: no conversation to end")
        return 0 if ended else 1
    if not ddir.is_dir():
        print(f"no dossier for {target} at {ddir} (create it with `residudes target add {target}`)")
        return 1

    exists = any(s["target"] == target for s in sessions())
    _banner(cfg, target, new=not exists)
    if not exists:
        found = available_harness(harness or (cfg.extra.get("agent", {}) or {}).get("cli"))
        if not found:
            print("no agent CLI on PATH (claude, codex, ag, gemini); `residudes setup` says how to install one")
            return 1
        _cmd, agent_name, _argv = found
        print(f"starting {green(agent_name)} in {dim(str(ddir))}")
        if not ensure(cfg, target, harness):
            print("could not open the window")
            return 1
    # auto is the default: a registered target runs to a packaged slate, and opening its conversation should not
    # stall it. A target set to move by hand (--no-auto) stays so until --auto.
    if auto or (auto is None and not autopilot.by_hand(cfg, target) and not autopilot.running(cfg, target)):
        print(autopilot.start(cfg, target))
    _dress_session()
    _tmux("select-window", "-t", where)            # arrive in this target, not wherever you left
    inside = bool(os.environ.get("TMUX"))          # switch-client inside tmux, attach outside it
    rc = _tmux("switch-client" if inside else "attach-session", "-t", where, capture=False).returncode
    try:                                           # on leaving, say where the target stands now
        d = Dossier.load(cfg.targets_dir, target)
        c = d.open_checkpoint()
        print(f"{bold(target)} is at {d.stage}" + (f", {yellow(c['id'] + ' open')}" if c else "")
              + dim(f" · rejoin with `residudes chat {target}`"))
    except Exception:
        pass
    return rc


def send(target: str, text: str) -> bool:
    """Put a line into the target's conversation, as if a person had typed it.

    This is how the daemon works through a session that is already open: one agent, one context,
    and whoever is attached watches it happen. What it cannot do is report back - tmux has no
    completion signal and no exit code - so the caller decides the step is over by watching the
    dossier, which is where the work is recorded anyway.
    """
    if not have_tmux():
        return False
    if not any(s["target"] == target for s in sessions()):
        return False
    where = agent_pane(target)
    # -l sends the text literally, then Enter separately, so a prompt with punctuation or a
    # leading slash is not read as a tmux key name
    if _tmux("send-keys", "-t", where, "-l", text).returncode != 0:
        return False
    return _tmux("send-keys", "-t", where, "Enter").returncode == 0
