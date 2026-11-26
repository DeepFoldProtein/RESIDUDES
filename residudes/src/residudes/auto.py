"""Auto: run a target to a packaged slate through the conversation a person talks to, without stopping for them.

The conversation's agent does what it is asked and stops when its turn ends, so a target someone is chatting about
only moves when someone asks for the next step. Auto is a small driver process that asks for it: whenever the
conversation has been quiet for a minute (the agent idle, nothing typed) it types the next step into it,
"[auto] Next step for T (STAGE): ...". Same agent, same context, so whoever is attached sees it happen and can
interrupt; the driver waits while they do. The first ask goes the moment auto is turned on.

Auto has no pause, only done. Turning it on grants the mandate (Dossier.authorise_automation, up to PACKAGED) if the
target has none, so the agent answers its own checkpoints and approves the slate on the record, the way a person
would have; a NEEDS HUMAN is answered with "nobody will answer while auto is on: decide it, say why, carry on".
Waiting on the agent or on running jobs is part of running. It is done when the target reaches PACKAGED (submission
is a person's act in every case), or, as a guard on cost, when MAX_IDLE asks pass with nothing new recorded or
MAX_ASKS asks at one stage: then it says so and stops.

`residudes chat T --auto` turns it on; `--no-auto` turns it off, takes back a mandate auto granted, and is
remembered until `--auto`. A plain `residudes chat T` turns it on only for a target that already has a mandate (a
`target add` without --review), so opening a reviewed target's conversation never hands its checkpoints over. The
daemon leaves a target on auto to its conversation, so there is one agent per target.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from residudes import chat
from residudes.config import Config
from residudes.dossier import Dossier

STATE = ".auto.json"
QUIET = 60                  # seconds without output in the window before the next ask
POLL = 10
MAX_IDLE = 5                # asks in a row with nothing new in the dossier: stuck, done
MAX_ASKS = 12               # asks at one stage, whatever was recorded: stuck, done
DONE_AT = ("PACKAGED", "SUBMITTED", "RETROSPECTIVE")
FLAG = "@residudes_auto"    # a window option: the tmux status line marks the target's tab with it
GRANT_NOTE = "granted by turning auto on: it runs the target to PACKAGED without stopping for a person"

_ESC = re.compile(r"\x1b\[[0-9;:?]*[A-Za-z]")
_DIM = re.compile(r"\x1b\[(?:[0-9;]*;)?(?:2|90|38;5;(?:8|24[0-9]|25[0-5]))(?:;[0-9;]*)?m")


def _path(cfg: Config, target: str) -> Path:
    return Path(cfg.targets_dir) / target / STATE


def _state(cfg: Config, target: str) -> dict:
    try:
        st = json.loads(_path(cfg, target).read_text())
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(cfg: Config, target: str, **st) -> None:
    _path(cfg, target).write_text(json.dumps({"since": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), **st}))


def by_hand(cfg: Config, target: str) -> bool:
    """Whether a person set this target to move only when asked (`--no-auto`)."""
    return bool(_state(cfg, target).get("off"))


def done(cfg: Config, target: str) -> str:
    """Why auto finished on this target ('PACKAGED', 'stuck at RUNNING: ...'), or ''."""
    return "" if running(cfg, target) else str(_state(cfg, target).get("done") or "")


def running(cfg: Config, target: str) -> dict | None:
    """The driver's record if its process is alive on this machine, else None."""
    st = _state(cfg, target)
    try:
        cmd = Path(f"/proc/{int(st['pid'])}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return st if b"--drive" in cmd and target.encode() in cmd else None


def _flag(target: str, on: bool) -> None:
    where = f"{chat.SESSION}:{target}"
    if on:
        chat._tmux("set-option", "-w", "-t", where, FLAG, "1")
    else:
        chat._tmux("set-option", "-wu", "-t", where, FLAG)


def _note(cfg: Config, target: str, text: str) -> None:
    with Dossier.locked(cfg.targets_dir, target) as d:
        d.note("residudes", text)


def _log(ddir: Path, target: str, text: str) -> None:
    with open(ddir / "auto.log", "a") as fh:
        fh.write(time.strftime("%H:%M:%S") + f" {target} {text}\n")


def _kill(st: dict | None) -> None:
    if st:
        try:
            os.kill(int(st["pid"]), signal.SIGTERM)
        except (OSError, KeyError, ValueError):
            pass


def start(cfg: Config, target: str, person: str) -> str:
    """Turn auto on: the mandate if the target has none, the driver (a fresh one if one runs), and the first ask."""
    if person == "agent":
        return f"{target}: only a person turns auto on (it hands the agent the target's checkpoints)"
    ddir = Path(cfg.targets_dir) / target
    old = running(cfg, target)
    granted = bool(old and old.get("granted"))
    with Dossier.locked(cfg.targets_dir, target) as d:
        if d.stage in DONE_AT:
            return f"{target}: at {d.stage}, nothing left for auto (submission is a person's act)"
        if not d.automation:
            d.authorise_automation(person, "PACKAGED", GRANT_NOTE)
            granted = True
    _kill(old)                                            # a fresh driver: this code, this mandate
    env = dict(os.environ)
    if getattr(cfg, "path", None):
        env["RESIDUDES_CONFIG"] = str(cfg.path)
    with open(ddir / "auto.log", "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", "residudes", "chat", target, "--drive"], cwd=str(ddir),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    d = Dossier.load(cfg.targets_dir, target)
    what, line = decide(d, pane(target), quiet=0)          # the first ask goes now, if the conversation is free
    asked = d.stage if what == "ask" and chat.send(target, line) else None
    _log(ddir, target, f"asked for the next step at {asked} (auto turned on)" if asked
         else f"turned on; first ask when the conversation is free ({line})")
    _write(cfg, target, pid=proc.pid, asked=asked, granted=granted)
    _flag(target, True)
    if not old:
        _note(cfg, target, "auto on: the conversation is asked for each next step until PACKAGED, without stopping "
                           "for a person" + (" (mandate granted with it)" if granted else ""))
    return (f"{target}: auto on - it runs to PACKAGED without stopping for you"
            + ("; the agent now answers checkpoints itself (mandate granted)" if granted and not old else "")
            + ("; asked for the next step now" if asked else f"; first ask when the conversation is free ({line})"))


def stop(cfg: Config, target: str, why: str = "turned off", remember: bool = False, person: str = "") -> str:
    """End the driver. `remember`: a person chose by hand, so the next `residudes chat T` leaves it off, and a
    mandate that auto granted is taken back with it."""
    st = running(cfg, target)
    revoked = False
    if remember:
        if st and st.get("granted") and person and person != "agent":
            with Dossier.locked(cfg.targets_dir, target) as d:
                if d.automation and d.automation.get("note") == GRANT_NOTE:
                    d.revoke_automation(person, "auto turned off")
                    revoked = True
        _write(cfg, target, off=True)
    elif st or not _state(cfg, target).get("done"):
        _path(cfg, target).unlink(missing_ok=True)
    _flag(target, False)
    if not st:
        return f"{target}: moves on when asked" + (" (remembered: `--auto` turns it back on)" if remember else "")
    _kill(st)
    _note(cfg, target, f"auto off ({why})")
    return (f"{target}: auto off - it moves on when asked" + ("; checkpoints are yours again" if revoked else "")
            + (" (remembered: `--auto` turns it back on)" if remember else ""))


def typed(screen: str) -> str:
    """What a person has typed into the conversation's input line, from `capture-pane -e`: the text after its
    last `❯`. A suggestion Claude Code offers there is drawn dim or grey and is not typing; text in the normal
    colour is, and is left alone (an ask typed after it would be sent with it)."""
    lines = [ln for ln in screen.splitlines() if "❯" in ln]
    if not lines:
        return ""
    after = lines[-1].split("❯", 1)[1]
    m = re.search(r"[^\s ]", _ESC.sub("", after))
    if not m:
        return ""
    lead = after[:after.find(m.group(0))]                 # the escapes in front of the first visible character
    return "" if _DIM.search(lead) else _ESC.sub("", after).strip("  ")


def pane(target: str) -> dict:
    """The conversation as the driver needs it: busy, a dialog open, typed text, seconds since any output."""
    where = chat.agent_pane(target)
    screen = chat._tmux("capture-pane", "-e", "-p", "-t", where).stdout
    plain = _ESC.sub("", screen)
    act = chat._tmux("display", "-p", "-t", where, "#{window_activity}").stdout.strip()
    return {"busy": "esc to interrupt" in plain,
            "dialog": "Enter to confirm" in plain or "Esc to cancel" in plain or "Esc to reject" in plain,
            # the input line drawn: an agent CLI that is still starting has drawn nothing yet, and an ask typed then
            # lands in whatever comes up first (a clean run's first ask went in ahead of the trust-this-folder prompt)
            "ready": "❯" in plain,
            "typed": typed(screen),
            "quiet": time.time() - int(act) if act.isdigit() else 0.0}


def ask_text(d: Dossier) -> str:
    from residudes.daemon import _needs_human
    goal, skill = d.next_step()
    open_question = _needs_human(d)
    text = ("[auto] "
            + ("The NEEDS HUMAN above will not be answered while auto is on. " if open_question else "")
            + f"Next step for {d.target} ({d.stage}): {goal}" + (f" Follow the {skill} skill." if skill else "")
            + " Auto is on: it runs to PACKAGED and nobody answers in between, so decide what is open yourself -"
            + " the option the evidence favours, or the conservative one when it cannot separate them -"
            + " record why in the dossier, and carry on; do not write NEEDS HUMAN.")
    return " ".join(text.split())                        # one line: a newline would send it early


def decide(d: Dossier, p: dict, quiet: float = QUIET) -> tuple[str, str]:
    """('ask', the line to type), ('wait', why) or ('done', why). Never waits on a person: only on the
    conversation (the agent working, a dialog, someone typing, recent output)."""
    if d.stage in DONE_AT:
        return "done", d.stage
    if not d.agent_may_act():                             # the mandate stops short of here
        return "done", f"the mandate ends before {d.stage}"
    if not p.get("ready", True):
        return "wait", "the conversation is still starting"
    if p["busy"] or p["dialog"]:
        return "wait", "the agent is working" if p["busy"] else "a dialog is open"
    if p["typed"]:
        return "wait", "someone is typing"
    if p["quiet"] < quiet:
        return "wait", "recent activity"
    return "ask", ask_text(d)


def status(cfg: Config, target: str) -> str:
    """Who moves the target on: 'auto' (running), 'done' / 'stuck' (auto finished), 'daemon', or ''."""
    if running(cfg, target):
        return "auto"
    fin = _state(cfg, target).get("done")
    if fin:
        return "stuck" if str(fin).startswith("stuck") else "done"
    lock = Path(cfg.targets_dir).parent / ".daemon.lock"
    try:
        words = lock.read_text().split()
        cmd = Path(f"/proc/{int(words[1])}/cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        return ""
    if b"daemon" not in cmd:
        return ""
    covered = words[words.index("targets") + 1].split(",") if "targets" in words else ["all"]
    return "daemon" if "all" in covered or target in covered else ""


def drive(cfg: Config, target: str, quiet: float = QUIET, poll: float = POLL, max_idle: int = MAX_IDLE,
          max_asks: int = MAX_ASKS) -> int:
    """The driver's loop (`residudes chat T --drive`, started by `start`)."""
    from residudes.daemon import WAITS_FOR_JOBS, _waiting_on_jobs
    log = lambda s: print(time.strftime("%H:%M:%S"), target, s, flush=True)   # noqa: E731

    def finish(why: str) -> int:
        st = _state(cfg, target)
        _write(cfg, target, done=why, granted=bool(st.get("granted")))
        _flag(target, False)
        _note(cfg, target, f"auto done: {why}")
        log(f"done: {why}")
        return 0

    log(f"driving (pid {os.getpid()}, quiet {quiet:.0f} s)")
    time.sleep(2)                                         # let `start` write the record this process checks
    st = running(cfg, target) or {}
    asked_at, entries, idle, asks, missing, said = st.get("asked"), None, 0, 0, 0, ""
    if asked_at:
        entries, asks = len(Dossier.load(cfg.targets_dir, target).log), 1
    while True:
        st = running(cfg, target)
        if not st or int(st["pid"]) != os.getpid():
            log("turned off")
            return 0
        if not any(s["target"] == target for s in chat.sessions()):
            missing += 1
            if missing >= 3:                              # the conversation ended (--stop, a reboot): so does this
                _path(cfg, target).unlink(missing_ok=True)
                _note(cfg, target, "auto off (the conversation ended)")
                log("the conversation ended")
                return 0
            time.sleep(poll)
            continue
        missing = 0
        d = Dossier.load(cfg.targets_dir, target)
        if asked_at is not None and d.stage != asked_at:
            idle, asks = 0, 0                             # the last ask moved it on
        if d.stage in WAITS_FOR_JOBS and _waiting_on_jobs(cfg, target):
            what, why = "wait", "jobs running"
        else:
            what, why = decide(d, pane(target), quiet)
        if what == "done":
            return finish(why)
        if what == "ask":
            mine = sum(1 for x in d.log[entries or 0:] if x["actor"] not in ("residudes",)) if entries is not None else 1
            idle = idle + 1 if asked_at == d.stage and not mine else 0
            if idle >= max_idle or asks >= max_asks:
                return finish(f"stuck at {d.stage}: {asks} asks there, the last {idle} with nothing new recorded")
            chat.send(target, why)
            asked_at, entries, asks = d.stage, len(d.log), asks + 1
            log(f"asked for the next step at {d.stage}")
            said = ""
        elif why != said:                                 # log a change of reason, not every poll
            log(f"waiting: {why}")
            said = why
        time.sleep(poll)
