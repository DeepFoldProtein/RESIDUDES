"""Auto (the default for `residudes chat T`): keep a target moving through the conversation a person talks to.

The conversation's agent does what it is asked and stops when its turn ends, so a target that a person is
chatting about only moves when someone asks for the next step. With --auto, a small driver process watches the
conversation and, whenever it has been quiet for a minute (the agent idle, nothing typed; the first time, a few
seconds, since the person has just turned it on), types the next step into it: "[auto] Next step for T (STAGE): ...". Same agent, same context, so whoever is attached sees it happen and
can answer or interrupt; the driver waits while they do.

It stops asking where the target waits for a person - a checkpoint that is theirs, the slate without a mandate,
PACKAGED, NEEDS HUMAN - and while recorded jobs are still running, and resumes once that is answered or done.
After MAX_IDLE asks with no stage change it writes NEEDS HUMAN, as the daemon does after idle steps.

`residudes chat T` turns it on unless the target was set to move by hand (`--no-auto`, remembered in the dossier
directory until `--auto`); ending the conversation (`--stop`) ends the driver too. The daemon leaves a target on
auto to its conversation, so there is one agent per target.
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
QUIET = 60                  # seconds without output in the window before it counts as quiet
FIRST = 5                   # ... for the first ask: the person just turned auto on, so it starts at once
POLL = 10
MAX_IDLE = 5                # asks at one stage before it stops and asks for a person
PERSONS = ("PACKAGED", "SUBMITTED", "RETROSPECTIVE")
FLAG = "@residudes_auto"    # a window option: the tmux status line marks the target's tab with it

_ESC = re.compile(r"\x1b\[[0-9;:?]*[A-Za-z]")
_DIM = re.compile(r"\x1b\[(?:[0-9;]*;)?(?:2|90|38;5;(?:8|24[0-9]|25[0-5]))(?:;[0-9;]*)?m")


def _path(cfg: Config, target: str) -> Path:
    return Path(cfg.targets_dir) / target / STATE


def by_hand(cfg: Config, target: str) -> bool:
    """Whether a person set this target to move only when asked (`--no-auto`)."""
    try:
        return bool(json.loads(_path(cfg, target).read_text()).get("off"))
    except (OSError, ValueError, AttributeError):
        return False


def running(cfg: Config, target: str) -> dict | None:
    """The driver's record if its process is alive on this machine, else None."""
    try:
        st = json.loads(_path(cfg, target).read_text())
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


def start(cfg: Config, target: str) -> str:
    st = running(cfg, target)
    if st:
        return f"{target}: auto is already on (since {st.get('since', '?')})"
    ddir = Path(cfg.targets_dir) / target
    env = dict(os.environ)
    if getattr(cfg, "path", None):
        env["RESIDUDES_CONFIG"] = str(cfg.path)
    with open(ddir / "auto.log", "a") as log:
        proc = subprocess.Popen([sys.executable, "-m", "residudes", "chat", target, "--drive"], cwd=str(ddir),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    since = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    _path(cfg, target).write_text(json.dumps({"pid": proc.pid, "since": since}))
    _flag(target, True)
    _note(cfg, target, f"auto on: the conversation is asked for the next step whenever it is quiet for {QUIET} s "
                       "(residudes chat --no-auto stops it)")
    return f"{target}: auto on - the conversation will be asked for each next step"


def stop(cfg: Config, target: str, why: str = "turned off", remember: bool = False) -> str:
    """End the driver. `remember`: the person chose by hand, so the next `residudes chat T` does not turn it on."""
    st = running(cfg, target)
    if remember:
        _path(cfg, target).write_text(json.dumps({"off": True, "since": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}))
    else:
        _path(cfg, target).unlink(missing_ok=True)
    _flag(target, False)
    if not st:
        return f"{target}: moves on when asked" + (" (remembered: `--auto` turns it back on)" if remember else "")
    try:
        os.kill(int(st["pid"]), signal.SIGTERM)
    except OSError:
        pass
    _note(cfg, target, f"auto off ({why})")
    return f"{target}: auto off - it moves on when asked" + (" (remembered: `--auto` turns it back on)" if remember else "")


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
            "dialog": "Enter to confirm" in plain or "Esc to cancel" in plain,
            "typed": typed(screen),
            "quiet": time.time() - int(act) if act.isdigit() else 0.0}


def ask_text(d: Dossier) -> str:
    goal, skill = d.next_step()
    text = (f"[auto] Next step for {d.target} ({d.stage}): {goal}" + (f" Follow the {skill} skill." if skill else "")
            + " Record what you find and decide in the dossier as usual; if only a person can decide, write "
            + f'`residudes note {d.target} "NEEDS HUMAN: ..."` and stop.')
    return " ".join(text.split())                        # one line: a newline would send it early


def decide(d: Dossier, p: dict, quiet: float = QUIET) -> tuple[str, str]:
    """('ask', the line to type) or ('wait', why). The dossier's rule first (Dossier.agent_may_act, NEEDS HUMAN, the
    stages a person owns), then the conversation's: never over the agent, a dialog or a person's typing."""
    from residudes.daemon import _needs_human
    if d.stage in PERSONS:
        return "wait", f"{d.stage} is a person's step"
    if not d.agent_may_act():
        return "wait", "waiting on you"
    c = d.open_checkpoint()
    if _needs_human(d) and not (c and d.agents_turn(c)):
        return "wait", "NEEDS HUMAN"
    if p["busy"] or p["dialog"]:
        return "wait", "the agent is working" if p["busy"] else "a dialog is open"
    if p["typed"]:
        return "wait", "someone is typing"
    if p["quiet"] < quiet:
        return "wait", "recent activity"
    return "ask", ask_text(d)


def status(cfg: Config, target: str) -> str:
    """What the target list shows: who moves the target on - this driver, the daemon - or nobody."""
    if running(cfg, target):
        return "auto"
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


def drive(cfg: Config, target: str, quiet: float = QUIET, poll: float = POLL, max_idle: int = MAX_IDLE) -> int:
    """The driver's loop (`residudes chat T --drive`, started by --auto)."""
    from residudes.daemon import WAITS_FOR_JOBS, _waiting_on_jobs
    log = lambda s: print(time.strftime("%H:%M:%S"), target, s, flush=True)   # noqa: E731
    log(f"driving (pid {os.getpid()}, quiet {quiet:.0f} s)")
    asked_at, idle, missing, said, first = None, 0, 0, "", True
    time.sleep(2)                                         # let `start` write the record this process checks
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
            idle = 0                                      # the last ask moved it on
        if d.stage in WAITS_FOR_JOBS and _waiting_on_jobs(cfg, target):
            what, why = "wait", "jobs running"
        else:
            what, why = decide(d, pane(target), min(quiet, FIRST) if first else quiet)
        if what == "ask":
            if asked_at == d.stage:
                idle += 1
            if idle >= max_idle:
                _note(cfg, target, f"NEEDS HUMAN: auto asked {max_idle} times without stage progress at {d.stage}")
                idle, asked_at = 0, None
                log(f"stopped asking: {max_idle} asks at {d.stage}")
            else:
                chat.send(target, why)
                asked_at, first = d.stage, False
                log(f"asked for the next step at {d.stage}")
                said = ""
        elif why != said:                                 # log a change of reason, not every poll
            log(f"waiting: {why}")
            said = why
        time.sleep(poll)
