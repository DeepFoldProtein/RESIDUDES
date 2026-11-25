"""Background agent sessions without a terminal multiplexer.

`residudes chat T` should be the whole interface, so this holds a conversation open by itself:
the agent runs under a pty in a detached process, everything it prints is appended to a log, and
attaching means replaying that log and forwarding the keyboard into the pty. Several people can
attach at once, and leaving (ctrl-\\) stops the attachment, not the agent.

tmux does this better and is used when it is there (see chat.py); this exists so that a machine
without tmux or screen is not a machine without sessions.
"""
from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

DIR = ".residudes-session"          # inside the target's dossier directory


def paths(ddir: Path) -> dict:
    d = Path(ddir) / DIR
    return {"dir": d, "pid": d / "pid", "log": d / "log", "fifo": d / "input", "started": d / "started"}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    return True


def running(ddir: Path) -> int | None:
    p = paths(ddir)
    try:
        pid = int(p["pid"].read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


def start(ddir: Path, argv: list[str]) -> int:
    """Run argv under a pty in a detached process; return its pid."""
    p = paths(ddir)
    p["dir"].mkdir(parents=True, exist_ok=True)
    for f in ("log", "fifo", "started"):
        p[f].unlink(missing_ok=True)
    os.mkfifo(p["fifo"])

    pid = os.fork()
    if pid:                                     # the caller waits for the pid file, then attaches
        for _ in range(100):
            if p["pid"].exists():
                break
            time.sleep(0.05)
        return int(p["pid"].read_text().strip())

    os.setsid()                                 # detach from the terminal that started us
    child, fd = pty.fork()
    if child == 0:
        os.environ["TERM"] = os.environ.get("TERM", "xterm-256color")
        os.chdir(ddir)
        os.execvp(argv[0], argv)
        os._exit(127)
    p["pid"].write_text(f"{os.getpid()}\n")
    p["started"].write_text(time.strftime("%Y-%m-%d %H:%M") + "\n")
    log = open(p["log"], "ab", buffering=0)
    fifo = os.open(p["fifo"], os.O_RDONLY | os.O_NONBLOCK)
    try:
        while True:
            r, _, _ = select.select([fd, fifo], [], [], 1.0)
            if fd in r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                log.write(data)
            if fifo in r:
                data = os.read(fifo, 65536)
                if data:
                    os.write(fd, data)
                else:                            # every writer left; reopen and keep waiting
                    os.close(fifo)
                    fifo = os.open(p["fifo"], os.O_RDONLY | os.O_NONBLOCK)
    finally:
        log.close()
        p["pid"].unlink(missing_ok=True)
        os._exit(0)


def _resize(fd: int) -> None:
    try:
        size = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    except OSError:
        pass


def attach(ddir: Path, detach_key: bytes = b"\x1c") -> int:
    """Replay the log, then stream it while forwarding the keyboard. ctrl-\\ detaches."""
    p = paths(ddir)
    if running(ddir) is None:
        print("that session is not running")
        return 1
    log = open(p["log"], "rb")
    sys.stdout.buffer.write(log.read())          # what was said before you arrived
    sys.stdout.buffer.flush()
    fifo = os.open(p["fifo"], os.O_WRONLY | os.O_NONBLOCK)
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if old:
        tty.setraw(sys.stdin.fileno())
    try:
        while running(ddir) is not None:
            r, _, _ = select.select([sys.stdin, log], [], [], 0.2)
            if sys.stdin in r:
                data = os.read(sys.stdin.fileno(), 4096)
                if not data or detach_key in data:
                    break
                try:
                    os.write(fifo, data)
                except OSError:
                    break
            chunk = log.read()
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
    finally:
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        os.close(fifo)
        log.close()
    print("\r\n(detached; the session keeps running)")
    return 0


def stop(ddir: Path) -> int:
    pid = running(ddir)
    if pid is None:
        return 1
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    return 0
