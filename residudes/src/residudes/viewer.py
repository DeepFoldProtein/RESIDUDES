"""The structure beside the conversation: a molterm pane the chat agent opens, drives and closes as the person asks.

`residudes chat` is a tmux window with the agent in it. When the conversation turns on a shape - "show me model 1",
"where does the peptide sit", "colour it by pLDDT", "put M3 on it", "close that" - the agent answers in words and
does it in a pane beside its own, in the same window: the person watches the structure move while they talk about
it. The person never types these; the agent does, from what was said:

  residudes view T show M1 [M3 ...]       M1-M5 the slate, C0.. a cluster medoid, tpl_6S9K_A a template, slate, FILE
  residudes view T do "color plddt"       any molterm command, in the open pane ("a; b" for several)
  residudes view T snap                   what the pane shows, as a PNG the agent can read
  residudes view T close

The pane is found again by a tmux pane option it carries (@residudes_view), so "show M3" replaces the structure in
the same pane rather than opening another. tmux older than 3.4 cannot pass pixel graphics through, so there the
pane uses molterm's text renderer; with a sixel-capable tmux and terminal, molterm draws pixels in it.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from residudes.dossier import Dossier

TAG = "@residudes_view"
SLOTS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def here() -> str | None:
    """The pane this process runs in (the agent's own), or None outside tmux."""
    return os.environ.get("TMUX_PANE") if os.environ.get("TMUX") and shutil.which("tmux") else None


def viewer_pane(anchor: str) -> str | None:
    r = _tmux("list-panes", "-t", anchor, "-F", "#{pane_id}\t#{" + TAG + "}")
    for line in r.stdout.splitlines() if r.returncode == 0 else []:
        pane, _, tag = line.partition("\t")
        if tag:
            return pane
    return None


def renderer() -> str | None:
    """None lets molterm pick (`auto`: pixels where it can); 'block' where tmux would swallow them. RESIDUDES_VIEW_RENDERER
    overrides. tmux passes sixel from 3.4 on, built with it, and then lists `sixel` among the client's features."""
    if os.environ.get("RESIDUDES_VIEW_RENDERER"):
        return os.environ["RESIDUDES_VIEW_RENDERER"]
    feats = _tmux("display-message", "-p", "#{client_termfeatures}").stdout
    return None if "sixel" in feats else "block"


def resolve(cfg, d: Dossier, items: list[str]) -> list[dict]:
    """Names the conversation uses -> files: the slate's M1-M5 (or the best model per stoichiometry), cluster medoids,
    the deck's templates, `slate` for all five, or a path."""
    from residudes import slides
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    known = {m["label"]: m for m in slides.figure_models(d, ddir) + slides.cluster_models(ddir)}
    try:
        for t in json.loads((ddir / "report" / "figs" / "templates.json").read_text()).get("templates", []):
            known[t["label"]] = {"label": t["label"], "path": t["path"], "small": t.get("small", [])}
    except (OSError, ValueError):
        pass
    out, unknown = [], []
    for it in items or [next(iter(known), "")]:
        if it == "slate":
            out += [m for k, m in known.items() if re.fullmatch(r"M\d+", k)]
        elif it in known:
            out.append(known[it])
        elif Path(it).expanduser().exists():
            p = Path(it).expanduser().resolve()
            out.append({"label": re.sub(r"\W", "_", p.stem), "path": str(p)})
        else:
            unknown.append(it)
    if unknown or not out:
        names = ", ".join(sorted(known)) or "none yet"
        raise ValueError(f"nothing to show for {', '.join(unknown) or 'this target'}: known here are {names}, or give a file")
    return out


def script(models: list[dict], small: tuple[str, ...], render: str | None, camera: str | None) -> str:
    """The molterm script the pane starts with: each model under its own name, superposed on the first by sequence,
    one colour per model when there are several, chains when there is one, a small chain as spheres."""
    # always said: ~/.molterm/init.mt runs first and may pin a renderer (`pixel` leaves a tmux 3.2a pane black)
    lines = [f"set renderer {render or 'auto'}"]
    for m in models:
        lines += [f"load {m['path']}", f"rename {Path(m['path']).stem} {m['label']}"]
    first = models[0]["label"]
    lines += [f"align {m['label']} to {first} automap" for m in models[1:]]
    lines += ["hide all", "show cartoon"]
    if len(models) == 1:
        lines.append("color chain")
        for ch in small or models[0].get("small", []):
            lines += [f"show spacefill chain {ch}", f"color #D55E00 chain {ch}"]
    else:
        lines += [f"color {SLOTS[i % len(SLOTS)]} {m['label']}" for i, m in enumerate(models)]
    lines += [f"camera load {camera}", f"zoom {first}"] if camera else [f"orient {first}"]
    return "\n".join(lines) + "\n"


def show(cfg, d: Dossier, items: list[str]) -> str:
    from residudes import report, slides
    from residudes.dossier import chain_letter
    anchor = here()
    if not anchor:
        return "not inside tmux: there is no window to put the structure in (describe it, or `molterm --no-tui` a PNG)"
    mt = slides.tool(cfg, "molterm")
    if not mt:
        return 'molterm not found: set [tools] molterm in residudes.toml'
    models = resolve(cfg, d, items)
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    small = tuple(chain_letter(i) for i, (_, seq) in enumerate(report._chains(d)) if len(seq) <= 40)
    small = small if not models[0]["label"].startswith("tpl_") else tuple(models[0].get("small", []))
    work = ddir / "report" / "view"
    work.mkdir(parents=True, exist_ok=True)
    camera = None
    if small:                                               # face the small chain, as the deck's pictures do
        cam = slides.face_camera(models[0]["path"], small)
        if cam:
            (work / "view.cam").write_text(cam)
            camera = str(work / "view.cam")
    sc = work / "view.mt"
    sc.write_text(script(models, small, renderer(), camera))
    # TERM: tmux's default `screen` tells ncurses there are 8 colours; molterm draws with more when told the truth
    cmd = f"TERM=xterm-256color {shlex.quote(mt)} --tui --script {shlex.quote(str(sc))}"
    pane = viewer_pane(anchor)
    if pane:
        r = _tmux("respawn-pane", "-k", "-t", pane, cmd)
    else:
        r = _tmux("split-window", "-h", "-d", "-l", "50%", "-t", anchor, "-P", "-F", "#{pane_id}", cmd)
        pane = r.stdout.strip()
        if r.returncode == 0:
            _tmux("set-option", "-p", "-t", pane, TAG, d.target)
    if r.returncode != 0:
        return f"tmux refused the pane: {r.stderr.strip()}"
    names = ", ".join(f"{m['label']} ({m.get('name') or Path(m['path']).name})" for m in models)
    return (f"showing {names} in the pane beside this one"
            + (" (superposed on the first, one colour each)" if len(models) > 1 else "")
            + ("; text renderer: this tmux cannot pass pixels" if renderer() else ""))


def do(cfg, d: Dossier, commands: str) -> str:
    """Type molterm commands into the open pane, as a person would: Escape, `:command`, Enter. Returns the pane's
    last line, which is where molterm answers - an error included."""
    anchor = here()
    pane = viewer_pane(anchor) if anchor else None
    if not pane:
        return "no structure is open beside this conversation: `residudes view T show ...` first"
    for c in [c.strip() for c in commands.split(";") if c.strip()]:
        _tmux("send-keys", "-t", pane, "Escape")
        _tmux("send-keys", "-t", pane, "-l", ":" + c)
        _tmux("send-keys", "-t", pane, "Enter")
        time.sleep(0.3)
    last = [ln for ln in _tmux("capture-pane", "-p", "-t", pane).stdout.splitlines() if ln.strip()]
    return f"molterm: {last[-1].strip() if last else '(nothing)'}"


def snap(cfg, d: Dossier) -> str:
    """What the person is looking at, as a PNG: molterm draws its screenshot offscreen in pixels whatever renderer
    the pane uses, so the agent can read the file and talk about the same view."""
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    out = ddir / "report" / "view" / f"snap-{time.strftime('%H%M%S')}.png"
    said = do(cfg, d, f"screenshot {out} 1200 900")
    for _ in range(20):
        if out.exists() and out.stat().st_size:
            return f"{out}  (read it to see what the pane shows)"
        time.sleep(0.25)
    return f"no snapshot written ({said})"


def close(cfg, d: Dossier) -> str:
    anchor = here()
    pane = viewer_pane(anchor) if anchor else None
    if not pane:
        return "nothing open beside this conversation"
    _tmux("kill-pane", "-t", pane)
    return "closed the structure pane"


def view_cmd(cfg, d: Dossier, action: str, rest: list[str]) -> str:
    if action == "show":
        return show(cfg, d, rest)
    if action == "do":
        return do(cfg, d, " ".join(rest))
    if action == "snap":
        return snap(cfg, d)
    if action == "close":
        return close(cfg, d)
    raise ValueError(f"view: show | do | snap | close, not {action!r}")
