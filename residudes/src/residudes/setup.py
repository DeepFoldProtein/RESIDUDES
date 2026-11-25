"""`residudes setup` (first run, done together with the user) and `residudes doctor` (re-check any time).

setup steps, each shown as findings -> proposal -> confirmation:
  1 resources   machine, GPUs, Slurm partitions and queue, storage and quota, tools; with recommendations
  2 catalog     databases / weights / tools from catalog.toml: present, age, what to do
  3 components  residudes-engine and residudes-qa: found? their own `doctor`
  4 config      residudes.toml: work root (suggested from free space), budgets, model
  5 skills      link residudes/skills into a project's .claude/skills (Claude Code), .agents/skills (Codex, Antigravity)
                and .agent/skills (older Antigravity); AGENTS.md / CLAUDE.md carry the shared instructions
  6 tour        the commands to start a target
Nothing is downloaded or overwritten without a yes (or --yes).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from residudes import catalog, resources
from residudes.config import RESIDUDES_DIR, Config, config_path, load, save

SKILLS = RESIDUDES_DIR / "skills"


def _ask(prompt: str, default: str, yes: bool) -> str:
    if yes:
        return default
    try:
        ans = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return ans or default


def _confirm(prompt: str, yes: bool) -> bool:
    return yes or _ask(prompt + " (y/n)", "y", False).lower().startswith("y")


def _component(root: str, exe: str) -> dict:
    b = Path(root) / "bin" / exe
    if not b.exists():
        return {"name": exe, "status": "missing", "path": str(b)}
    try:
        r = subprocess.run([str(b), "doctor"], capture_output=True, text=True, timeout=600)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
        return {"name": exe, "status": "ok" if r.returncode == 0 else "check", "path": str(b), "doctor": lines[-12:]}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"name": exe, "status": "check", "path": str(b), "doctor": [str(exc)]}


def report(cfg: Config) -> dict:
    paths = [cfg.work_root, cfg.targets_root, str(RESIDUDES_DIR),
             *(d for d in os.environ.get("RESIDUDES_WORK_ROOTS", "").split(":") if d)]
    r = resources.detect(paths)
    return {"resources": r, "recommendations": resources.recommendations(r, cfg.work_root),
            "catalog": catalog.check_all(paths=cfg.extra.get("databases") or {}),
            "components": [_component(cfg.residudes_engine, "residudes-engine"), _component(cfg.residudes_qa, "residudes-qa")]}


# ---------------------------------------------------------------------- colour
# Setup prints a wall of status lines, and the three that matter (a stale database, a missing
# component, a filesystem that is nearly full) are easy to miss in a plain one. Colour only when
# the output is a terminal and the user has not asked for none; NO_COLOR is respected because
# this output gets piped into files and issues.
_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def green(s): return _c(s, "32")
def yellow(s): return _c(s, "33")
def red(s): return _c(s, "31")
def bold(s): return _c(s, "1")
def dim(s): return _c(s, "2")


MARKS = {"ok": (green, "ok   "), "stale": (yellow, "STALE"), "missing": (red, "MISS "),
         "unconfigured": (yellow, "UNSET"), "check": (yellow, "check"), "found": (green, "found")}


def mark(status: str) -> str:
    colour, label = MARKS.get(status, (dim, f"{status:5s}"))
    return "[" + colour(label) + "]"


def _print_report(rep: dict) -> None:
    _print_resources(rep)
    _print_catalog_and_components(rep)


def _print_resources(rep: dict) -> None:
    r = rep["resources"]
    h = r["host"]
    print(bold(f"\n== 1. resources on {h['host']}") + f"  ({h['cpus']} CPUs, {h['mem_gb']} GB RAM, python {h['python']})")
    if r["gpus"]:
        names = sorted({g["name"] for g in r["gpus"]})
        print(f"   local GPUs: {len(r['gpus'])} x {', '.join(names)} (driver {r['gpus'][0]['driver']})")
    if r["slurm"]:
        for name, p in r["slurm"]["partitions"].items():
            gpus = ", ".join(f"{k} x{v}" for k, v in p["gpus"].items()) or "no GPUs"
            nodes = ", ".join(f"{k} {v}" for k, v in p["nodes"].items())
            print(f"   slurm {name}{' (default)' if p['default'] else ''}: {gpus}; nodes {nodes}")
        print(f"   queue: {', '.join(f'{k} {v}' for k, v in r['slurm']['queue'].items()) or 'empty'}")
    for s in r["storage"]:
        use = s["used_pct"]
        shown = f"{use}% used" if use is None else (red(f"{use}% used") if use >= 95 else
                                                    yellow(f"{use}% used") if use >= 90 else f"{use}% used")
        print(f"   storage {s['path']}: {shown}, {s['free_tb']} TB free of {s['total_tb']} TB")
    if r["gpfs_quota"]:
        print("   gpfs quota:\n      " + "\n      ".join(r["gpfs_quota"].splitlines()[:4]))
    print("   tools: " + ", ".join(k if v else yellow(f"{k} (missing)") for k, v in r["tools"].items()))
    for rec in rep["recommendations"]:
        print("   " + yellow("->") + f" {rec}")


def _print_catalog_and_components(rep: dict) -> None:
    print(bold("\n== 4. databases, weights, tools") + " (catalog.toml)")
    for c in rep["catalog"]:
        print(f"   {mark(c['status'])} {c['name']:32s} {dim(c['date'] or ''):10s}  {c['note']}")
        if c["fix"]:
            print("           " + dim(f"fix: {c['fix']}"))
    print(bold("\n== 5. components"))
    for c in rep["components"]:
        print(f"   {mark(c['status'])} {c['name']} {dim('(' + str(c['path']) + ')')}")
        for line in c.get("doctor", [])[-8:]:
            print(f"             {line}")


def _suggest_work_root(rep: dict, current: str) -> tuple[str, list[str]]:
    """What to put on the work_root prompt, and the other places worth offering.

    A first run should not have to know one cluster's directory layout, so the default is a
    folder in the directory the command was run from. Shared dossier directories that actually
    exist are offered beside it, because on this cluster that is where the team's targets are.
    """
    if os.path.isabs(current) and not Path(current).resolve().is_relative_to(RESIDUDES_DIR.resolve()):
        return current, []                                     # an existing choice is left alone
    here = str(Path.cwd() / "work_root")
    # A site can name its shared dossier directories in RESIDUDES_WORK_ROOTS (colon separated);
    # the ones that exist are offered beside the default instead of being hard-coded here.
    others = [d for d in os.environ.get("RESIDUDES_WORK_ROOTS", "").split(":") if d and Path(d).is_dir()]
    return here, others


HARNESS_SKILL_DIRS = {                                   # where each harness looks for workspace skills
    ".claude/skills": "Claude Code",
    ".agents/skills": "Codex and Antigravity",
    ".agent/skills": "Antigravity (older builds)",
}


COPY_MARKER = ".residudes-skill-copy"     # in a copied skill folder: where it came from, so a rerun may refresh it


def link_skills(project: Path, copy: bool = False) -> list[str]:
    """Each residudes skill into each harness's skill folder: a symlink, or with copy=True a real copy (for a harness
    that does not follow symlinks: Codex documents that it does, Antigravity's docs do not say). A copy carries a
    marker and is refreshed in place on the next run; folders residudes did not create are never touched."""
    import shutil
    done = []
    for sub in HARNESS_SKILL_DIRS:
        d = project / sub
        d.mkdir(parents=True, exist_ok=True)
        for s in sorted(SKILLS.iterdir()):
            if not (s / "SKILL.md").exists():
                continue
            dst = d / s.name
            ours = dst.is_dir() and not dst.is_symlink() and (dst / COPY_MARKER).exists()
            if dst.is_symlink():
                target = Path(os.readlink(dst))
                residudes_link = dst.resolve() == s.resolve() or (target.name == s.name and target.parent.name == "skills")
                if (copy and residudes_link) or not dst.exists():
                    dst.unlink()                       # residudes's own (or a dangling) link: replaced, no data lost
            if ours or (copy and not (dst.is_symlink() or dst.exists())):
                shutil.copytree(s.resolve(), dst, dirs_exist_ok=True, symlinks=False)
                (dst / COPY_MARKER).write_text(f"copied from {s.resolve()} by `residudes setup --copy-skills`; rerun it to refresh\n")
                done.append(str(dst))
                continue
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(s.resolve())
            done.append(str(dst))
    return done


# What the catalogue asks for, in the order someone would answer it: the big, slow things first,
# because a machine that already has 630 GB of databases should never be asked to download them,
# and the licensed ones cannot be downloaded at all.
DB_QUESTIONS = [
    ("af3_db", "AlphaFold 3 databases and af3.bin (the nine genetic databases; ~630 GB)"),
    ("colabfold_db", "ColabFold databases (the faster MSA route)"),
    ("foldseek_db", "foldseek PDB (structure search)"),
    ("foldseek_pdb_2026", "a newer foldseek PDB, if this machine has one"),
    ("campaign", "an existing campaign tree (targets, analysis, esm)"),
    ("containers", "a directory of .sif containers (OpenStructure)"),
    ("sabdab", "SAbDab (antibody structures)"),
]


def ask_databases(cfg, yes: bool) -> dict:
    """Ask what this machine already has, before anything is checked or installed."""
    have = dict(cfg.extra.get("databases") or {})
    if yes:
        return have
    print(bold("\n== 3. what this machine already has"))
    print("   " + dim("one line each, empty if this machine does not have it; `residudes doctor` re-checks later"))
    for key, what in DB_QUESTIONS:
        print(f"\n   {what}")                     # the description on its own line, then the prompt
        ans = _ask(f"     [databases] {key}", have.get(key, ""), yes)
        if ans:
            have[key] = ans
        else:
            have.pop(key, None)
    cfg.extra["databases"] = have
    return have

def run_setup(cfg_path: str | None, yes: bool, as_json: bool, project: str | None, copy_skills: bool = False) -> int:
    cfg = load(cfg_path)
    if as_json:
        print(json.dumps(report(cfg), indent=1, default=str))
        return 0
    _print_resources(report(cfg))          # 1. what this machine is
    choose_harness(cfg, yes)               # 2. which agent CLI drives it
    ask_databases(cfg, yes)                # 3. what it already has, asked before anything is checked
    rep = report(cfg)                      # 3-4. the catalogue and the components, with those answers
    _print_catalog_and_components(rep)

    print(bold("\n== 6. residudes.toml"))
    p = config_path(cfg_path)
    suggested, others = _suggest_work_root(rep, cfg.work_root)
    if others:                      # say what else is there rather than silently preferring it
        print("   " + dim("shared dossier directories on this machine: " + ", ".join(others)))
    cfg.work_root = _ask("   where should target dossiers live (work_root)", suggested, yes)
    cfg.targets_root = _ask("   existing predictions tree (residudes-engine layout), empty if none", cfg.targets_root
                            or "", yes)
    cfg.agent_usd_per_step = float(_ask("   agent budget per step (USD)", str(cfg.agent_usd_per_step), yes))
    cfg.gpu_hours_per_target = float(_ask("   GPU hours per target before asking", str(cfg.gpu_hours_per_target), yes))
    if not p.exists() or _confirm(f"   overwrite {p}?", yes):
        save(cfg, p)
        Path(cfg.work_root, "targets").mkdir(parents=True, exist_ok=True)
        print("   " + green("wrote") + f" {p}; dossiers in {cfg.work_root}/targets")

    print(bold("\n== 7. skills") + " (playbooks for Claude Code, Codex and Antigravity)")
    proj = Path(project) if project else None
    if proj is None and not yes:
        ans = _ask("   project directory to link the skills into (empty to skip)", "", yes)
        proj = Path(ans) if ans else None
    if proj:
        linked = link_skills(proj.resolve(), copy=copy_skills)
        print(f"   linked {len(linked)} skill dirs into {proj}/" + ", ".join(HARNESS_SKILL_DIRS) + " ("
              + "; ".join(f"{k}: {v}" for k, v in HARNESS_SKILL_DIRS.items()) + ")")
    print("   inside residudes/ every harness already finds them, and AGENTS.md (CLAUDE.md imports it) holds the rules")

    # Setting a machine up and never fetching what runs on it is a half-setup: the next command
    # would fail on a missing image. The guided mode's agent does this by running verify.sh; the
    # questions have to offer it too.
    verify = RESIDUDES_DIR.parent / "containers" / "verify.sh"
    print(bold("\n== 8. container images"))
    if not verify.exists():
        print("   " + dim(f"no {verify} in this checkout: skipping"))
    elif yes:
        # --yes means "take the defaults", not "spend an hour": obtaining eight images is a
        # deliberate act, so it is printed rather than started.
        print("   " + dim(f"not started under --yes: {verify}"))
    elif _confirm("   fetch the images now and check the installation? (containers/verify.sh)", False):
        print(dim("   " + str(verify)))
        rc = subprocess.call(["bash", str(verify)])
        print(("   " + green("the installation checks out")) if rc == 0 else
              ("   " + yellow(f"verify.sh exited {rc}: see the lines above")))
    else:
        print("   " + dim(f"later: {verify}   (--gpu for the GPU images, --af3-db DIR --deep for the databases)"))

    print(bold("\n== 9. how to start a target"))
    print("   residudes target add T1234 --fasta T1234.fasta [--stoich A3]   # creates the dossier")
    print("   residudes next T1234                                        # what the next step is, and its skill")
    print("   residudes agent step T1234                                  # let the agent do that step (asks nothing outward)")
    print("   residudes target show T1234                                 # dossier: stage, evidence, decisions")
    print("   residudes approve T1234 --what 'slate v1'                   # only a person moves a target past SLATE_PROPOSED")
    return 0


def skill_check(roots: list[Path]) -> list[dict]:
    """For each harness skill folder under each root: how many residudes skills a harness would find there, and any
    broken entry (a dangling symlink, a folder without SKILL.md). A broken one leaves that harness without the skill."""
    names = {s.name for s in SKILLS.iterdir() if (s / "SKILL.md").exists()}
    out = []
    for root in roots:
        for sub, who in HARNESS_SKILL_DIRS.items():
            d = root / sub
            if not d.exists() and not d.is_symlink():
                continue
            found, broken = set(), []
            for e in sorted(d.iterdir()) if d.is_dir() else []:
                if e.name.startswith("."):
                    continue
                if e.is_symlink() and not e.exists():
                    broken.append(e.name)              # dangling
                elif (e / "SKILL.md").exists():
                    if e.name in names:
                        found.add(e.name)
                elif e.name in names:
                    broken.append(e.name)              # a residudes skill's folder without its SKILL.md
            out.append({"folder": str(d), "harness": who, "found": len(found), "of": len(names),
                        "missing": sorted(names - found), "broken": broken})
    return out


def run_doctor(cfg_path: str | None, as_json: bool) -> int:
    cfg = load(cfg_path)
    rep = report(cfg)
    rep["skills"] = skill_check([RESIDUDES_DIR, RESIDUDES_DIR.parent])
    bad = [c for c in rep["catalog"] if c["status"] == "missing"] + [c for c in rep["components"] if c["status"] != "ok"] \
        + [x for x in rep["skills"] if x["broken"] or x["missing"]]
    if as_json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        _print_report(rep)
        print("\n== skills, as each harness finds them")
        for x in rep["skills"]:
            state = "ok" if not (x["broken"] or x["missing"]) else "CHECK"
            print(f"   [{state:5s}] {x['folder']} ({x['harness']}): {x['found']}/{x['of']} residudes skills"
                  + (f"; missing {', '.join(x['missing'])}" if x["missing"] else "")
                  + (f"; broken {', '.join(x['broken'])}" if x["broken"] else ""))
        print(f"\nconfig: {cfg.path if Path(cfg.path).exists() else 'none (run `residudes setup`)'}; "
              f"claude: {shutil.which('claude') or 'missing'}")
    return 1 if bad else 0


# ------------------------------------------------------------------ guided setup
# Setting a campaign up means reading a report, deciding what to install, and then running the
# installs - which is work a coding agent can do, and which this project already has a skill for.
# So `residudes setup` hands the job to whichever agent CLI is on the machine and falls back to
# the questions below when there is none. `--manual` always gets the questions.
HARNESS_CLI = (                       # command, name, how to pass a prompt and keep the terminal
    ("claude", "Claude Code", lambda prompt: ["claude", prompt]),
    ("codex", "Codex CLI", lambda prompt: ["codex", prompt]),
    ("ag", "Antigravity CLI", lambda prompt: ["ag", prompt]),
    ("gemini", "Gemini CLI", lambda prompt: ["gemini", "-i", prompt]),
)

# An agent CLI authenticates as a person and is licensed to one, so it is not in the image and
# not installed for you: setup picks from what is here and otherwise says where to get one.
HARNESS_INSTALL = {
    "claude": "npm install -g @anthropic-ai/claude-code   (then `claude` once, to sign in)",
    "codex": "npm install -g @openai/codex                (then `codex` once, to sign in)",
    "ag": "see the Antigravity docs for your platform",
    "gemini": "npm install -g @google/gemini-cli          (then `gemini` once, to sign in)",
}


def choose_harness(cfg, yes: bool) -> str | None:
    """Which agent CLI this machine will use, from the ones installed."""
    here = [(cmd, name) for cmd, name, _ in HARNESS_CLI if shutil.which(cmd)]
    current = (cfg.extra.get("agent", {}) or {}).get("cli") if cfg else None
    print(bold("\n== 2. the agent CLI"))
    if not here:
        print("   " + yellow("none installed.") + " residudes drives one of these; install whichever you have access to:")
        for cmd, name, _ in HARNESS_CLI:
            print(f"     {name:16} {dim(HARNESS_INSTALL[cmd])}")
        print("   " + dim("then run `residudes setup` again, or `residudes doctor` to re-check"))
        return None
    print("   installed: " + ", ".join(green(n) for _, n in here))
    if len(here) == 1 and not current:
        chosen = here[0][0]
        print(f"   using {green(here[0][1])}")
    else:
        chosen = _ask("   which should residudes use", current or here[0][0], yes)
    cfg.extra.setdefault("agent", {})["cli"] = chosen
    return chosen


def available_harness(preferred: str | None = None) -> tuple[str, str, list] | None:
    for cmd, name, argv in HARNESS_CLI:
        if preferred and cmd != preferred:
            continue
        if shutil.which(cmd):
            return cmd, name, argv
    return None


SETUP_BRIEF = """Set this machine up to run RESIDUDES, with me. One step at a time: ask, wait for my
answer, then do that step before moving on. Read {skill} first - it is the project's own setup
procedure.

1. ASK FIRST, before checking or installing anything, what this machine already has. Most of it is
   large or licensed, so finding it is worth more than fetching it:
     - the AlphaFold 3 genetic databases (~630 GB) and af3.bin, which is licensed per person and
       cannot be downloaded by you at all
     - ColabFold databases, foldseek PDB, SAbDab
     - model checkpoints and TensorRT engines from an earlier installation
     - an existing campaign tree with targets in it
     - Rosetta and a MODELLER key, if relaxation or templates are wanted
   Take the paths I give you, write them into residudes.toml ([databases], [tools]) and move on.
   Where I do not have something, say what it is for and what it costs to get, and let me decide.

2. THEN check, with `{residudes} doctor`, so the report reflects my answers rather than defaults.
   Read it with me: what is present, what is stale, what is missing.

3. THEN install what is missing, one component at a time, explaining each command before you run
   it. The published images cover every part ({containers}); `containers/verify.sh` obtains them
   and checks the installation. Stop at anything that costs money, needs a licence or writes
   outside this machine.

4. FINISH by running `containers/verify.sh` (add `--gpu` where there is a card, and
   `--af3-db DIR --deep` where the databases are) and show me the result.

Do not submit anything anywhere, and do not change RESIDUDES_ACTOR.
"""


def run_setup_guided(cfg_path: str | None, harness: str | None) -> int:
    """Hand the setup to a coding agent on this machine."""
    found = available_harness(harness)
    if not found:
        names = ", ".join(c for c, _, _ in HARNESS_CLI)
        print(yellow(f"   no agent CLI found on PATH ({names}); continuing with the questions"))
        return 1
    cmd, name, argv = found
    root = RESIDUDES_DIR
    prompt = SETUP_BRIEF.format(
        skill=root / "skills" / "residudes-setup" / "SKILL.md",
        residudes=root / "bin" / "residudes",
        containers=root.parent / "containers" / "README.md",
    )
    print(f"\n{bold('== guided setup')} with {green(name)}  ({dim('residudes setup --manual for the questions instead')})")
    try:
        return subprocess.call(argv(prompt))
    except KeyboardInterrupt:
        return 130
