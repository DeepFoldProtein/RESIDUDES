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
    paths = [cfg.work_root, cfg.targets_root, str(RESIDUDES_DIR), "/gpfs/deepfold", "/store/deepfold"]
    r = resources.detect(paths)
    return {"resources": r, "recommendations": resources.recommendations(r, cfg.work_root),
            "catalog": catalog.check_all(paths=cfg.extra.get("databases") or {}),
            "components": [_component(cfg.residudes_engine, "residudes-engine"), _component(cfg.residudes_qa, "residudes-qa")]}


def _print_report(rep: dict) -> None:
    r = rep["resources"]
    h = r["host"]
    print(f"\n== 1. resources on {h['host']}  ({h['cpus']} CPUs, {h['mem_gb']} GB RAM, python {h['python']})")
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
        print(f"   storage {s['path']}: {s['used_pct']}% used, {s['free_tb']} TB free of {s['total_tb']} TB")
    if r["gpfs_quota"]:
        print("   gpfs quota:\n      " + "\n      ".join(r["gpfs_quota"].splitlines()[:4]))
    print("   tools: " + ", ".join(f"{k}{'' if v else ' (missing)'}" for k, v in r["tools"].items()))
    for rec in rep["recommendations"]:
        print(f"   -> {rec}")
    print("\n== 2. databases, weights, tools (catalog.toml)")
    for c in rep["catalog"]:
        mark = {"ok": "ok   ", "stale": "STALE", "missing": "MISS ", "unconfigured": "UNSET"}[c["status"]]
        print(f"   [{mark}] {c['name']:32s} {c['date'] or '':10s}  {c['note']}")
        if c["fix"]:
            print(f"           fix: {c['fix']}")
    print("\n== 3. components")
    for c in rep["components"]:
        print(f"   [{c['status']:7s}] {c['name']} ({c['path']})")
        for line in c.get("doctor", [])[-8:]:
            print(f"             {line}")


def _suggest_work_root(rep: dict, current: str) -> str:
    if os.path.isabs(current) and not Path(current).resolve().is_relative_to(RESIDUDES_DIR.resolve()):
        return current
    store = [s for s in rep["resources"]["storage"] if s["path"].startswith("/store")]
    if store and (100 if store[0]["used_pct"] is None else store[0]["used_pct"]) < 90:
        return "/store/deepfold/lantern"
    return current


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


def run_setup(cfg_path: str | None, yes: bool, as_json: bool, project: str | None, copy_skills: bool = False) -> int:
    cfg = load(cfg_path)
    rep = report(cfg)
    if as_json:
        print(json.dumps(rep, indent=1, default=str))
        return 0
    _print_report(rep)

    print("\n== 4. residudes.toml")
    p = config_path(cfg_path)
    cfg.work_root = _ask("   where should target dossiers live (work_root)", _suggest_work_root(rep, cfg.work_root), yes)
    cfg.targets_root = _ask("   existing predictions tree (residudes-engine layout), empty if none", cfg.targets_root
                            or ("/gpfs/deepfold/casp/casp17/targets" if Path("/gpfs/deepfold/casp/casp17/targets").exists() else ""), yes)
    cfg.agent_usd_per_step = float(_ask("   agent budget per step (USD)", str(cfg.agent_usd_per_step), yes))
    cfg.gpu_hours_per_target = float(_ask("   GPU hours per target before asking", str(cfg.gpu_hours_per_target), yes))
    if not p.exists() or _confirm(f"   overwrite {p}?", yes):
        save(cfg, p)
        Path(cfg.work_root, "targets").mkdir(parents=True, exist_ok=True)
        print(f"   wrote {p}; dossiers in {cfg.work_root}/targets")

    print("\n== 5. skills (playbooks for Claude Code, Codex and Antigravity)")
    proj = Path(project) if project else None
    if proj is None and not yes:
        ans = _ask("   project directory to link the skills into (empty to skip)", "", yes)
        proj = Path(ans) if ans else None
    if proj:
        linked = link_skills(proj.resolve(), copy=copy_skills)
        print(f"   linked {len(linked)} skill dirs into {proj}/" + ", ".join(HARNESS_SKILL_DIRS) + " ("
              + "; ".join(f"{k}: {v}" for k, v in HARNESS_SKILL_DIRS.items()) + ")")
    print("   inside residudes/ every harness already finds them, and AGENTS.md (CLAUDE.md imports it) holds the rules")

    print("\n== 6. how to start a target")
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
