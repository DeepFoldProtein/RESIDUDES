"""What this machine and cluster have right now. Read-only, cheap probes only (no du/find on shared storage)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    exe = shutil.which(cmd[0]) or (cmd[0] if Path(cmd[0]).exists() else None)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def host() -> dict:
    mem = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem = int(line.split()[1]) / 1e6
    except OSError:
        pass
    return {"host": platform.node(), "os": f"{platform.system()} {platform.release()}", "cpus": os.cpu_count(),
            "mem_gb": round(mem, 1), "python": platform.python_version()}


def local_gpus() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").strip().splitlines():
        name, mem, drv = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "mem_gb": round(float(mem) / 1024, 1), "driver": drv})
    return gpus


def slurm() -> dict | None:
    out = _run(["sinfo", "-h", "-o", "%P|%a|%D|%T|%G|%f"])
    if out is None:
        return None
    parts: dict[str, dict] = {}
    for line in out.strip().splitlines():
        part, avail, nodes, state, gres, feat = line.split("|")
        p = parts.setdefault(part.rstrip("*"), {"default": part.endswith("*"), "avail": avail, "gpus": {}, "nodes": {}})
        p["nodes"][state] = p["nodes"].get(state, 0) + int(nodes)
        for g in gres.split(","):
            bits = g.split("(")[0].split(":")              # gpu:RTX4090:6
            if bits[0] == "gpu" and len(bits) >= 3 and bits[-1].isdigit():
                p["gpus"][bits[1]] = p["gpus"].get(bits[1], 0) + int(bits[-1]) * int(nodes)
    q = _run(["squeue", "-h", "-o", "%T"]) or ""
    states: dict[str, int] = {}
    for s in q.split():
        states[s] = states.get(s, 0) + 1
    return {"partitions": parts, "queue": states}


def storage(paths: list[str]) -> list[dict]:
    rows, seen = [], set()
    for p in paths:
        if not p or not Path(p).exists():
            continue
        try:
            st = os.statvfs(p)
        except OSError:
            continue
        key = (st.f_blocks, st.f_bsize)
        if key in seen:
            continue
        seen.add(key)
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        rows.append({"path": p, "total_tb": round(total / 1e12, 1), "free_tb": round(free / 1e12, 1),
                     "used_pct": round(100 * (1 - free / total), 1) if total else None})
    return rows


def gpfs_quota() -> str | None:
    for exe in ("mmlsquota", "/usr/lpp/mmfs/bin/mmlsquota"):
        out = _run([exe, "--block-size", "auto"], timeout=15)
        if out:
            return out.strip()
    return None


def tools() -> dict:
    """Programs lantern calls, and the Python packages its commands import (marked "(python)")."""
    import importlib.util
    found = {}
    for t in ("claude", "codex", "uv", "apptainer", "pueue", "sbatch", "rsvg-convert", "typst", "hmmsearch", "hmmscan"):
        found[t] = shutil.which(t) or ""
    for mod, what in (("gemmi", "gemmi (python)"), ("claude_agent_sdk", "claude-agent-sdk (python)")):
        found[what] = "yes" if importlib.util.find_spec(mod) else ""
    return found


def detect(paths: list[str]) -> dict:
    return {"host": host(), "gpus": local_gpus(), "slurm": slurm(), "storage": storage(paths),
            "gpfs_quota": gpfs_quota(), "tools": tools()}


def recommendations(r: dict, work_root: str) -> list[str]:
    rec = []
    for s in r["storage"]:
        if s["used_pct"] is not None and s["used_pct"] >= 95:
            rec.append(f"{s['path']} is {s['used_pct']}% full ({s['free_tb']} TB free): keep work_root and new databases elsewhere")
    if not r["gpus"] and not r["slurm"]:
        rec.append("no GPU and no Slurm here: prediction and GPU QA must run on another machine")
    if r["slurm"]:
        idle = sum(n.get("idle", 0) for p in r["slurm"]["partitions"].values() for n in [p["nodes"]])
        pend = r["slurm"]["queue"].get("PENDING", 0)
        if pend and not idle:
            rec.append(f"{pend} Slurm jobs pending and no idle node: expect queueing; consider other sites (pueue pods)")
    if not r["tools"].get("gemmi (python)"):
        rec.append("gemmi is missing: `uv sync` in lantern/ (lantern gate and the chain check in lantern review need it)")
    if not r["tools"].get("typst"):
        rec.append("typst is missing: `lantern slides` writes report.typ but no PDF")
    if not r["tools"].get("claude"):
        rec.append("Claude Code CLI not found: `lantern agent` needs it (the Agent SDK drives it)")
    return rec
