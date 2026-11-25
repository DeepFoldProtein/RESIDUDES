"""Deterministic pool operations the agent (and people) call: pool import, residudes-qa jobs, job status, slate.

  residudes pool T [--dir DIR]                    read <targets_root>/T (residudes-engine layout) -> pool.csv + evidence "pool"
  residudes qa T --sto S --what interface|cluster submit residudes-qa through sbatch; outputs under <dossier>/qa/
  residudes jobs T                                refresh the state of every recorded Slurm job
  residudes slate T --sto S                       propose 5 models -> slate_vN.json/.md + evidence "slate_proposal"
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path

from residudes import pool as poolmod
from residudes import slate as slatemod
from residudes.config import Config
from residudes.dossier import Dossier, now


def target_dir(cfg: Config, d: Dossier, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if d.sequence.get("source_dir"):
        return Path(d.sequence["source_dir"])
    if cfg.targets_root and (Path(cfg.targets_root) / d.target).is_dir():
        return Path(cfg.targets_root) / d.target
    raise FileNotFoundError(f"no prediction directory for {d.target}: pass --dir or set targets_root in residudes.toml")


def pool_cmd(cfg: Config, d: Dossier, explicit: str | None, actor: str, refresh: bool = False, threads: int = 8) -> str:
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    if (ddir / "pool.csv").exists() and d.evidence.get("pool") and not refresh:
        last = d.evidence["pool"][-1]["value"]
        lines = [f"pool already imported: {last['n_models']} models from {last['dir']} -> {last['csv']} (--refresh to re-read)"]
        for k, v in last["by_sto_variant_runner"].items():
            lines.append(f"  {k:28s} n={v['n']:5d}  rs median {v['ranking_score_median']}  max {v['ranking_score_max']}  ipTM max {v['iptm_max']}")
        return "\n".join(lines)
    root = target_dir(cfg, d, explicit)
    empty: list[str] = []
    rows = poolmod.scan(root, threads=threads, empty=empty)
    if not rows:
        raise FileNotFoundError(f"no summary.json under {root}/results or {root}/<chain>/results")
    poolmod.write_csv(rows, ddir / "pool.csv")
    summ = poolmod.summarize(rows)
    d.sequence["source_dir"] = str(root)
    d.evidence.setdefault("pool", []).append({"time": now(), "by": actor, "value": {"dir": str(root), "n_models": len(rows),
                                              "csv": str(ddir / "pool.csv"), "by_sto_variant_runner": summ}})
    d.note(actor, f"pool imported: {len(rows)} models from {root}")
    d.evidence["pool"][-1]["value"]["empty_models"] = {"n": len(empty), "first": empty[:20]}
    lines = [f"{len(rows)} models from {root} -> {ddir / 'pool.csv'}"]
    if empty:
        lines.append(f"left out {len(empty)} models whose model.cif is missing or empty (e.g. {empty[0]}): "
                     "a silent failure of those runs; `residudes review` flags it")
    for k, v in summ.items():
        lines.append(f"  {k:28s} n={v['n']:5d}  rs median {v['ranking_score_median']}  max {v['ranking_score_max']}  ipTM max {v['iptm_max']}")
    return "\n".join(lines)


def _sbatch(cmd: str, name: str, log: Path, gpu: bool, hours: int, after: list[str] | None = None, cpus: int = 8) -> str:
    if not shutil.which("sbatch"):
        raise FileNotFoundError("sbatch not found: this needs a Slurm host (gates also run here with --local, up to 100 models)")
    args = ["sbatch", "--parsable", "-p", "normal", "-c", str(cpus), "-t", f"{hours:02d}:00:00", "-J", name, "-o", str(log)]
    if gpu:
        args += ["--gres=gpu:1"]
    if after:
        args += [f"--dependency=afterok:{':'.join(after)}"]
    r = subprocess.run(args + ["--wrap", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"sbatch failed: {r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]


def qa_cmd(cfg: Config, d: Dossier, sto: str, what: str, variants: str | None, actor: str, mode: str = "ppi",
           ab_chains: str | None = None, ag_chains: str | None = None) -> str:
    if "@" in sto:
        raise ValueError(f"{sto}: residudes-qa reads <target>/results only; per-chain runs (A1@B) are not supported yet")
    root = target_dir(cfg, d)
    lqa = Path(cfg.residudes_qa) / "bin" / "residudes-qa"
    qa = Dossier.path(cfg.targets_dir, d.target).parent / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    if what == "interface" and mode == "ab":
        # antibody-antigen: the CASP17 ab model (xgb_ab); residudes-qa writes xgb_scores.csv, which the slate reads too
        if not (ab_chains and ag_chains):
            raise ValueError("--mode ab needs --ab-chains and --ag-chains (model chain ids, e.g. --ab-chains CD --ag-chains AB)")
        out = qa / f"interface_{sto}_ab"
        cmd = (f"{lqa} interface run-all --mode ab --target-dir {shlex.quote(str(root))} --sto {sto} -o {out} "
               f"--ab-chains {shlex.quote(ab_chains)} --ag-chains {shlex.quote(ag_chains)}")
        if variants:
            cmd += f" --variants {variants}"
    elif what == "interface":
        out = qa / f"interface_{sto}"
        cmd = f"{lqa} interface run-ppi-v4 --target-dir {shlex.quote(str(root))} --sto {sto} --targets dockq,ics,ips -o {out}"
        if variants:
            cmd += f" --variants {variants}"
    elif what == "cluster":
        out = qa / f"cluster_{sto}"
        cmd = f"{lqa} cluster {shlex.quote(str(root))} --sto {sto} --method tm --gpu 0 --no-medoid-pdbs --output-base {out}"
        # residudes-qa's cluster defaults to the variant `af3_msa`, the CASP17 campaign's name for its main input;
        # any other layout failed on the first try (H1388: results/A1B1/{base,sep}, "default variant af3_msa
        # absent"). Without --variants, name the ones this target actually has.
        if not variants:
            found = root / "results" / sto
            variants = ",".join(sorted(x.name for x in found.iterdir() if x.is_dir())) if found.is_dir() else None
        if variants:
            cmd += f" --variants {variants}"
    else:
        raise ValueError("what must be interface or cluster")
    job = _sbatch(cmd, f"residudes-{what}-{d.target}", qa / f"{what}_{sto}.%j.log", gpu=True, hours=6)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm", "what": f"residudes-qa {what} {sto}", "state": "submitted",
                   "cmd": cmd, "out": str(out)})
    d.note(actor, f"submitted residudes-qa {what} for {sto}: job {job}")
    return f"job {job}: {cmd}"


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
            "UNKNOWN", "done", "failed", "cancelled"}           # UNKNOWN = purged from sacct; everything else is still active


def scheduler(r: dict) -> str:
    """slurm | pueue | other. Runs recorded before the field existed: residudes's own sbatch jobs are slurm."""
    if r.get("scheduler"):
        return r["scheduler"]
    return "slurm" if str(r.get("what", "")).startswith(("residudes-qa", "retro:")) and str(r.get("job", "")).isdigit() else "other"


def tracked(r: dict) -> bool:
    return scheduler(r) in ("slurm", "pueue") and str(r.get("job", "")).strip() != ""


def state_of(r: dict) -> str:
    """Normalised job state: "CANCELLED by 123" and sacct's truncated "CANCELLED+" are both CANCELLED."""
    return str(r.get("state") or "submitted").split()[0].rstrip("+")


def active(r: dict) -> bool:
    return state_of(r) not in TERMINAL


def _slurm_state(job: str) -> str | None:
    try:
        q = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, timeout=60)
        if q.returncode == 0 and q.stdout.strip():
            return q.stdout.strip().splitlines()[0]
        s = subprocess.run(["sacct", "-n", "-X", "-j", job, "-o", "State"], capture_output=True, text=True, timeout=60)
        return (s.stdout.split() or ["UNKNOWN"])[0] if s.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None                                               # Slurm not reachable from here: keep the last state


def _pueue_states() -> dict[str, str] | None:
    try:
        r = subprocess.run(["pueue", "status", "--json"], capture_output=True, text=True, timeout=60)
        tasks = json.loads(r.stdout).get("tasks", {}) if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    out = {}
    for tid, t in (tasks or {}).items():
        st = t.get("status")
        if isinstance(st, dict) and "Done" in st:
            res = st["Done"].get("result", st["Done"]) if isinstance(st["Done"], dict) else st["Done"]
            out[str(tid)] = "done" if res == "Success" else "failed"
        else:
            out[str(tid)] = str(st if isinstance(st, str) else next(iter(st), "unknown"))
    return out


def jobs_cmd(d: Dossier) -> str:
    """Refresh tracked runs: Slurm via squeue/sacct, pueue via the local daemon. A pueue task the local daemon
    does not know (a remote pod) stays active until someone records its state with `residudes run-record --state`."""
    lines, pueue = [], None
    for r in d.runs:
        job, sch = str(r.get("job", "")), scheduler(r)
        if tracked(r) and active(r):
            if sch == "slurm":
                r["state"] = _slurm_state(job) or r.get("state")
            else:
                pueue = _pueue_states() if pueue is None else pueue
                r["state"] = (pueue or {}).get(job, r.get("state") if r.get("state") not in (None, "submitted") else "pueue-unknown")
        lines.append(f"  {job:>8s} {sch:6s} {str(r.get('state')):12s} {r.get('what')}")
    return "\n".join(lines) or "no runs recorded"


def slate_cmd(cfg: Config, d: Dossier, sto: str | None, use_interface: bool, actor: str, models: str | None = None,
              gates: str | None = None, site: str | None = None) -> str:
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    pc = ddir / "pool.csv"
    if not pc.exists():
        raise FileNotFoundError(f"no pool.csv for {d.target}: run `residudes pool {d.target}` first")
    qa = ddir / "qa" if (ddir / "qa").exists() else None
    from residudes import gate as gatemod
    if models:
        names = [m.strip() for m in models.split(",") if m.strip()]
        s = slatemod.from_models(pc, names, qa, use_interface=use_interface)
        fails: dict = {}
        for part in {m.split("__")[0] for m in names}:
            c = gatemod.result_csv(ddir, part, gates)
            if c:
                fails.update(gatemod.failures(c))
        for m in s["models"]:
            m["gate_fail"] = fails.get(m["model"]) if fails else None
    elif sto:
        allowed, notes = {}, {}
        for part, _ in slatemod.parse_spec(sto):
            keep, note = gatemod.for_slate(ddir, part, gates, site)
            if keep is not None:
                allowed[part], notes[part] = keep, note
        s = slatemod.propose(pc, sto, qa, use_interface=use_interface, allowed=allowed or None)
        s["gated"] = notes
    else:
        raise ValueError("give --sto (propose) or --models (record a chosen slate)")
    version = len(d.evidence.get("slate_proposal", [])) + 1
    j, md = slatemod.write(s, ddir, version)
    d.evidence.setdefault("slate_proposal", []).append({"time": now(), "by": actor, "value": {"version": version, "json": str(j),
                                                         "chosen": bool(s.get("chosen")), "models": [m["model"] for m in s["models"]],
                                                         "gated": s.get("gated"),
                                                         **({"relaxed": {m["model"]: m["relaxed"] for m in s["models"] if m.get("relaxed")}}
                                                            if any(m.get("relaxed") for m in s["models"]) else {}),
                                                         **({"gate_fail": {m["model"]: m["gate_fail"] for m in s["models"]
                                                                           if m.get("gate_fail")}} if s.get("chosen") else {})}})
    return md.read_text()
