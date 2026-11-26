"""Deterministic pool operations the agent (and people) call: pool import, residudes-qa jobs, job status, slate.

  residudes pool T [--dir DIR]                    read <targets_root>/T (residudes-engine layout) -> pool.csv + evidence "pool"
  residudes qa T --sto S --what interface|cluster submit residudes-qa through sbatch; outputs under <dossier>/qa/
  residudes jobs T                                refresh the state of every recorded Slurm job
  residudes slate T --sto S                       propose 5 models -> slate_vN.json/.md + evidence "slate_proposal"
"""
from __future__ import annotations

import json
import re
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
    own = Dossier.path(cfg.targets_dir, d.target).parent
    if (own / "results").is_dir():                         # the runs this campaign made (`--out results/<sto>/...`)
        return own
    if cfg.targets_root and (Path(cfg.targets_root) / d.target).is_dir():
        return Path(cfg.targets_root) / d.target
    raise FileNotFoundError(f"no prediction directory for {d.target}: pass --dir or set targets_root in residudes.toml")


def _named_by_person(d: Dossier, root: Path) -> bool:
    """Whether a person pointed at this directory: in a note, a checkpoint reply or a resolution."""
    from residudes.dossier import AGENT_ACTORS
    machine = AGENT_ACTORS + ("residudes",)
    texts = [x["note"] for x in d.log if x["actor"] not in machine]
    for c in d.checkpoints:
        texts += [m.get("text", "") for m in c.get("thread", []) if m.get("by") not in machine]
        if c.get("resolution") and c["resolution"].get("by") not in machine:
            texts.append(c["resolution"].get("note", ""))
    keys = {str(root).rstrip("/"), str(root.resolve()).rstrip("/")}
    return any(k and k in t for t in texts for k in keys)


def pool_source_allowed(cfg: Config, d: Dossier, root: Path, actor: str) -> str | None:
    """Why the agent may not import this directory, or None. Its own runs (the dossier directory), the configured
    targets_root and a directory a person named are its to import; predictions it found elsewhere are not. On a clean
    run an agent imported 840 team models it found beside the FASTA it was given (T1431), another the team's
    257,800-model June pool (T1364), and neither campaign was then its own predictions. A person decides that."""
    from residudes.dossier import AGENT_ACTORS
    if actor not in AGENT_ACTORS:
        return None
    r = root.resolve()
    allowed = [Dossier.path(cfg.targets_dir, d.target).parent.resolve()]
    if cfg.targets_root:
        allowed.append(Path(cfg.targets_root).resolve())
    if any(r == a or a in r.parents for a in allowed) or _named_by_person(d, root):
        return None
    return (f"{root} holds predictions this campaign did not make, outside targets_root "
            f"({cfg.targets_root or 'none configured'}): importing them is a person's call. Record what you found "
            f"(`residudes evidence {d.target} existing_pool '{{\"dir\": \"{r}\", ...}}'`) and go on with your own runs; "
            f"a person who wants them imported names the directory in a note (`residudes note {d.target} "
            f"\"import the pool at {r}\"`) or sets targets_root in residudes.toml")


def pool_cmd(cfg: Config, d: Dossier, explicit: str | None, actor: str, refresh: bool = False, threads: int = 8) -> str:
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    if (ddir / "pool.csv").exists() and d.evidence.get("pool") and not refresh:
        last = d.evidence["pool"][-1]["value"]
        lines = [f"pool already imported: {last['n_models']} models from {last['dir']} -> {last['csv']} (--refresh to re-read)"]
        for k, v in last["by_sto_variant_runner"].items():
            lines.append(f"  {k:28s} n={v['n']:5d}  rs median {v['ranking_score_median']}  max {v['ranking_score_max']}  ipTM max {v['iptm_max']}")
        return "\n".join(lines)
    root = target_dir(cfg, d, explicit)
    why = pool_source_allowed(cfg, d, root, actor)
    if why:
        raise PermissionError(why)
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


def slurm_flags(gpu: bool, slurm: dict | None = None) -> list[str]:
    """Where residudes' own jobs go, from residudes.toml [slurm]: `partition` (default: the cluster's default
    partition), `gres` for GPU jobs (default gpu:1; e.g. gpu:RTX4090:1 to match the engines' card) and `extra` sbatch
    flags (e.g. ["--exclude=login1"]). It used to be `-p normal` for every site - this cluster's name for it."""
    if slurm is None:
        from residudes.config import load
        try:
            slurm = (load().extra or {}).get("slurm", {}) or {}
        except Exception:
            slurm = {}
    out = ["-p", str(slurm["partition"])] if slurm.get("partition") else []
    if gpu:
        out.append(f"--gres={slurm.get('gres') or 'gpu:1'}")
    return out + [str(x) for x in slurm.get("extra", []) or []]


def _sbatch(cmd: str, name: str, log: Path, gpu: bool, hours: int, after: list[str] | None = None, cpus: int = 8,
            wait: bool = False) -> str:
    """Submit `cmd` and return the job id. With wait, block until the job ends (sbatch --wait) and raise if it failed:
    for work the caller needs before it can go on, such as the final deck's pictures."""
    if not shutil.which("sbatch"):
        raise FileNotFoundError("sbatch not found: this needs a Slurm host (gates also run here with --local, up to 100 models)")
    args = ["sbatch", "--parsable", *slurm_flags(gpu), "-c", str(cpus), "-t", f"{hours:02d}:00:00", "-J", name, "-o", str(log)]
    if after:
        args += [f"--dependency=afterok:{':'.join(after)}"]
    if wait:
        args += ["--wait"]
    r = subprocess.run(args + ["--wrap", cmd], capture_output=True, text=True,
                       timeout=hours * 3600 + 1800 if wait else None)     # the queue wait comes on top of the limit
    job = r.stdout.strip().split(";")[0].splitlines()[0] if r.stdout.strip() else ""
    if r.returncode != 0:
        raise ValueError(f"sbatch {'job ' + job + ' ' if job else ''}failed: {r.stderr.strip()[-500:]}"
                         + (f" (log: {str(log).replace('%j', job)})" if job else ""))
    return job


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
            "UNKNOWN", "done", "failed", "cancelled",
            "BLOCKED"}             # UNKNOWN = purged from sacct; BLOCKED = its dependency failed; the rest is active
REFRESHED = {"BLOCKED"}    # not active, but asked about again: re-pointed with `scontrol update dependency=`, it runs


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
    """One recorded run's state. A run may name several jobs (agents record a batch of seeds as one run, the ids
    one per line or comma-separated): it is active while any of them is, and otherwise the first that did not
    complete. Reading only the first id took a batch's state from one job."""
    ids = [x for x in re.split(r"[\s,]+", str(job).strip()) if x]
    if not ids:
        return None
    try:
        q = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %r"], capture_output=True, text=True, timeout=60)
        queued = [ln.split(None, 2) + ["", ""] for ln in q.stdout.splitlines() if ln.strip()] if q.returncode == 0 else []
        # A job chained (afterok) on one that failed can never start, and Slurm keeps it queued: counted as pending
        # it held a clean run's target "waiting on jobs" for good (120 predictions after a failed MSA job). It is
        # BLOCKED: not active, so the agent gets a step, and left queued - M1370's agent resubmitted the MSA job
        # and re-pointed all 120 at it with scontrol, which beats cancelling and resubmitting them.
        dead = [x[0] for x in queued if x[1] == "PENDING" and x[2].strip() == "DependencyNeverSatisfied"]
        alive = [x[1] for x in queued if x[0] not in dead]
        if alive:
            return "RUNNING" if "RUNNING" in alive else alive[0]
        if dead:
            return "BLOCKED"
        s = subprocess.run(["sacct", "-n", "-X", "-j", ",".join(ids), "-o", "State"], capture_output=True, text=True, timeout=60)
        if s.returncode != 0:
            return None
        states = [x.rstrip("+") for x in s.stdout.split() if x != "by" and not x.isdigit()]
        return next((x for x in states if x != "COMPLETED"), states[0] if states else "UNKNOWN")
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
        if tracked(r) and (active(r) or state_of(r) in REFRESHED):
            if sch == "slurm":
                r["state"] = _slurm_state(job) or r.get("state")
            else:
                pueue = _pueue_states() if pueue is None else pueue
                r["state"] = (pueue or {}).get(job, r.get("state") if r.get("state") not in (None, "submitted") else "pueue-unknown")
        ids = [x for x in re.split(r"[\s,]+", job.strip()) if x]
        shown = ids[0] + (f" +{len(ids) - 1}" if len(ids) > 1 else "") if ids else ""   # a batch: its first id and how many more
        lines.append(f"  {shown:>8s} {sch:6s} {str(r.get('state')):12s} {str(r.get('what')).splitlines()[0] if r.get('what') else ''}")
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
