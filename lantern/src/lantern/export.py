"""`lantern export T DIR`: copy what a target produced into a self-contained folder (examples/, reports).

  dossier.json            the whole dossier
  analysis.md             `lantern report`: the priors, the decisions, and which priors each decision rests on
  steps.json              one row per agent step (stage, result, cost, turns, log), parsed from the dossier log
  agent_logs/             the transcript of every agent step
  slate_v*.md|json        slate proposals
  retro.md|json           the retrospective — only with --with-retro: scores against a native that may still be under
                          embargo (CASP assessment natives are). By default everything recorded from the first native-derived
                          result on (notes, evidence, decisions, agent files and logs) is left out too.
  pool_summary.json       per stoichiometry x variant x runner counts and confidence (not the per-model pool.csv)
  report/                 the slide report (report.typ, report.pdf), its PyMOL scenes and pictures
  checkpoints/            the briefs of the plan / results checkpoints (the conversations are in analysis.md)
  agent_files/            what the agent wrote in the dossier directory (gate tables, scripts), files under 2 MB
  qa/                     lantern-qa summaries: cluster_summary.csv per run, top 50 rows of xgb_dockq.csv
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
from pathlib import Path

from lantern.dossier import Dossier

STEP_RX = re.compile(r"agent step: (?P<subtype>\w+), \$(?P<cost>[0-9.]+|None), turns (?P<turns>\d+|None), log (?P<log>\S+)")


def steps(d: Dossier) -> list[dict]:
    out, stage = [], "NEW"
    for x in d.log:
        m = STEP_RX.search(x["note"])
        if m and x["actor"] == "lantern":
            log = Path(m["log"]).name
            st = log.split("_", 1)[1].removesuffix(".log") if "_" in log else stage
            out.append({"step": len(out) + 1, "time": x["time"], "stage": st, "subtype": m["subtype"],
                        "cost_usd": round(float(m["cost"]), 4) if m["cost"] != "None" else 0.0,
                        "turns": int(m["turns"]) if m["turns"] != "None" else 0, "log": log})
    return out


NATIVE_EVIDENCE = ("retrospective", "native")


def native_cutoff(d: Dossier) -> dt.datetime | None:
    """When the first native-derived result entered the dossier: native evidence, a retrospective, or a retro job.
    Everything recorded from then on (notes, evidence, decisions, files, agent logs) may carry native scores."""
    times = [e["time"] for k in NATIVE_EVIDENCE for e in d.evidence.get(k, []) if e.get("time")]
    times += [r["time"] for r in d.runs if str(r.get("what", "")).startswith("retro:") and r.get("time")]
    times += [x["time"] for x in d.log if x["note"].startswith("scoring ") or "native released" in x["note"]]
    return min((dt.datetime.fromisoformat(t) for t in times), default=None)


def _before(t: str | None, cutoff: dt.datetime | None) -> bool:
    return cutoff is None or not t or dt.datetime.fromisoformat(t) < cutoff


def _public(d: Dossier) -> dict:
    """The dossier as it stood before anything was scored against the native."""
    out = json.loads(json.dumps(d.__dict__))
    cut = native_cutoff(d)
    out["evidence"] = {k: [e for e in v if _before(e.get("time"), cut)] for k, v in out["evidence"].items() if k not in NATIVE_EVIDENCE}
    out["evidence"] = {k: v for k, v in out["evidence"].items() if v}
    out["decisions"] = [dict(x, outcome=None) for x in out["decisions"] if _before(x.get("time"), cut)]
    out["runs"] = [r for r in out["runs"] if _before(r.get("time"), cut) and not str(r.get("what", "")).startswith("retro:")]
    out["log"] = [x for x in out["log"] if _before(x["time"], cut)]
    return out


def _log_time(name: str) -> dt.datetime | None:
    try:                                                  # agent_logs/20261121-185808_POOL_READY.log (local time)
        return dt.datetime.strptime(name[:15], "%Y%m%d-%H%M%S").astimezone()
    except ValueError:
        return None


def run(d: Dossier, ddir: Path, dest: Path, with_retro: bool = False) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    pub = d.__dict__ if with_retro else _public(d)
    cut = None if with_retro else native_cutoff(d)
    (dest / "dossier.json").write_text(json.dumps(pub, indent=1, ensure_ascii=False))
    from lantern import report
    (dest / "analysis.md").write_text(report.build(Dossier(**pub), with_retro, ddir))
    st = steps(Dossier(**pub))
    (dest / "steps.json").write_text(json.dumps(st, indent=1))
    n = 0
    if (ddir / "agent_logs").is_dir():
        (dest / "agent_logs").mkdir(exist_ok=True)
        for p in sorted((ddir / "agent_logs").glob("*.log")):
            t = _log_time(p.name)
            if cut and (t is None or t >= cut):
                continue
            shutil.copy2(p, dest / "agent_logs" / p.name)
            n += 1
    for pat in ("slate_v*.md", "slate_v*.json") + (("retro.md", "retro.json") if with_retro else ()):
        for p in sorted(ddir.glob(pat)):
            if cut and pat.startswith("slate") and dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() >= cut:
                continue
            shutil.copy2(p, dest / p.name)
    skip = {"dossier.json", "pool.csv", "analysis.md", "links.json"}      # lantern writes these; they are not the agent's files
    own = [p for p in sorted(ddir.iterdir()) if p.is_file() and p.name not in skip and not p.name.startswith(("slate_v", "retro", "."))
           and p.stat().st_size < 2_000_000
           and (cut is None or dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() < cut)]
    if own:
        (dest / "agent_files").mkdir(exist_ok=True)
        for p in own:
            shutil.copy2(p, dest / "agent_files" / p.name)
    rep = ddir / "report"
    if rep.is_dir():                                        # the slide report: typ, pdf, pictures, PyMOL scenes
        def before(p: Path) -> bool:
            return cut is None or dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() < cut
        built = (d.evidence.get("report") or [{}])[-1].get("value", {})
        agent_typ = rep / "agent.typ"
        # the deck is native-free when it was built without --with-retro and the agent's own slides predate the native
        deck_ok = with_retro or (not built.get("with_retro") and (not agent_typ.exists() or before(agent_typ)))
        files = [rep / "scenes.pml"] + sorted(rep.glob("figs/*.jpg")) \
            + [p for p in sorted(rep.glob("figs/*.png")) if not p.with_suffix(".jpg").exists()] \
            + [p for p in sorted(rep.glob("figs/*.svg")) if with_retro or before(p)]   # the analysis links these
        if deck_ok:
            files += [rep / "report.typ", rep / "report.pdf", agent_typ]
        files += [p for p in sorted(rep.glob("pml/*.pml")) if with_retro or before(p)]
        for p in files:
            if p.exists():
                (dest / "report" / p.relative_to(rep)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / "report" / p.relative_to(rep))
    for nd in [ddir / "notes"] if (ddir / "notes").is_dir() else []:    # the write-up, as its own document
        for q in sorted(nd.glob("*.md")):
            if cut is None or dt.datetime.fromtimestamp(q.stat().st_mtime).astimezone() < cut:
                (dest / "notes").mkdir(exist_ok=True)
                from lantern.report import relativise           # links written as dossier paths travel as relative
                (dest / "notes" / q.name).write_text(relativise(q.read_text(), ddir).replace("](report/", "](../report/"))
    cps = sorted((ddir / "checkpoints").glob("*.md")) if (ddir / "checkpoints").is_dir() else []
    cps = [p for p in cps if cut is None or dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() < cut]
    if cps:
        (dest / "checkpoints").mkdir(exist_ok=True)
        for p in cps:
            shutil.copy2(p, dest / "checkpoints" / p.name)
    if d.evidence.get("pool"):
        (dest / "pool_summary.json").write_text(json.dumps(d.evidence["pool"][-1]["value"], indent=1))
    qa = ddir / "qa"
    if qa.is_dir():
        (dest / "qa").mkdir(exist_ok=True)
        for p in sorted(qa.rglob("cluster_summary.csv")):
            shutil.copy2(p, dest / "qa" / f"{p.relative_to(qa).parts[0]}_cluster_summary.csv")
        for p in sorted(qa.rglob("xgb_dockq.csv")):
            with open(p) as fh:
                rows = list(csv.reader(fh))[:51]
            with open(dest / "qa" / f"{p.relative_to(qa).parts[0]}_xgb_dockq_top50.csv", "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
        for p in sorted(qa.rglob("*clustering.csv")):       # the four columns the clustering figure is drawn from,
            with open(p) as fh:                             # so the picture in this folder can be checked against
                rows = list(csv.DictReader(fh))             # the numbers it was drawn from
            keep = [c for c in ("cluster", "structure", "umap1", "umap2") if rows and c in rows[0]]
            if not keep:
                continue
            parts = p.relative_to(qa).parts         # cluster_<sto>/<sto>/<run>/<method>/results/<file>
            name = f"{parts[0]}_{parts[-4]}" if len(parts) >= 4 else p.stem
            with open(dest / "qa" / f"{name}_clustering.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    gates = ddir / "gates"
    if gates.is_dir():                                   # specs, drafts and summaries; the per-model CSVs stay behind
        files = [p for p in sorted(gates.glob("*.json")) if not (cut and dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone() >= cut)]
        if files:
            (dest / "gates").mkdir(exist_ok=True)
            for p in files:
                shutil.copy2(p, dest / "gates" / p.name)
    total = sum(s["cost_usd"] for s in st)
    return (f"exported {d.target} -> {dest}: {len(st)} agent steps (${total:.2f}), {n} logs"
            + ("" if with_retro else "; native-derived results left out (--with-retro to include)"))
