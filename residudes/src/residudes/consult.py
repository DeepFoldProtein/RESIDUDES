"""Checkpoints: where residudes stops to work something out with a person, and the first-results review.

Two checkpoints are part of every campaign (dossier stages PLAN_REVIEW and RESULTS_REVIEW):
  plan      after the target analysis and literature: the agent lays out what it found, the priors behind each
            decision and the prediction plan, asks its open questions (each with options and a recommendation),
            and talks it through with a person until they agree (`resolve --outcome proceed`) or send it back.
  results   after the first round: `residudes review T` runs the standard checks on the pool and QA; anything
            unusual (or anything the agent sees) is a trigger, and a trigger means a conversation with a person,
            who decides: proceed to the slate, another round of sampling (loop:RUNNING) or new hypotheses
            (loop:HYPOTHESES).

  residudes consult T --kind plan|results [--brief FILE] --ask "question (options; recommendation)" ...
  residudes reply T C1 "text"                      either side; the agent answers when a person wrote last
  residudes resolve T C1 --outcome proceed|loop:RUNNING|loop:HYPOTHESES [--note]     people only
  residudes checkpoints T [C1]                     list, or show one conversation
  residudes review T [--add "trigger: why" ...]    the first-results checks -> evidence results_review
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from residudes import report
from residudes import slate as slatemod
from residudes.gate import SITE_KEYS, source_of
from residudes.dossier import Dossier, now


def _ddir(cfg, d: Dossier) -> Path:
    return Dossier.path(cfg.targets_dir, d.target).parent


def plan_gaps(d: Dossier, ddir) -> list[str]:
    """What a plan checkpoint is missing, in the words of the thing that is missing.

    Not a block: a de novo design has no organism and a peptide has no MSA, and the agent is right to say so. But a
    person approving a plan should see which of the basics were never recorded, rather than infer it from a short
    document."""
    from pathlib import Path

    from residudes.gate import SITE_KEYS, site_from_value
    ev = d.evidence
    gaps: list[str] = []
    ident = (ev.get("identity") or [{}])[-1].get("value")
    flat = json.dumps(ident) if ident else ""
    # the organism is as often in the construct record as in the identity one, and either is fine
    described = flat + json.dumps((ev.get("construct") or [{}])[-1].get("value") or "")
    if not ident:
        gaps.append("no `identity`: what the chains are was never recorded")
    else:
        if not re.search(r"organism|species|taxon|strain|lineage|sapiens|coli|bacter|archae|algae|virus|yeast",
                         described, re.I):
            gaps.append("no organism in `identity` or `construct`")
        if not re.search(r"famil|fold|pfam|interpro|superfamily|de novo|design", flat, re.I):
            gaps.append("`identity` records no family or fold")
    if not (ev.get("msa_depth") or ev.get("msa")):
        gaps.append("no `msa_depth`: the MSAs were never measured")
    if not (ev.get("templates") or ev.get("public_structures")):
        gaps.append("no `templates` or `public_structures`: the structures that exist were never looked for")
    for key in sorted(k for k in ev if SITE_KEYS.search(k)):
        if ev[key] and site_from_value(ev[key][-1].get("value"))[2] == "read from the text":
            gaps.append(f"`{key}` has no residues field: a gate drafted from it is read out of the prose")
    if ddir and not sorted((Path(ddir) / "notes").glob("*.md")):
        gaps.append("no `notes/target.md`: there is no write-up for a person to read")
    return gaps


def open_cmd(cfg, d: Dossier, kind: str, asks: list[str], brief: str | None, actor: str) -> str:
    if not asks:
        raise ValueError("ask at least one question (--ask), each with its options and your recommendation")
    out = _ddir(cfg, d) / "checkpoints"
    out.mkdir(exist_ok=True)
    cid = f"C{len(d.checkpoints) + 1}"
    body = Path(brief).read_text() if brief else ""
    text = [f"# {d.target} — checkpoint {cid} ({kind})", "", f"Opened by {actor} at stage {d.stage}.", "",
            "## Questions", ""] + [f"{i + 1}. {q}" for i, q in enumerate(asks)] + [""]
    if body:
        text += ["## Brief", "", body.strip(), ""]
    gaps = plan_gaps(d, _ddir(cfg, d)) if kind == "plan" else []
    if gaps:
        text += ["## What the analysis did not record", "",
                 "Each of these may be right for this target - a design has no organism, a peptide has no MSA - but "
                 "the plan is being approved without them, so they are listed here rather than left to be noticed.",
                 ""] + [f"- {g}" for g in gaps] + [""]
    # the figures are drawn here, not only by `residudes report`: a checkpoint is where a person reads the analysis,
    # and a figure from an earlier state of the dossier is worse than none
    ddir = _ddir(cfg, d)
    from residudes import figures
    drawn, _ = figures.draw_all(d, ddir)
    if drawn:
        text += ["## The figures", "",
                 "Drawn from this dossier, each from the evidence it is about; every one is also a page of "
                 "`report/report.pdf`.", ""]
        text += [f"- [`{name}.svg`](../report/figs/{name}.svg) — {caption}"
                 for name, caption in figures.FIGURES if name in drawn] + [""]
    # the brief sits in checkpoints/, one level under the figures the analysis links
    text += ["## The analysis behind it", "",
             report.build(d, False, ddir).split("\n", 2)[2].replace("](report/", "](../report/")]
    path = out / f"{cid}_{kind}.md"
    path.write_text("\n".join(text))
    c = d.consult(actor, kind, asks, str(path))
    return (f"{c['id']} open ({kind}): {path}\n"
            + ("not recorded: " + "; ".join(gaps) + "\n" if gaps else "")
            + f"A person answers with `residudes reply {d.target} {c['id']} \"...\"` and decides with "
            + f"`residudes resolve {d.target} {c['id']} --outcome proceed|loop:RUNNING|loop:HYPOTHESES`.")


def show_cmd(d: Dossier, cid: str | None) -> str:
    if not cid:
        rows = [f"  {c['id']:4s} {c['kind']:8s} {c['status']:9s} at {c['stage']:15s} "
                + (f"-> {c['resolution']['outcome']}" if c["resolution"] else
                   ("agent's turn" if d.agents_turn(c) else "waiting for a person")) for c in d.checkpoints]
        return "\n".join(rows) or "no checkpoints"
    c = d.checkpoint(cid)
    lines = [f"{c['id']} ({c['kind']}) at {c['stage']}, opened {c['time']} by {c['by']}: {c['status']}", f"brief: {c['brief']}", "",
             "questions:"] + [f"  {i + 1}. {q}" for i, q in enumerate(c["questions"])] + ["", "conversation:"]
    lines += [f"  [{x['time'][11:16]}] {x['by']}: {x['text']}" for x in c["thread"]] or ["  (no replies yet)"]
    if c["resolution"]:
        r = c["resolution"]
        lines += ["", f"resolved by {r['by']} at {r['time']}: {r['outcome']}" + (f" ({r['note']})" if r["note"] else "")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- first-results review
def _copies(sto: str) -> int:
    return sum(int(n) for n in re.findall(r"[A-Z](\d+)", sto.split("@")[0])) or 1


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _chain_counts(by_run: dict) -> dict[str, tuple[int, int]]:
    """One model per stoichiometry x variant x runner: (polymer chains, expected). A run that lost a chain loses it in
    every model (CASP17 M2377: OpenFold3/Protenix with contact restraints, Boltz-2 JAX and OpenFold3 JAX ran without
    the RNA, 310 models), so one model per run is enough."""
    try:
        from residudes import gate
    except ImportError:
        return {}
    out = {}
    for (sto, var, run), rs in sorted(by_run.items()):
        exp = gate.expected_chains(sto)
        if not exp or not rs:
            continue
        try:
            n = len(gate.polymer_chains(gate.load(rs[0]["path"])[0]))
        except Exception:                                    # unreadable or absent: missing_models covers it
            continue
        out[f"{sto}/{var}/{run}"] = (n, exp)
    return out


def checks(pool_csv: Path, qa_dir: Path | None, gates_dir: Path | None = None) -> tuple[dict, list[str]]:
    """The standard checks on a first round. Returns (numbers, triggers)."""
    with open(pool_csv) as fh:
        rows = list(csv.DictReader(fh))
    by: dict = {}
    for r in rows:
        by.setdefault((r["sto"], r["runner"]), []).append(r)
    num, trig = {}, []
    multi = sorted({s for s, _ in by if _copies(s) > 1})
    mono = sorted({s for s, _ in by if _copies(s) == 1})

    # confidence of the best model per stoichiometry x runner
    best = {}
    for (sto, run), rs in by.items():
        key = "iptm" if _copies(sto) > 1 else "ptm"
        vals = [v for v in (_f(r[key]) for r in rs) if v is not None]
        best[(sto, run)] = max(vals) if vals else None
    num["best_confidence"] = {f"{s}/{r}": v for (s, r), v in sorted(best.items())}
    if best and all(v is None or v < 0.6 for v in best.values()):
        trig.append("low_confidence: no predictor has a model above 0.6 (ipTM for complexes, pTM for monomers)")

    # n-sweep: which copy number each runner prefers, and whether the sweep is flat
    if len(multi) >= 2:
        pref, flat = {}, []
        for run in sorted({r for _, r in by}):
            scored = sorted(((best.get((s, run)) or 0.0, s) for s in multi), reverse=True)
            if scored and scored[0][0] > 0:
                pref[run] = scored[0][1]
                if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
                    flat.append(run)
        num["preferred_stoichiometry"] = pref
        if len(set(pref.values())) > 1:
            trig.append("stoichiometry_split: predictors prefer different copy numbers " + ", ".join(f"{k} {v}" for k, v in pref.items()))
        if flat and len(flat) * 2 >= len(pref):
            trig.append(f"flat_nsweep: the top two copy numbers are within 0.05 ipTM for {', '.join(flat)}")

    # pLDDT and clashes of each runner's best models
    low, clash = [], []
    for (sto, run), rs in by.items():
        top = sorted(rs, key=lambda r: -(_f(r["ranking_score"]) or 0))[:10]
        pl = [v for v in (_f(r["plddt"]) for r in top) if v is not None]
        if pl and max(pl) < 70:
            low.append(f"{sto}/{run} {max(pl):.0f}")
        cl = [r for r in top if str(r.get("has_clash")).lower() in ("true", "1", "1.0")]
        if top and len(cl) * 2 > len(top):
            clash.append(f"{sto}/{run}")
    if low:
        trig.append("low_plddt: best pLDDT under 70 for " + ", ".join(low))
    if clash:
        trig.append("clashes: most top models clash for " + ", ".join(clash))

    # missing seeds: a runner far behind the others in the same cell
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["sto"], r["variant"]), {}).setdefault(r["runner"], 0)
        cells[(r["sto"], r["variant"])][r["runner"]] += 1
    short = [f"{s}/{v}/{run} {n}/{max(c.values())}" for (s, v), c in cells.items() for run, n in c.items() if n < 0.8 * max(c.values())]
    num["models_per_cell"] = {f"{s}/{v}": c for (s, v), c in sorted(cells.items())}
    if short:
        trig.append("missing_models: " + ", ".join(short[:8]) + (" ..." if len(short) > 8 else ""))

    # every run has every chain of its stoichiometry
    by_run: dict = {}
    for r in rows:
        by_run.setdefault((r["sto"], r["variant"], r["runner"]), []).append(r)
    cc = _chain_counts(by_run)
    lost = [f"{k} has {n} of {e} chains" for k, (n, e) in cc.items() if n < e]
    if cc:
        num["chains_per_run"] = {k: n for k, (n, _) in cc.items()}
    if lost:
        trig.append("missing_chains: " + ", ".join(lost[:8]) + (" ..." if len(lost) > 8 else "")
                    + " (leave these runs out of the slate and find out why)")

    # QA: architectures and interfaces
    if qa_dir and qa_dir.is_dir():
        for sto in multi + mono:
            cl = slatemod.load_clusters(qa_dir, sto)
            if cl:
                # labels are "<variant>:<cluster>", clustered per variant: compare each runner's best model *within* a
                # variant the runners share (the af3_msa input usually), not their overall tops from different variants
                tops: dict = {}
                for (s, run), rs in by.items():
                    if s != sto:
                        continue
                    for r in sorted(rs, key=lambda r: -(_f(r["ranking_score"]) or 0)):
                        k = slatemod.key(r["sto"], r["variant"], r["runner"], r["seed"], r["sample"])
                        if k in cl and run not in tops.setdefault(r["variant"], {}):
                            tops[r["variant"]][run] = cl[k]
                num[f"top_cluster_{sto}"] = tops
                split = {v: t for v, t in tops.items() if len(t) > 1 and len(set(t.values())) > 1}
                if split:
                    trig.append(f"architecture_split: at {sto} the predictors' best models sit in different clusters "
                                + "; ".join(f"{v}: " + ", ".join(f"{k} {c}" for k, c in t.items()) for v, t in split.items()))
            iface = slatemod.load_interface(qa_dir, sto)
            if iface:
                num[f"interface_best_{sto}"] = round(max(iface.values()), 3)
                if max(iface.values()) < 0.5:
                    trig.append(f"weak_interface: the best predicted interface DockQ at {sto} is {max(iface.values()):.2f}")
    # sites (residudes gate census): do the most confident models agree on where the partner sits?
    if gates_dir and gates_dir.is_dir():
        for sto in multi + mono:
            res = sorted(gates_dir.glob(f"*_{sto}.csv"), key=lambda p: p.stat().st_mtime)
            if not res:
                continue
            with open(res[-1]) as fh:
                gr = [r for r in csv.DictReader(fh) if r.get("census") and str(r.get("pass")) == "1"]   # the slate's candidates
            if not gr:
                continue
            census = {r["key"]: r["census"] for r in gr}
            try:                                       # the slate's own candidates and order (interface blend included)
                top = slatemod._candidates(rows, sto, qa_dir, True, set(census))[0][:10]
            except ValueError:
                continue
            labels: dict = {}
            for r in top:
                labels[census[r["key"]]] = labels.get(census[r["key"]], 0) + 1
            num[f"census_top10_{sto}"] = labels
            if sum(1 for v in labels.values() if v >= 2) >= 2:
                trig.append(f"site_split: at {sto} the slate's first ten candidates sit at different sites "
                            + ", ".join(f"{k} {v}" for k, v in labels.items())
                            + " (confidence will not settle it: decide by what is known about the site, not by more seeds)")
    return num, trig




def review_cmd(cfg, d: Dossier, add: list[str], actor: str) -> str:
    ddir = _ddir(cfg, d)
    pc = ddir / "pool.csv"
    if not pc.exists():
        raise FileNotFoundError(f"no pool.csv for {d.target}: run `residudes pool {d.target}` first")
    num, trig = checks(pc, ddir / "qa" if (ddir / "qa").exists() else None, ddir / "gates")
    pool_ev = (d.evidence.get("pool") or [{}])[-1].get("value", {}) if d.evidence.get("pool") else {}
    empty = (pool_ev.get("empty_models") or {}).get("n", 0)
    if empty:
        trig.append(f"empty_models: {empty} models have a summary but no model file (e.g. "
                    f"{pool_ev['empty_models']['first'][0]}); they were left out of the pool: find out why those runs failed")
    sites = sorted(k for k in d.evidence if SITE_KEYS.search(k))
    if sites and not d.evidence.get("gates"):
        trig.append(f"site_prior_without_gate: evidence {', '.join(sites)} names a site, but no structural gate uses it "
                    f"(`residudes gate {d.target} --spec ...`; CASP17 H2321 epitope gate: DockQ 0.59 -> 0.85)")
    unsourced = [k for k in sites if not source_of((d.evidence[k] or [{}])[-1].get("value"))]
    if unsourced:
        trig.append(f"site_without_source: evidence {', '.join(unsourced)} names a site but not where it comes from. "
                    "Record the paper, structure or experiment (escape mutants, mutagenesis, a crystal contact): a gate "
                    "whose source is written down can be checked against it, and the retrospective can score the prior")
    trig += [a for a in (add or []) if a.strip()]
    d.evidence.setdefault("results_review", []).append({"time": now(), "by": actor, "value": {"checks": num, "triggers": trig}})
    d.note(actor, f"results review: {len(trig)} trigger(s)" + (": " + " | ".join(t.split(':')[0] for t in trig) if trig else ", nothing unusual"))
    out = [f"results review for {d.target}: " + (f"{len(trig)} trigger(s)" if trig else "nothing unusual")]
    out += [f"  - {t}" for t in trig]
    out += ["", "next: " + (f"talk it through with a person: `residudes consult {d.target} --kind results --ask ...`" if trig
                            else f"`residudes slate {d.target} --sto ...`")]
    out += ["", json.dumps(num, indent=1)[:3000]]
    return "\n".join(out)
