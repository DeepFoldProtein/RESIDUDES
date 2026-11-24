"""`residudes retro T`: score the proposed / decided slate against the native structure.

Two sources of scores, used together:
  --native PDB   submits one sbatch job that runs OpenStructure compare-structures on each slate model
                 -> <dossier>/retro_native/<model>.json ; run `residudes retro T` again when it finishes
  --scores CSV   a table of pool models already scored against the native (pool-wide context: percentiles, field rank)

The scores CSV has one row per pool model scored against the native (e.g. by OpenStructure):
  identity columns  stoich (or sto), runner (or model), msa (or variant), seed, sample — or a `name` like T1417__A3__af3_msa__af3__s1_0
  metric columns    any of lddt, tm_score, dockq_wave, ics, qs_best, rank_m1 (field model-1 rank), n_field
Writes retro.md / retro.json next to the dossier, fills `outcome` of the selection decisions, and moves the
target to RETROSPECTIVE when it is SUBMITTED.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from residudes.dossier import Dossier, now
from residudes.slate import key as slate_key

METRICS = ["lddt", "tm_score", "dockq_wave", "ics", "qs_best", "rank_m1"]


def load_scores(path: Path) -> tuple[dict[str, dict], int | None]:
    """Keys are residudes.slate.key: <sto>__<variant>__<runner>__s<seed>_<sample>."""
    out, n_field = {}, None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            sto = r.get("stoich") or r.get("sto")
            runner, variant, seed, sample = r.get("runner") or r.get("model"), r.get("msa") or r.get("variant"), r.get("seed"), r.get("sample")
            if r.get("name"):
                parts = r["name"].split("__")               # <tdir>__<sto>__<msa>__<runner>__s<seed>_<sample>
                if len(parts) == 5 and parts[4].startswith("s") and "_" in parts[4]:
                    sto, variant, runner = parts[1], parts[2], parts[3]
                    seed, sample = parts[4][1:].split("_", 1)
            if not (sto and runner and variant and seed not in (None, "") and sample not in (None, "")):
                continue
            vals = {k: float(r[k]) for k in METRICS if r.get(k) not in (None, "", "nan")}
            out[slate_key(sto, variant, runner, seed, sample)] = vals
            if r.get("n_field"):
                n_field = int(float(r["n_field"]))
    return out, n_field


def _pct_better(vals: list[float], v: float, higher_is_better: bool = True) -> float:
    better = sum(1 for x in vals if (x > v if higher_is_better else x < v))
    return round(100 * better / len(vals), 1) if vals else float("nan")


# site default; `residudes.toml` [tools] ost_sif wins
OST_SIF = "/gpfs/deepfold/containers/openstructure.sif"


def _latest_slate(d: Dossier) -> dict:
    props = d.evidence.get("slate_proposal", [])
    if not props:
        raise ValueError(f"{d.target} has no slate proposal to score")
    return json.loads(Path(props[-1]["value"]["json"]).read_text())


def _native_chains(native: Path) -> int:
    chains = set()
    with open(native) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                chains.add(line[21])
            elif line.startswith("_atom_site."):
                return 2                                    # mmCIF: let OST decide, score as oligo
    return len(chains)


def score_native_cmd(cfg, d: Dossier, ddir: Path, native: Path, mode: str | None, actor: str) -> str:
    from residudes import ops
    _latest_slate(d)                                        # raises when there is no slate
    seen, models = set(), []                               # the models of every slate version, once each
    for pr in d.evidence["slate_proposal"]:
        for m in json.loads(Path(pr["value"]["json"]).read_text())["models"]:
            if m["model"] not in seen:
                seen.add(m["model"])
                models.append(m)
    native = native.resolve()
    if not native.exists():
        raise FileNotFoundError(f"native not found: {native}")
    mode = mode or ("mono" if _native_chains(native) == 1 else "oligo")
    flags = "--lddt --tm-score --rigid-scores" if mode == "mono" else "--lddt --qs-score --ics --ips --dockq --tm-score --rigid-scores"
    out = ddir / "retro_native"
    out.mkdir(exist_ok=True)
    sif = cfg.extra.get("tools", {}).get("ost_sif", OST_SIF)
    paths = [native, out] + [Path(m["path"]) for m in models]
    binds = ",".join(sorted({"/" + p.resolve().parts[1] for p in paths}))
    rf = "mmcif" if native.suffix == ".cif" else "pdb"
    lines = ["#!/bin/bash", "set -u"]
    for m in models:
        mf = "mmcif" if m["path"].endswith(".cif") else "pdb"
        lines.append(f"apptainer run --bind {binds} {sif} compare-structures -m {m['path']} -mf {mf} -r {native} -rf {rf} "
                     f"-ft {flags} -o {out / (m['model'] + '.json')} > /dev/null")
    script = out / "score.sh"
    script.write_text("\n".join(lines) + "\n")
    job = ops._sbatch(f"bash {script}", f"residudes-retro-{d.target}", out / "score.%j.log", gpu=False, hours=2)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm", "what": f"retro: OST {mode} on slate models vs {native.name}",
                   "state": "submitted", "cmd": f"bash {script}", "out": str(out)})
    d.evidence.setdefault("native", []).append({"time": now(), "by": actor, "value": {"path": str(native), "mode": mode}})
    d.note(actor, f"scoring {len(models)} slate models against {native.name} ({mode}): job {job}")
    return f"job {job}: {len(models)} models of {len(d.evidence['slate_proposal'])} slate versions vs {native} ({mode}); run `residudes retro {d.target}` when it finishes"


def load_native_jsons(ddir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted((ddir / "retro_native").glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if j.get("status") != "SUCCESS":
            continue
        vals = {k: float(j[k]) for k in METRICS if isinstance(j.get(k), (int, float))}
        if "qs_best" not in vals and isinstance(j.get("qs_global"), (int, float)):
            vals["qs_best"] = float(j["qs_global"])
        out[p.stem] = vals
    return out


def _score_slate(slate: dict, sc: dict, direct: dict, pool: list[dict]) -> list[dict]:
    rows = []
    for m in slate["models"]:
        v = {**direct.get(m["model"], {}), **sc.get(m["model"], {})}      # the pool table wins (it has field ranks)
        src = "pool table" if m["model"] in sc else ("direct" if m["model"] in direct else "unscored")
        row = {"slot": m["slot"], "model": m["model"], "source": src, **{k: v.get(k) for k in METRICS}}
        if "lddt" in v and pool:
            row["lddt_pct_of_pool_better"] = _pct_better([p["lddt"] for p in pool if "lddt" in p], v["lddt"])
        rows.append(row)
    return rows


def _summary(label: str, rows: list[dict]) -> dict:
    m1 = rows[0]
    out = {"slate": label, "model1": m1["model"]}
    for k in ("lddt", "dockq_wave", "rank_m1"):
        out[f"model1_{k}"] = m1.get(k)
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[f"best5_{k}"] = (min(vals) if k == "rank_m1" else max(vals)) if vals else None
    return out


def _baselines(ddir: Path, sc: dict) -> list[dict]:
    """What plain confidence would have picked as model 1 (needs the pool-wide table)."""
    pc = ddir / "pool.csv"
    if not sc or not pc.exists():
        return []
    with open(pc) as fh:
        rows = [r for r in csv.DictReader(fh) if slate_key(r["sto"], r["variant"], r["runner"], r["seed"], r["sample"]) in sc]
    out = []
    for label, filt, col in (("highest ranking_score, any runner", lambda r: True, "ranking_score"),
                             ("highest ipTM, any runner", lambda r: r.get("iptm") not in ("", None) and r["sto"] != "A1", "iptm"),
                             ("highest af3 ranking_score", lambda r: r["runner"] == "af3", "ranking_score")):
        cand = [r for r in rows if filt(r) and r.get(col) not in ("", None)]
        if cand:
            b = max(cand, key=lambda r: float(r[col]))
            k = slate_key(b["sto"], b["variant"], b["runner"], b["seed"], b["sample"])
            out.append({"slate": f"baseline: {label}", "model1": k, **{f"model1_{x}": sc[k].get(x) for x in ("lddt", "dockq_wave", "rank_m1")}})
    return out


def _gate_audit(ddir: Path, sc: dict) -> list[dict]:
    """Did the gates keep the good models? Per gate result (gates/<name>_<sto>.csv) against the native scores:
    pass rate, how many of the top 10 % pass, median score passing vs failing, whether the best model passes, and
    the model the slate rules would lead with (af3's most confident) among the passing models vs among all."""
    gdir = ddir / "gates"
    if not sc or not gdir.is_dir():
        return []
    out = []
    for c in sorted(gdir.glob("*_*.csv")):
        with open(c) as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("key") in sc]
        if len(rows) < 5:
            continue
        metric = "dockq_wave" if any("dockq_wave" in sc[r["key"]] for r in rows) else "lddt"
        rows = [r for r in rows if metric in sc[r["key"]]]
        if len(rows) < 5:
            continue
        val = lambda r: sc[r["key"]][metric]
        from residudes.slate import interleave

        def lead(rs):                                   # the slate's order on confidence alone (no interface blend)
            return interleave([dict(r, score=float(r.get("rs_z_runner") or 0), runner=r.get("runner") or r["key"].split("__")[2],
                                    ranking_score=r.get("ranking_score") or 0) for r in rs])[0] if rs else None
        passing = [r for r in rows if str(r.get("pass")) == "1"]
        failing = [r for r in rows if str(r.get("pass")) != "1"]
        top = sorted(rows, key=val, reverse=True)[:max(1, len(rows) // 10)]
        med = lambda xs: sorted(val(r) for r in xs)[len(xs) // 2] if xs else None
        best = max(rows, key=val)
        pick_all = lead(rows)
        pick_pass = lead(passing)
        out.append({"gates": c.stem, "metric": metric, "n": len(rows), "n_pass": len(passing),
                    "top10_pass": round(sum(str(r.get("pass")) == "1" for r in top) / len(top), 3),
                    "median_pass": med(passing), "median_fail": med(failing),
                    "best": val(best), "best_passes": str(best.get("pass")) == "1",
                    "pick_all": val(pick_all), "pick_pass": val(pick_pass) if pick_pass else None,
                    "pick_pass_model": pick_pass["key"] if pick_pass else None})
    return out


def run(d: Dossier, ddir: Path, scores: Path | None, actor: str) -> str:
    sc, n_field = load_scores(scores) if scores else ({}, None)
    direct = load_native_jsons(ddir)
    if not sc and not direct:
        raise ValueError(f"no scores for {d.target}: pass --scores CSV, or --native PDB and wait for the job")
    props = d.evidence.get("slate_proposal", [])
    if not props:
        raise ValueError(f"{d.target} has no slate proposal to score")
    pool = list(sc.values())
    versions = []
    for pr in props:
        sl = json.loads(Path(pr["value"]["json"]).read_text())
        label = f"v{pr['value'].get('version')} ({'chosen by review' if sl.get('chosen') else 'rules'}, {pr.get('by', '?')})"
        versions.append((label, _score_slate(sl, sc, direct, pool)))
    label, slots = versions[-1]
    res = {"time": now(), "scores": str(scores) if scores else None, "direct": len(direct), "n_scored_pool": len(pool),
           "n_field": n_field, "slate": label, "slots": slots,
           "compare": [_summary(l, r) for l, r in versions] + _baselines(ddir, sc)}
    for k in ("lddt", "dockq_wave"):
        vals = sorted(p[k] for p in pool if k in p)
        if vals:
            res[f"pool_{k}_median"] = vals[len(vals) // 2]
            res[f"pool_{k}_best"] = vals[-1]
    ranks = [p["rank_m1"] for p in pool if "rank_m1" in p]
    if ranks:
        res["pool_best_field_rank"] = min(ranks)
    s1 = slots[0]
    best5 = max((r for r in slots if r.get("lddt") is not None), key=lambda r: r["lddt"], default=None)
    src = []
    if scores:
        src.append(f"{scores} ({len(pool)} pool models" + (f"; field of {n_field} groups)" if n_field else ")"))
    if direct:
        src.append(f"{len(direct)} slate models scored directly ({ddir / 'retro_native'})")
    f = lambda r, k, fmt="{:.3f}": fmt.format(r[k]) if r.get(k) is not None else "–"
    lines = [f"# retrospective — {d.target}", "", "Scored against the native: " + "; ".join(src), "",
             f"## slate {label}", "",
             "| slot | model | lDDT | DockQ | TM | field rank (model 1 scale) | pool models better (lDDT) | source |",
             "|---|---|---|---|---|---|---|---|"]
    for r in slots:
        pct = r.get("lddt_pct_of_pool_better")
        lines.append(f"| {r['slot']} | {r['model']} | {f(r, 'lddt')} | {f(r, 'dockq_wave')} | {f(r, 'tm_score')} | "
                     f"{f(r, 'rank_m1', '#{:.0f}')} | {str(pct) + '%' if pct is not None else '–'} | {r['source']} |")
    lines += ["", f"Pool: lDDT median {res.get('pool_lddt_median', '–')}, best {res.get('pool_lddt_best', '–')}; "
                  f"DockQ median {res.get('pool_dockq_wave_median', '–')}, best {res.get('pool_dockq_wave_best', '–')}; "
                  f"best field rank reachable from the pool #{res.get('pool_best_field_rank', '–')}."]
    lines += ["", "## every slate version, and what plain confidence would have picked", "",
              "| slate | model 1 | lDDT | DockQ | field rank | best of 5: lDDT | DockQ | field rank |", "|---|---|---|---|---|---|---|---|"]
    for c in res["compare"]:
        lines.append(f"| {c['slate']} | {c['model1']} | {f(c, 'model1_lddt')} | {f(c, 'model1_dockq_wave')} | "
                     f"{f(c, 'model1_rank_m1', '#{:.0f}')} | {f(c, 'best5_lddt')} | {f(c, 'best5_dockq_wave')} | {f(c, 'best5_rank_m1', '#{:.0f}')} |")
    res["gates"] = _gate_audit(ddir, sc)
    if res["gates"]:
        lines += ["", "## did the gates keep the good models?", "",
                  "Per gate result: the pass rate, how many of the best tenth pass, the median score of models that pass "
                  "and fail, whether the best model passes, and the lead pick in the slate's order on confidence alone "
                  "(af3 first; no interface score) among all models vs among those that pass.", "",
                  "| gates | metric | pass | best 10 % passing | median: pass / fail | best | best passes | lead pick: all → gated |",
                  "|---|---|---|---|---|---|---|---|"]
        for g in res["gates"]:
            fm = lambda x: "–" if x is None else f"{x:.3f}"
            lines.append(f"| {g['gates']} | {g['metric']} | {g['n_pass']}/{g['n']} | {g['top10_pass']:.0%} | "
                         f"{fm(g['median_pass'])} / {fm(g['median_fail'])} | {fm(g['best'])} | {'yes' if g['best_passes'] else 'no'} | "
                         f"{fm(g['pick_all'])} → {fm(g['pick_pass'])} |")
    (ddir / "retro.json").write_text(json.dumps(res, indent=1))
    (ddir / "retro.md").write_text("\n".join(lines) + "\n")
    for dec in d.decisions:
        if dec["type"] == "selection" and dec.get("outcome") is None:
            dec["outcome"] = {"slate": label, "model1": s1, "best_of_5_lddt": best5["lddt"] if best5 else None, "scored": now()}
    d.evidence.setdefault("retrospective", []).append({"time": now(), "by": actor, "value": {"md": str(ddir / "retro.md"), "slate": label,
                                                       "model1_lddt": s1.get("lddt"), "model1_field_rank": s1.get("rank_m1")}})
    if d.stage == "SUBMITTED":
        d.move("RETROSPECTIVE", actor, "native released and scored")
    return "\n".join(lines)
