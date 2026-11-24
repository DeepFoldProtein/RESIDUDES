"""`residudes slate T --sto A3 | A2:3,A1:2`: a deterministic first slate of five models from pool.csv (+ residudes-qa outputs).

A hedged slate splits the five slots between stoichiometries in the order given (A2:3,A1:2 -> slots 1-3 A2, 4-5 A1).
Models are named <sto>__<variant>__<runner>__s<seed>_<sample>, the CASP17 pool naming without the target.

The rules are the CASP17 selection lessons that can be applied without judgement:
  - order within a runner by its own confidence (robust z of ranking_score: median / MAD, with a floor); across runners
    alternate, af3 first: model 1 is af3's most confident model (see `interleave` for the CASP17 numbers)
  - when interface scores exist (residudes-qa interface: xgb_pred, a predicted DockQ), blend them in;
    they were reliable for homo-oligomers and not for antibody-antigen interfaces (flag: --no-interface)
  - when clusters exist (residudes-qa cluster), take at most one model per cluster for slots 1-3
  - slots 1-3 never hold two samples of the same seed (near-copies)
  - keep at least two runners in the slate, and at most two models from one runner (more only when there are too few runners to fill five)
The agent (or a person) then reviews the proposal against the structural gates of the target and edits it;
the final slate is recorded with `residudes decide --type selection`.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path


def _read_csv(p: Path) -> list[dict]:
    with open(p) as fh:
        return list(csv.DictReader(fh))


NAME_RX = re.compile(r"(?P<runner>.+)_seed(?P<seed>\d+)_sample(?P<sample>\d+)")


def _n(x) -> str:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)                                   # sample_best, seed_ref: kept as written


def key(sto: str, variant: str, runner: str, seed, sample) -> str:
    return f"{sto}__{variant}__{runner}__s{_n(seed)}_{_n(sample)}"


def _model_key(row: dict) -> str:
    return key(row["sto"], row["variant"], row["runner"], row["seed"], row["sample"])


def parse_spec(spec: str, n: int = 5) -> list[tuple[str, int]]:
    """'A3' -> [(A3, 5)];  'A2:3,A1:2' -> [(A2, 3), (A1, 2)]."""
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    if len(parts) == 1 and ":" not in parts[0]:
        return [(parts[0], n)]
    out = []
    for x in parts:
        sto, _, k = x.partition(":")
        if not k.isdigit():
            raise ValueError(f"slate spec {spec!r}: write A2:3,A1:2 (counts for every stoichiometry)")
        out.append((sto, int(k)))
    if sum(k for _, k in out) != n:
        raise ValueError(f"slate spec {spec!r}: the counts must add up to {n}")
    return out


def load_interface(qa_dir: Path, sto: str) -> dict[str, float]:
    """Predicted DockQ per model from residudes-qa interface outputs under qa/interface_<sto>*:
    run-ppi-v4 -> xgb_dockq.csv (model, msa, seed, sample, xgb_mean = mean over interfaces);
    legacy run-all -> xgb_scores.csv (model_id like 'af3_seed1_sample0__af3_msa.pdb', xgb_pred)."""
    out: dict[str, float] = {}
    for d in sorted(qa_dir.glob(f"interface_{sto}*")):
        if d.name.endswith("_ab") and not os.environ.get("RESIDUDES_USE_AB_INTERFACE"):
            continue                                   # antibody-antigen scores failed in CASP17 (H2442 rho -0.28)
        for p in sorted(d.rglob("xgb_dockq.csv")):
            for r in _read_csv(p):
                try:
                    out[key(sto, r["msa"], r["model"], r["seed"], r["sample"])] = float(r["xgb_mean"])
                except (KeyError, ValueError):
                    pass
        for p in sorted(d.rglob("xgb_scores.csv")):
            for r in _read_csv(p):
                name, _, variant = r.get("model_id", "").removesuffix(".pdb").removesuffix(".cif").partition("__")
                m = NAME_RX.fullmatch(name)
                try:
                    if m and variant:
                        out.setdefault(key(sto, variant, m["runner"], m["seed"], m["sample"]), float(r["xgb_pred"]))
                except (KeyError, ValueError):
                    pass
    return out


def load_clusters(qa_dir: Path, sto: str) -> dict[str, str]:
    """cluster id per model from residudes-qa cluster outputs under qa/cluster_<sto>*: <sto>/<variant>/<method>/results/
    *clustering.csv, structure names like 'af3_seed1_sample0'. Clusters are per variant, so ids are '<variant>:<id>'."""
    out: dict[str, str] = {}
    for d in sorted(qa_dir.glob(f"cluster_{sto}*")):
        for p in sorted(d.rglob("*clustering.csv")):
            rel = p.relative_to(d).parts                    # (<sto>, <variant>, <method>, results, file)
            variant = rel[1] if len(rel) >= 5 else "?"
            for r in _read_csv(p):
                name = (r.get("structure") or r.get("name") or "").split("__")[0].removesuffix(".pdb").removesuffix(".cif")
                cl = r.get("cluster") or r.get("cluster_id")
                m = NAME_RX.fullmatch(name)
                if m and cl is not None:
                    out[key(sto, variant, m["runner"], m["seed"], m["sample"])] = f"{variant}:{cl}"
    return out


def _candidates(all_rows: list[dict], sto: str, qa_dir: Path | None, use_interface: bool,
                allowed: set[str] | None = None) -> tuple[list[dict], bool, bool]:
    rows = [dict(r) for r in all_rows if r["sto"] == sto and r["rs_z_runner"] not in ("", None)]
    if allowed is not None:                            # structural gates (residudes gate): only the models that pass
        rows = [r for r in rows if _model_key(r) in allowed]
    if not rows:
        raise ValueError(f"no models for stoichiometry {sto} in the pool" + (" that pass the gates" if allowed is not None else ""))
    iface = load_interface(qa_dir, sto) if (qa_dir and use_interface) else {}
    clusters = load_clusters(qa_dir, sto) if qa_dir else {}
    vals = list(iface.values())
    m = sum(vals) / len(vals) if vals else 0.0
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 1.0
    for r in rows:
        r["key"] = _model_key(r)
        r["xgb_pred"] = iface.get(r["key"])
        r["cluster"] = clusters.get(r["key"])
        # blend: per-runner confidence z, plus the interface score scaled to the same spread when present
        score = float(r["rs_z_runner"])
        if r["xgb_pred"] is not None:
            score = 0.5 * score + 0.5 * (r["xgb_pred"] - m) / (sd or 1.0)
        r["score"] = round(score, 3)
    return interleave(rows), bool(iface), bool(clusters)


# The order runners lead and alternate in: each runner's own top-ranking_score model against CASP17's released natives
# (mean DockQ over 57 complexes: af3 0.536, bt2 0.493, bt2_jax 0.473, pt2 0.448, of3 0.397, ef2 0.369). Raw
# ranking_score is not comparable across runners, so it only orders runners missing from this list (after it).
RUNNER_PRIORITY = ("af3", "bt2", "bt2_jax", "pt2", "pt2_jax", "of3", "of3_jax", "ef2")
LEAD_RUNNER = RUNNER_PRIORITY[0]


def interleave(rows: list[dict]) -> list[dict]:
    """Best first, alternating runners: within a runner by its own score (z, blended with the interface score when
    there is one); across runners in RUNNER_PRIORITY order (af3 first), unknown runners after by best raw score. Over CASP17's released
    natives (53 complexes / 31 monomers) model 1 by af3's ranking_score averaged DockQ 0.535 / lDDT 0.783, by the raw
    ranking_score across runners 0.504 / 0.770, and by the per-runner z this slate used before 0.440 / 0.747: a z
    rewards standing out within one's own runner, not being right."""
    by: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: -r["score"]):
        by.setdefault(r["runner"], []).append(r)
    best_rs = {k: max(float(r["ranking_score"]) for r in v) for k, v in by.items()}
    order = sorted(by, key=lambda k: (RUNNER_PRIORITY.index(k) if k in RUNNER_PRIORITY else len(RUNNER_PRIORITY), -best_rs[k]))
    out, i = [], 0
    while any(i < len(by[k]) for k in order):
        out += [by[k][i] for k in order if i < len(by[k])]
        i += 1
    return out


def _pick(rows: list[dict], k: int, slate: list[dict], cap: int) -> None:
    """Append k models from rows (best first) to slate. The runner cap counts the whole slate (every part of a hedge);
    slots 1-3 of the whole slate get cluster and seed diversity."""
    start = len(slate)
    per_runner: dict = {}
    for x in slate:
        per_runner[x["runner"]] = per_runner.get(x["runner"], 0) + 1
    for strict, lim in ((True, cap), (False, cap), (False, 10 ** 9)):   # diversity, then without it; the cap goes last
        for r in rows:
            if len(slate) - start == k:
                break
            if any(r["key"] == x["key"] for x in slate) or per_runner.get(r["runner"], 0) >= lim:
                continue
            diverse = (r["cluster"] is None or r["cluster"] not in {x["cluster"] for x in slate}) and \
                (r["sto"], r["runner"], r["variant"], r["seed"]) not in {(x["sto"], x["runner"], x["variant"], x["seed"]) for x in slate}
            if strict and len(slate) < 3 and not diverse:
                continue                               # samples of one seed are near-copies
            why = [w for w, hit in (("runner cap", per_runner.get(r["runner"], 0) >= cap),
                                    ("cluster/seed diversity", len(slate) < 3 and not diverse)) if hit]
            if why:
                r["relaxed"] = ", ".join(why)
            slate.append(r)
            per_runner[r["runner"]] = per_runner.get(r["runner"], 0) + 1

def propose(pool_csv: Path, spec: str, qa_dir: Path | None = None, n: int = 5, use_interface: bool = True,
            allowed: dict[str, set[str]] | None = None) -> dict:
    parts = parse_spec(spec, n)
    all_rows = _read_csv(pool_csv)
    slate, used_iface, used_cl, n_cand = [], {}, {}, {}
    cands = {}
    for sto, k in parts:
        cands[sto], used_iface[sto], used_cl[sto] = _candidates(all_rows, sto, qa_dir, use_interface,
                                                                (allowed or {}).get(sto))
        n_cand[sto] = len(cands[sto])
    n_runners = len({r["runner"] for rows in cands.values() for r in rows})
    cap = max(2, -(-n // n_runners))                   # at most 2 per runner, unless there are too few runners to fill n
    for sto, k in parts:
        _pick(cands[sto], k, slate, cap)
    last_sto = parts[-1][0]
    if len(slate) > 1 and len({r["runner"] for r in slate}) < 2 and slate[-1]["sto"] == last_sto:
        other = next((r for r in cands[last_sto] if r["runner"] != slate[0]["runner"]), None)
        if other:                                      # a second runner in the last slot of the last part
            slate[-1] = other
    return {"sto": spec, "parts": [{"sto": s_, "n": k} for s_, k in parts], "n_candidates": n_cand, "short": n - len(slate),
            "used_interface": used_iface, "used_clusters": used_cl, "cap": cap,
            "models": [{"slot": i + 1, "model": r["key"], "sto": r["sto"], "runner": r["runner"], "variant": r["variant"],
                        "ranking_score": float(r["ranking_score"]), "iptm": float(r["iptm"]) if r["iptm"] not in ("", None) else None,
                        "rs_z_runner": float(r["rs_z_runner"]), "xgb_pred": r["xgb_pred"], "cluster": r["cluster"],
                        "score": r["score"], "path": r["path"], **({"relaxed": r["relaxed"]} if r.get("relaxed") else {})}
                       for i, r in enumerate(slate)]}


def from_models(pool_csv: Path, models: list[str], qa_dir: Path | None = None, use_interface: bool = True) -> dict:
    """A slate the agent or a person chose by hand (after reviewing a proposal): names as in `residudes slate` output."""
    all_rows = _read_csv(pool_csv)
    stos = sorted({m.split("__")[0] for m in models})
    by_key = {}
    for sto in stos:
        rows, _, _ = _candidates(all_rows, sto, qa_dir, use_interface)
        by_key.update({r["key"]: r for r in rows})
    missing = [m for m in models if m not in by_key]
    if missing:
        raise ValueError(f"not in pool.csv: {', '.join(missing)} (names look like A3__af3_msa__af3__s97_3)")
    if len(set(models)) != len(models):
        raise ValueError("a model is listed twice")
    picked = [by_key[m] for m in models]
    parts: list[dict] = []
    for r in picked:                                   # counts per stoichiometry (A2:2,A1:1); the slot order is the models'
        part = next((x for x in parts if x["sto"] == r["sto"]), None)   # own order, each named with its stoichiometry
        if part:
            part["n"] += 1
        else:
            parts.append({"sto": r["sto"], "n": 1})
    return {"sto": ",".join(f"{x['sto']}:{x['n']}" for x in parts) if len(parts) > 1 else parts[0]["sto"],
            "parts": parts, "chosen": True,
            "n_candidates": {x["sto"]: sum(1 for r in by_key.values() if r["sto"] == x["sto"]) for x in parts},
            "used_interface": {x["sto"]: any(r["xgb_pred"] is not None for r in by_key.values() if r["sto"] == x["sto"]) for x in parts},
            "used_clusters": {x["sto"]: any(r["cluster"] is not None for r in by_key.values() if r["sto"] == x["sto"]) for x in parts},
            "models": [{"slot": i + 1, "model": r["key"], "sto": r["sto"], "runner": r["runner"], "variant": r["variant"],
                        "ranking_score": float(r["ranking_score"]), "iptm": float(r["iptm"]) if r["iptm"] not in ("", None) else None,
                        "rs_z_runner": float(r["rs_z_runner"]), "xgb_pred": r["xgb_pred"], "cluster": r["cluster"],
                        "score": r["score"], "path": r["path"]} for i, r in enumerate(picked)]}


def write(slate: dict, out_dir: Path, version: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / f"slate_v{version}.json"
    j.write_text(json.dumps(slate, indent=1))
    md = out_dir / f"slate_v{version}.md"
    info = "; ".join(f"{p['sto']} x{p['n']}: {slate['n_candidates'][p['sto']]} candidates, interface scores "
                     f"{'used' if slate['used_interface'][p['sto']] else 'absent'}, clusters {'used' if slate['used_clusters'][p['sto']] else 'absent'}"
                     for p in slate["parts"])
    kind = "chosen by review" if slate.get("chosen") else "proposed by the rules"
    rule = ("" if slate.get("chosen") else
            "Rule: model 1 is af3's most confident candidate, then predictors alternate, each by its own confidence; "
            f"one model per cluster and per seed in slots 1-3, at most {slate.get('cap', 2)} per runner, while the candidates allow.")
    relaxed = [f"slot {m['slot']} ({m['relaxed']})" for m in slate["models"] if m.get("relaxed")]
    if relaxed and rule:
        rule += " Relaxed for want of candidates: " + ", ".join(relaxed) + "."
    gated = "; ".join(f"{sto}: {g['pass']} of {g['of']} models pass gates {g['gates']}" + (f" at site {g['site']}" if g.get("site") else "")
                      for sto, g in (slate.get("gated") or {}).items())
    if slate.get("gated") == {} and not slate.get("chosen"):
        gated = "no structural gates applied (`residudes gate`): confidence only"
    short = [f"only {len(slate['models'])} model{'s' if len(slate['models']) != 1 else ''} qualify for {len(slate['models']) + slate['short']} "
             "slots: fill the rest by judgement (`--models`), or loosen a gate and say why", ""] if slate.get("short") else []
    lines = [f"# slate v{version} — {slate['sto']} ({kind})", "", info, ""] + ([rule, ""] if rule else []) + ([gated, ""] if gated else []) + short + [
             "| slot | model | runner | ranking_score | ipTM | z (runner) | xgb_pred | cluster | score |", "|---|---|---|---|---|---|---|---|---|"]
    for m in slate["models"]:
        lines.append(f"| {m['slot']} | {m['model']} | {m['runner']} | {m['ranking_score']:.3f} | {m['iptm'] if m['iptm'] is not None else '–'} | "
                     f"{m['rs_z_runner']:+.2f} | {m['xgb_pred'] if m['xgb_pred'] is not None else '–'} | {m['cluster'] if m['cluster'] is not None else '–'} | {m['score']:+.2f} |")
    failing = [f"- slot {m['slot']} {m['model']} fails gates: {', '.join(m['gate_fail'])}" for m in slate["models"] if m.get("gate_fail")]
    if failing:
        lines += ["", "Models that fail the structural gates (say why in the selection decision):"] + failing
    md.write_text("\n".join(lines) + "\n")
    return j, md
