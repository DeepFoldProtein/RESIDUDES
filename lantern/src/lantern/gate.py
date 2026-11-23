"""`lantern gate T --spec gates.json --sto S`: structural gates and a site census over a prediction pool.

In CASP17, model 1 was chosen by gates first and confidence second, and that beat every confidence rule on the same
pool (37 complexes: mean DockQ 0.651, against 0.564 for the highest ranking_score over all predictors and 0.589 for
af3's top ipTM). The gates were knowledge the agent had collected before any model existed:
  H2321  the paratope within 6 A of S611/R615 (2A2 escape mutants), the spike core within 1.0 A of apo spike 8UFO,
         the spike dimer interface intact, Fv folded with VH-VL paired, no clashes   -> DockQ 0.85 (top ranking_score 0.59)
  M2415  a pocket census of 815 ligand poses: the 5'ss bulge 444, the Sm site 186, U1-70K RRM 110, other 75; the
         literature puts the splicing modifier at the bulge -> model 1 = best ranking_score inside the bulge, ligand
         8.8 A from the released pose (the overall top ranking_score: 51.8 A)
This module makes those gates a tool: one spec per target, evaluated on every model of a stoichiometry, a CSV the
slate reads (`lantern slate` keeps only the models that pass), and a summary recorded as evidence `gates`.

Spec (JSON). Residue numbers are input-sequence positions (1..N): every predictor's model.cif in a lantern-engine pool
uses the same chain ids (A, B, ... in input order) and numbering.
{
  "gates": [
    {"name": "core",     "type": "rmsd",      "select": "A:1-500,B:1-500", "ref": "8ufo.cif",
                         "ref_chains": {"A": "A", "B": "B"}, "max": 1.0},
    {"name": "dimer",    "type": "interface", "a": "A", "b": "B", "min_contacts": 20},
    {"name": "dimer_ref","type": "interface", "a": "A", "b": "B", "ref": "8ufo.cif",
                         "ref_chains": {"A": "A", "B": "B"}, "min_fraction": 0.6},
    {"name": "epitope",  "type": "contact",   "site": "A:611,615", "partner": "C,D", "within": 6.0},
    {"name": "pocket",   "type": "contact",   "site": "B:1-12", "partner": "ligand", "within": 5.0},
    {"name": "clash",    "type": "clash",     "cutoff": 2.0, "max": 5}
  ],
  "census": {"probe": "ligand", "within": 5.0, "sites": {"bulge": "B:1-12", "sm": "D:20-40"}}
}
  select / site   "A:1-500,611,B:3-9" (a chain prefix holds until the next one); "A" alone is the whole chain
  partner / probe model chain ids "C,D", or "ligand" (every chain with no polymer residue), or "ligand:LIG" by residue name
  rmsd            CA (C4' for nucleic acids) after superposition; model and reference residues are paired by a
                  sequence alignment per chain, so the reference keeps its own numbering
  interface       residue pairs with heavy atoms within `cutoff` (default 5 A) between the groups a and b;
                  min_contacts, and/or min_fraction of the reference's pairs present
  contact         passes when the closest heavy atom of the partner is within `within` of the site
                  (and at least min_residues site residues are in contact, default 1)
  clash           inter-chain heavy-atom pairs closer than `cutoff` (default 2.0 A); passes when there are <= max (default 5)
  census          labels each model by the named site its probe contacts most ("other" when none), so a split
                  between sites shows as counts per predictor
Every model also gets n_chains, and the gate `complete` when the stoichiometry gives a chain count (A2B2C2 -> 6):
CASP17 M2377 had 310 models whose runs had lost the RNA chain.

    python -m lantern.gate run --spec S.json --pool pool.csv --sto A3 --out gates/default_A3.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

GATE_TYPES = ("rmsd", "interface", "contact", "clash", "symmetry", "stacking")


# ---------------------------------------------------------------------------------------------------- spec parsing
def parse_select(s: str) -> dict[str, set[int] | None]:
    """'A:1-500,611,B:3-9' -> {'A': {1..500, 611}, 'B': {3..9}};  'A,B' -> whole chains."""
    out: dict[str, set[int] | None] = {}
    chain = None
    for tok in [t.strip() for t in str(s).split(",") if t.strip()]:
        if ":" in tok:
            chain, _, tok = tok.partition(":")
            chain = chain.strip()
            out.setdefault(chain, set())
        elif not re.fullmatch(r"-?\d+(--?\d+)?", tok):
            out[tok] = None                                    # a bare chain id: the whole chain
            chain = None
            continue
        if chain is None:
            raise ValueError(f"selection {s!r}: residue {tok!r} has no chain (write A:{tok})")
        if out.get(chain) is None:
            out[chain] = set()
        m = re.fullmatch(r"(-?\d+)-(-?\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out[chain].update(range(min(a, b), max(a, b) + 1))
        else:
            out[chain].add(int(tok))
    return out


def parse_partner(s: str) -> tuple[set[str], bool, str | None, dict | None]:
    """(chains, is_ligand, ligand name, residues per chain or None):
    'H,L' -> ({'H','L'}, False, None, None);  'ligand' -> (set(), True, None, None);  'ligand:LIG' -> ligand by name;
    'A:7' or 'A:5-9,B' -> only those residues of A (and all of B) count, e.g. the phosphoserine of a peptide."""
    s = str(s).strip()
    if s.lower().startswith("ligand"):
        _, _, name = s.partition(":")
        return set(), True, (name.strip().upper() or None), None
    if ":" in s:
        sel = parse_select(s)
        return set(sel), False, None, sel
    return {c.strip() for c in s.split(",") if c.strip()}, False, None, None


def expected_chains(sto: str) -> int | None:
    """A2B2C2 -> 6;  A1@B (a per-chain run) -> 1;  anything else -> None."""
    base = sto.split("@")[0]
    if "@" in sto:
        return 1
    parts = re.findall(r"[A-Z](\d+)", base)
    return sum(int(n) for n in parts) if parts and re.fullmatch(r"([A-Z]\d+)+", base) else None


def check_spec(spec: dict) -> dict:
    gates = spec.get("gates", [])
    names = set()
    for g in gates:
        if g.get("type") not in GATE_TYPES:
            raise ValueError(f"gate {g.get('name')!r}: type must be one of {', '.join(GATE_TYPES)}")
        if not g.get("name") or not re.fullmatch(r"[A-Za-z0-9_]+", g["name"]) or g["name"] in names | {"complete", "census"}:
            raise ValueError(f"gate names must be unique words (letters, digits, _): {g.get('name')!r}")
        names.add(g["name"])
        need = {"rmsd": ("select", "ref", "max"), "interface": ("a", "b"), "contact": ("site", "partner", "within"), "clash": (),
                "symmetry": ("chains", "max"), "stacking": ()}[g["type"]]
        missing = [k for k in need if k not in g]
        if missing:
            raise ValueError(f"gate {g['name']!r} ({g['type']}) needs {', '.join(missing)}")
        if g["type"] == "interface" and not ("min_contacts" in g or ("ref" in g and "min_fraction" in g)):
            raise ValueError(f"gate {g['name']!r}: give min_contacts, or ref + min_fraction")
        for k in ("max", "within", "cutoff", "min_contacts", "min_fraction", "min_residues", "min_pairs"):
            if k in g and (isinstance(g[k], bool) or not isinstance(g[k], (int, float))):
                raise ValueError(f"gate {g['name']!r}: {k} must be a number, not {g[k]!r}")
        for k in ("select", "site"):
            if k in g:
                parse_select(g[k])
    c = spec.get("census")
    if c:
        if not c.get("sites") or "probe" not in c:
            raise ValueError("census needs probe and sites")
        for v in c["sites"].values():
            parse_select(v)
    return spec                                        # no gates at all still checks `complete`


# ---------------------------------------------------------------------------------------------------- structures
def _gemmi():
    import gemmi
    return gemmi


_POLY: dict[str, bool] = {}


def _is_poly(name: str) -> bool:
    if name not in _POLY:
        t = _gemmi().find_tabulated_residue(name)
        _POLY[name] = bool(t and (t.is_amino_acid() or t.is_nucleic_acid()))
    return _POLY[name]


def load(path: str):
    g = _gemmi()
    st = g.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_waters()
    return st


def polymer_chains(model) -> list[str]:
    return [ch.name for ch in model if any(_is_poly(r.name) for r in ch)]


def _one(name: str) -> str:
    t = _gemmi().find_tabulated_residue(name)
    c = t.one_letter_code.upper() if t else "X"
    return c if c.strip() and c != "?" else "X"


def _rep_atom(res):
    for n in ("CA", "C4'", "P"):
        a = res.find_atom(n, "*")
        if a:
            return a
    return None


_MAPS: dict = {}


def resmap(model_chain, ref_chain) -> dict[int, tuple]:
    """model residue number -> (model residue, the reference residue it aligns to), by a global alignment of the
    two chains' sequences (cached per sequence pair: every model of a pool has the same sequences)."""
    g = _gemmi()
    mres = [r for r in model_chain if _is_poly(r.name)]
    rres = [r for r in ref_chain if _is_poly(r.name)]
    mseq, rseq = "".join(_one(r.name) for r in mres), "".join(_one(r.name) for r in rres)
    if (mseq, rseq) not in _MAPS:
        al = g.align_string_sequences(list(mseq), list(rseq), [])
        ga, gb = al.add_gaps(mseq, 1), al.add_gaps(rseq, 2)
        pairs, i, j = [], 0, 0
        for a, b in zip(ga, gb):
            if a != "-" and b != "-":
                pairs.append((i, j))
            i += a != "-"
            j += b != "-"
        _MAPS[(mseq, rseq)] = pairs
    return {mres[i].seqid.num: (mres[i], rres[j]) for i, j in _MAPS[(mseq, rseq)]}


_REFS: dict = {}
_REFPAIRS: dict = {}


def _ref(path: str):
    if path not in _REFS:
        _REFS[path] = load(path)
    return _REFS[path]


def _ns(st, radius: float):
    g = _gemmi()
    return g.NeighborSearch(st[0], st.cell, max(radius, 1.0) + 0.5).populate()


def _contact_pairs(st, a: set[str], b: set[str], cutoff: float, ns=None) -> set[tuple]:
    g = _gemmi()
    cs = g.ContactSearch(cutoff)
    cs.ignore = g.ContactSearch.Ignore.SameChain
    out = set()
    for r in cs.find_contacts(ns or _ns(st, cutoff)):
        c1, c2 = r.partner1.chain.name, r.partner2.chain.name
        if c1 in a and c2 in b:
            out.add((c1, r.partner1.residue.seqid.num, c2, r.partner2.residue.seqid.num))
        elif c2 in a and c1 in b:
            out.add((c2, r.partner2.residue.seqid.num, c1, r.partner1.residue.seqid.num))
    return out


def _site_contacts(st, ns, site: dict, partner: tuple, within: float) -> tuple[float | None, int]:
    """(closest partner atom to the site, number of site residues within `within`)."""
    chains, ligand, lname, pres = partner
    model = st[0]
    lig_chains = {ch.name for ch in model} - set(polymer_chains(model)) if ligand else set()
    best, touched = None, 0
    for ch in model:
        if ch.name not in site:
            continue
        sel = site[ch.name]
        for res in ch:
            if sel is not None and res.seqid.num not in sel:
                continue
            hit = False
            for at in res:
                for mk in ns.find_atoms(at.pos, "\0", radius=within):
                    cra = mk.to_cra(model)
                    restricted = pres is not None and pres.get(cra.chain.name) is not None
                    if cra.chain.name == ch.name and not ligand and not restricted:
                        continue                               # the site's own chain counts only when named by residue
                    if ligand:
                        if cra.chain.name not in lig_chains or (lname and cra.residue.name.upper() != lname):
                            continue
                    elif cra.chain.name not in chains:
                        continue
                    elif restricted and cra.residue.seqid.num not in pres[cra.chain.name]:
                        continue
                    if cra.chain.name == ch.name and abs(cra.residue.seqid.num - res.seqid.num) <= 2:
                        continue                               # a residue and its chain neighbours are not a contact
                    d = at.pos.dist(cra.atom.pos)
                    if d <= within:
                        hit = True
                        best = d if best is None else min(best, d)
            touched += hit
    return best, touched


# ---------------------------------------------------------------------------------------------------- one model
def evaluate(path: str, spec: dict, n_expected: int | None) -> dict:
    g = _gemmi()
    out: dict = {}
    st = load(path)
    model = st[0]
    out["n_chains"] = len(polymer_chains(model))
    if n_expected:
        out["complete_pass"] = int(out["n_chains"] >= n_expected)
    radius = max([float(x.get(k, 0)) for x in spec.get("gates", []) for k in ("within", "cutoff")]
                 + [float((spec.get("census") or {}).get("within", 0)), 5.0])
    ns = _ns(st, radius)
    for gt in spec.get("gates", []):
        name, typ = gt["name"], gt["type"]
        val, ok = None, False
        if typ == "contact":
            val, n = _site_contacts(st, ns, parse_select(gt["site"]), parse_partner(gt["partner"]), float(gt["within"]))
            ok = val is not None and n >= float(gt.get("min_residues", 1))
            out[f"{name}_residues"] = n
        elif typ == "clash":
            cs = g.ContactSearch(float(gt.get("cutoff", 2.0)))
            cs.ignore = g.ContactSearch.Ignore.SameChain
            lig = {ch.name for ch in model} - set(polymer_chains(model))
            # a glycan or covalent ligand sits in its own chain, bonded at ~1.4-1.6 A: that is a bond, not a clash
            val = sum(1 for c in cs.find_contacts(ns)
                      if not ((c.partner1.chain.name in lig or c.partner2.chain.name in lig) and c.dist < 1.75))
            ok = val <= float(gt.get("max", 5))
        elif typ == "interface":
            pa, pb = parse_partner(gt["a"]), parse_partner(gt["b"])
            a, b = pa[0], pb[0]
            cutoff = float(gt.get("cutoff", 5.0))

            def keep(pr, sa=pa[3], sb=pb[3]):                  # residue selections in a / b (model numbering)
                ok_a = not sa or sa.get(pr[0]) is None or pr[1] in sa[pr[0]]
                ok_b = not sb or sb.get(pr[2]) is None or pr[3] in sb[pr[2]]
                return ok_a and ok_b
            pairs = {x for x in _contact_pairs(st, a, b, cutoff, ns) if keep(x)}
            val = len(pairs)
            ok = val >= float(gt.get("min_contacts", 0))
            if "ref" in gt:
                ref = _ref(gt["ref"])
                cmap = dict(gt.get("ref_chains") or {c: c for c in a | b})   # same ids unless mapped
                inv = {}
                for mc, rc in cmap.items():
                    if mc in [c.name for c in model] and rc in [c.name for c in ref[0]]:
                        for num, (_, rres) in resmap(model[mc], ref[0][rc]).items():
                            inv[(rc, rres.seqid.num)] = (mc, num)
                ra = {cmap[c] for c in a if c in cmap}
                rb = {cmap[c] for c in b if c in cmap}
                rkey = (gt["ref"], tuple(sorted(ra)), tuple(sorted(rb)), cutoff)
                if rkey not in _REFPAIRS:
                    _REFPAIRS[rkey] = _contact_pairs(ref, ra, rb, cutoff)
                rpairs = {(inv[(x[0], x[1])], inv[(x[2], x[3])]) for x in _REFPAIRS[rkey]
                          if (x[0], x[1]) in inv and (x[2], x[3]) in inv}
                rpairs = {r for r in rpairs if keep((r[0][0], r[0][1], r[1][0], r[1][1]))}
                mpairs = {((p[0], p[1]), (p[2], p[3])) for p in pairs}
                frac = len(rpairs & mpairs) / len(rpairs) if rpairs else 0.0
                out[f"{name}_fraction"] = round(frac, 3)
                ok = ok and frac >= float(gt["min_fraction"])
        elif typ == "stacking":
            # cross-beta: each chain's nearest neighbour (CA centroids) about 4.8 A away; helical lattices sit ~8-10 A
            want = [c.strip() for c in str(gt.get("chains", "")).split(",") if c.strip()] or polymer_chains(model)
            cents = []
            for c in want:
                if c in [x.name for x in model]:
                    pts = [a.pos for r in model[c] if _is_poly(r.name) for a in [_rep_atom(r)] if a]
                    if pts:
                        cents.append(g.Position(sum(q.x for q in pts) / len(pts), sum(q.y for q in pts) / len(pts),
                                                sum(q.z for q in pts) / len(pts)))
            lo, hi = float(gt.get("min", 4.4)), float(gt.get("max", 5.3))
            if len(cents) >= 2:
                nn = sorted(min(a.dist(b) for j, b in enumerate(cents) if j != i) for i, a in enumerate(cents))
                val = nn[len(nn) // 2]
                out[f"{name}_fraction"] = round(sum(lo <= x <= hi for x in nn) / len(nn), 3)
                ok = lo <= val <= hi
        elif typ == "symmetry":
            # Cn: the copies in ring order (A,B,C) superposed on the same copies shifted by one (B,C,A). A symmetric
            # ring maps onto itself (small RMSD); a ring that is open or uneven does not.
            ring = [c.strip() for c in str(gt["chains"]).split(",") if c.strip()]
            names = [c.name for c in model]
            if len(ring) >= 2 and all(c in names for c in ring):
                per = []
                for c in ring:
                    per.append({r.seqid.num: a.pos for r in model[c] if _is_poly(r.name) for a in [_rep_atom(r)] if a})
                common = sorted(set.intersection(*(set(x) for x in per)))
                P = [per[i][n] for i in range(len(ring)) for n in common]
                Q = [per[(i + 1) % len(ring)][n] for i in range(len(ring)) for n in common]
                out[f"{name}_pairs"] = len(common)
                if len(common) >= float(gt.get("min_pairs", 10)):
                    val = g.superpose_positions(P, Q).rmsd
                    val = None if val != val else val
                    ok = val is not None and val <= float(gt["max"])
        elif typ == "rmsd":
            ref = _ref(gt["ref"])
            sel = parse_select(gt["select"])
            cmap = gt.get("ref_chains") or {c: c for c in sel}
            P, Q = [], []
            for mc, nums in sel.items():
                rc = cmap.get(mc)
                if rc is None or mc not in [c.name for c in model] or rc not in [c.name for c in ref[0]]:
                    continue
                for num, (mres, rres) in resmap(model[mc], ref[0][rc]).items():
                    if nums is not None and num not in nums:
                        continue
                    ma, ra = _rep_atom(mres), _rep_atom(rres)
                    if ma and ra:
                        P.append(ma.pos)
                        Q.append(ra.pos)
            out[f"{name}_pairs"] = len(P)
            if len(P) >= float(gt.get("min_pairs", 10)):
                val = g.superpose_positions(Q, P).rmsd
                if val != val:                               # NaN: degenerate (collinear) positions
                    val = None
                ok = val is not None and val <= float(gt["max"])
        out[f"{name}_value"] = round(val, 3) if isinstance(val, float) else val
        out[f"{name}_pass"] = int(ok)
    c = spec.get("census")
    if c:
        partner = parse_partner(c["probe"])
        counts = {}
        for site, sel in c["sites"].items():
            _, n = _site_contacts(st, ns, parse_select(sel), partner, float(c.get("within", 5.0)))
            counts[site] = n
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("other", 0)
        out["census"] = top[0] if top[1] > 0 else "other"
    passes = [v for k, v in out.items() if k.endswith("_pass")]
    out["pass"] = int(all(passes)) if passes else 1
    return out


# ---------------------------------------------------------------------------------------------------- a pool
_SPEC: dict = {}


def _init(spec: dict, n_expected):
    _SPEC.update(spec=spec, n=n_expected)


def _one_row(row: dict) -> dict:
    try:
        res = evaluate(row["path"], _SPEC["spec"], _SPEC["n"])
    except Exception as e:                                   # a model that cannot be read fails every gate
        res = {"error": f"{type(e).__name__}: {e}"[:200], "pass": 0}
    return {**row, **res}


def run(spec: dict, pool_csv: Path, sto: str, out_csv: Path, workers: int = 1) -> dict:
    from lantern.slate import _model_key
    check_spec(spec)
    with open(pool_csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["sto"] == sto]
    if not rows:
        raise ValueError(f"no models for {sto} in {pool_csv}")
    base = [{"key": _model_key(r), "sto": r["sto"], "variant": r["variant"], "runner": r["runner"], "seed": r["seed"],
             "sample": r["sample"], "ranking_score": r["ranking_score"], "rs_z_runner": r.get("rs_z_runner", ""),
             "path": r["path"]} for r in rows]
    n_exp = expected_chains(sto)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(spec, n_exp)) as p:
            res = list(p.imap(_one_row, base, chunksize=8))
    else:
        _init(spec, n_exp)
        res = [_one_row(r) for r in base]
    cols = list(base[0].keys())
    for r in res:
        cols += [k for k in r if k not in cols]
    cols = [c for c in cols if c != "path"] + ["path"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(res)
    summary = summarize(res, spec, sto, n_exp)
    summary["csv"] = str(out_csv)
    out_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def summarize(res: list[dict], spec: dict, sto: str, n_exp: int | None) -> dict:
    names = (["complete"] if n_exp else []) + [x["name"] for x in spec.get("gates", [])]
    per_runner: dict = {}
    for r in res:
        pr = per_runner.setdefault(r["runner"], {"n": 0, "pass": 0, "best_pass": None})
        pr["n"] += 1
        if int(r.get("pass", 0)):
            pr["pass"] += 1
            try:
                rs = float(r["ranking_score"])
                if pr["best_pass"] is None or rs > pr["best_pass"][1]:
                    pr["best_pass"] = (r["key"], rs)
            except (TypeError, ValueError):
                pass
    census: dict = {}
    for r in res:
        if "census" in r:
            census.setdefault(r["runner"], Counter())[r["census"]] += 1
    incomplete = Counter(f"{r['variant']}/{r['runner']}" for r in res if str(r.get("complete_pass", "1")) == "0")
    return {"sto": sto, "n": len(res), "n_pass": sum(int(r.get("pass", 0)) for r in res),
            "errors": sum(1 for r in res if r.get("error")),
            "gates": {n: sum(int(r.get(f"{n}_pass", 0) or 0) for r in res) for n in names},
            "per_runner": per_runner,
            "census": {k: dict(v.most_common()) for k, v in census.items()},
            "incomplete": dict(incomplete), "expected_chains": n_exp,
            "spec": spec}


def load_pass(csv_path: Path, site: str | None = None) -> tuple[set[str], int]:
    """(keys of the models that pass every gate [and sit at `site` in the census], number of models gated)."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    keep = {r["key"] for r in rows if str(r.get("pass")) == "1" and (site is None or r.get("census") == site)}
    return keep, len(rows)


def failures(csv_path: Path) -> dict[str, list[str]]:
    """model key -> the gates it fails."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    return {r["key"]: [k[:-5] for k, v in r.items() if k.endswith("_pass") and str(v) == "0"] + (["error"] if r.get("error") else [])
            for r in rows}


# ---------------------------------------------------------------------------------------------------- commands
LOCAL_MAX = 100                     # more models than this go through sbatch (a login node is shared)


def _gdir(cfg, d) -> Path:
    from lantern.dossier import Dossier
    return Dossier.path(cfg.targets_dir, d.target).parent / "gates"


def gate_cmd(cfg, d, spec_path: str, sto: str, name: str, local: bool, workers: int, actor: str, hours: int = 4) -> str:
    """Check the spec, keep a copy in the dossier, and evaluate it on every model of `sto` (sbatch, or here when small)."""
    from lantern import ops
    from lantern.dossier import Dossier, now
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"--name {name!r}: letters, digits and _ only")
    pc = Dossier.path(cfg.targets_dir, d.target).parent / "pool.csv"
    if not pc.exists():
        raise FileNotFoundError(f"no pool.csv for {d.target}: run `lantern pool {d.target}` first")
    src = Path(spec_path).resolve()
    spec = check_spec(json.loads(src.read_text()))
    for g in spec.get("gates", []):                         # references resolve against the spec's folder
        if "ref" in g:
            ref = Path(g["ref"]) if Path(g["ref"]).is_absolute() else (src.parent / g["ref"]).resolve()
            if not ref.exists():
                raise FileNotFoundError(f"gate {g['name']}: reference {ref} not found")
            g["ref"] = str(ref)
    have = set(chain_ids(sto.split("@")[0])) if expected_chains(sto) else None
    if have:                                           # an id the models do not have would fail every model silently
        named = set()
        for g in spec.get("gates", []):
            for k in ("select", "site", "partner", "a", "b", "chains"):
                if k in g and not str(g[k]).lower().startswith("ligand"):
                    named |= set(parse_select(g[k])) if ":" in str(g[k]) else {c.strip() for c in str(g[k]).split(",") if c.strip()}
        c = spec.get("census")
        if c:
            for v in list(c.get("sites", {}).values()) + ([c["probe"]] if not str(c.get("probe", "")).lower().startswith("ligand") else []):
                named |= set(parse_select(v)) if ":" in str(v) else {x.strip() for x in str(v).split(",") if x.strip()}
        unknown = sorted(named - have)
        if unknown:
            raise ValueError(f"the spec names chain(s) {', '.join(unknown)}, but a {sto} model has {', '.join(sorted(have))} "
                             "(model chain ids: A, B, ... in input order, copies counted)")
    with open(pc) as fh:
        n = sum(1 for r in csv.DictReader(fh) if r["sto"] == sto)
    if not n:
        raise ValueError(f"no models for {sto} in pool.csv")
    gdir = _gdir(cfg, d)
    gdir.mkdir(parents=True, exist_ok=True)
    sp = gdir / f"{name}_{sto}.json"                         # per stoichiometry: a queued job keeps its own spec
    sp.write_text(json.dumps(spec, indent=1))
    out = gdir / f"{name}_{sto}.csv"
    if local:
        if n > LOCAL_MAX:
            raise ValueError(f"{n} models: more than {LOCAL_MAX}, so this goes through sbatch (drop --local)")
        run(spec, pc, sto, out, workers=min(workers, 4))
        d.note(actor, f"gates {name} evaluated on {n} {sto} models")
        return show_cmd(cfg, d, name, sto, actor)
    stamp = now().replace(":", "")
    moved = []
    for old_out in (out, out.with_suffix(".summary.json")):
        if old_out.exists():                           # not deleted: kept beside, so --show waits for the new result
            aside = old_out.with_name(old_out.name + f".replaced-{stamp}")
            old_out.rename(aside)
            moved.append((aside, old_out))
    cmd = (f"cd {Path(__file__).resolve().parents[1]} && {sys.executable} -m lantern.gate run --spec {sp} --pool {pc} "
           f"--sto {sto} --out {out} --workers {workers}")
    try:
        job = ops._sbatch(cmd, f"lantern-gate-{d.target}", gdir / f"{name}_{sto}.%j.log", gpu=False, hours=hours, cpus=workers)
    except Exception:
        for aside, orig in moved:                      # nothing was submitted: the earlier result stands
            aside.rename(orig)
        raise
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm", "what": f"lantern gate {name} {sto}",
                   "state": "submitted", "cmd": cmd, "out": str(out)})
    d.note(actor, f"submitted gates {name} on {n} {sto} models: job {job}")
    return (f"job {job}: gates {name} on {n} {sto} models -> {out}\n"
            f"when it finishes: `lantern gate {d.target} --show --name {name} --sto {sto}` (records evidence `gates`)")


def show_cmd(cfg, d, name: str, sto: str, actor: str) -> str:
    from lantern.dossier import now
    sj = _gdir(cfg, d) / f"{name}_{sto}.summary.json"
    if not sj.exists():
        return f"no result for gates {name} at {sto} yet ({sj}); `lantern jobs {d.target}` shows the job"
    s = json.loads(sj.read_text())
    spec = s.pop("spec", {})
    brief = [{k: v for k, v in g.items() if k != "ref"} | ({"ref": Path(g["ref"]).name} if "ref" in g else {})
             for g in spec.get("gates", [])]
    value = {"name": name, **s, "spec": {"gates": brief, **({"census": spec["census"]} if spec.get("census") else {})}}
    d.evidence.setdefault("gates", []).append({"time": now(), "by": actor, "value": value})
    lines = [f"gates {name} at {sto}: {s['n_pass']} of {s['n']} models pass every gate" + (f" ({s['errors']} unreadable)" if s["errors"] else ""),
             "", "  gate              pass"]
    lines += [f"  {g:16s} {n:6d}" for g, n in s["gates"].items()]
    lines += ["", "  runner   models   pass   best passing model (ranking_score)"]
    for r, v in sorted(s["per_runner"].items()):
        bp = f"{v['best_pass'][0]} ({v['best_pass'][1]:.3f})" if v["best_pass"] else "-"
        lines.append(f"  {r:8s} {v['n']:6d} {v['pass']:6d}   {bp}")
    if s.get("census"):
        lines += ["", f"census (probe {spec['census']['probe']}):"]
        lines += [f"  {r:8s} " + " · ".join(f"{k} {v}" for k, v in c.items()) for r, c in sorted(s["census"].items())]
    if s.get("incomplete"):
        lines += ["", f"models missing chains (expected {s['expected_chains']}): "
                  + ", ".join(f"{k} {v}" for k, v in s["incomplete"].items())]
    lines += ["", f"recorded as evidence `gates`; `lantern slate {d.target} --sto {sto}` now keeps only the passing models"]
    return "\n".join(lines)


SITE_KEYS = re.compile(r"epitope|paratope|binding[-_ ]?site|pocket|recognition|interface[-_ ]?prior|active[-_ ]?site|motif", re.I)
_SOURCE_KEY = re.compile(r"source|reference|^refs?$|paper|citation|pmid|pmc|doi", re.I)
_CITE = re.compile(r"\bPMID\b|\bPMC\d{4,}\b|\b10\.\d{4,}/|\bdoi\b|\bet al\b|\bbioRxiv\b|\bpreprint\b"
                   r"|\b[1-9](?=[A-Za-z0-9]{3}\b)(?=[0-9]*[A-Za-z])[A-Za-z0-9]{3}\b", re.I)   # PDB id: 8UFO, 7RK1

_SEL = re.compile(r"(?<![\w.])([A-Z]{1,2}):(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)(?![\w.])")
_RES3 = re.compile(r"\b(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)(\d{1,5})\b")
_RES1 = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{2,5})\b")
_ACCESSION = re.compile(r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")   # UniProt
_TAG = re.compile(r"\bHis\d+(?=[- ]?tag)|\b\d+x?His\b", re.I)                                       # His6-tag, 6xHis


def source_of(value) -> str | None:
    """Where a finding came from, as the evidence records it: a `source`-like field, or a citation in its own text
    (PMID, PMC, a DOI, "et al", bioRxiv, a PDB id). A site prior whose source is not recorded cannot be checked
    against the paper later, and the retrospective cannot score the prior that produced the gate."""
    if isinstance(value, dict):
        for k, v in value.items():
            if _SOURCE_KEY.search(k) and isinstance(v, (str, int, float)) and str(v).strip():
                return (str(v).strip() if k.lower() in ("source", "ref", "refs", "reference", "citation")
                        else f"{k}: {str(v).strip()}")[:120]
    m = _CITE.search(value if isinstance(value, str) else json.dumps(value))
    return m.group(0) if m else None


def _clean(text: str, skip: set[str]) -> str:
    """Text without what looks like a residue but is not: UniProt accessions, His tags, the target's own ids."""
    text = _TAG.sub(" ", _ACCESSION.sub(" ", text))
    for t in skip:
        text = re.sub(rf"\b{re.escape(t)}\b", " ", text)
    return text


def target_ids(target: str) -> set[str]:
    """H1388 -> {'H1388', 'H2388'} (and the other CASP stage's id), so a target's own id is never read as a residue."""
    m = re.match(r"([A-Z])([12])(\d{3})", target or "")
    return {target} | ({f"{m.group(1)}{d}{m.group(3)}" for d in "12"} if m else set())


def site_from_text(text: str, skip: set[str] = frozenset()) -> tuple[str | None, bool]:
    """A selection from free text, or (None, False) when the text names no residues reliably.
    - 'A:611,615' as written (upper-case chain ids, no spaces inside the list);
    - three-letter residues ('Ser548', 'Lys49 and Arg56');
    - one-letter residues only as a list of two or more ('S611/R615'), so that a lone token such as a target id
      (H1392), a CDR (H3), a stoichiometry (A24) or a subsite (S1) is not read as a residue.
    Returns (selection, chain_guessed): residues without a chain get chain A, flagged."""
    text = _clean(text, set(skip))
    m = _SEL.findall(text)
    if m:
        return ",".join(f"{c}:{r}" for c, r in m), False
    nums = {int(n) for n in _RES3.findall(text)}
    ones = _RES1.findall(text)
    if len({a + n for a, n in ones}) >= 2:
        nums |= {int(n) for _, n in ones}
    return (f"A:{','.join(map(str, sorted(nums)))}", True) if nums else (None, False)


def site_from_value(val, skip: set[str] = frozenset()) -> tuple[str | None, bool, str]:
    """A selection from a recorded site, preferring what was recorded as structure over what was written as prose.

    A site arrives either as a `select` string ("A:542,543") or as {chain, residues: [...]}, and both are the
    numbering the agent says it converted to. The prose beside them often quotes the native numbering as well -
    H1388 recorded `residues: [49, 56, 131, 132]` with "input numbering; native K50 R57 R132 Y133" in the same
    entry - so reading the text first picks the wrong frame and the gate lands one residue off. Returns
    (selection, chain_guessed, how it was read)."""
    if isinstance(val, dict):
        sel = next((v for k, v in val.items() if k.lower() in ("select", "selection") and isinstance(v, str) and ":" in v), None)
        if sel:
            return sel, False, "select as recorded"
        res = next((v for k, v in val.items()
                    if k.lower() in ("residues", "residue", "positions", "resnums") and isinstance(v, list)), None)
        nums = sorted({int(x) for x in (res or [])
                       if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())})
        if nums:
            chain = next((v for k, v in val.items()
                          if k.lower() in ("chain", "chains") and isinstance(v, str) and 1 <= len(v) <= 2), None)
            picked = ",".join(map(str, nums))
            return (f"{chain}:{picked}", False, "residues as recorded") if chain else (f"A:{picked}", True, "residues as recorded")
    text = val if isinstance(val, str) else json.dumps(val)
    sel, guessed = site_from_text(text, skip)
    return sel, guessed, "read from the text"


_THREE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
          "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
          "Tyr": "Y", "Val": "V"}


def residue_tokens(text: str, skip: set[str] = frozenset()) -> list[tuple[str, int]]:
    """(one-letter residue, number) for the residues site_from_text reads without a chain: Ser548 -> ('S', 548)."""
    text = _clean(text, set(skip))
    out = [(_THREE[m.group(0)[:3]], int(m.group(1))) for m in _RES3.finditer(text)]
    ones = [(a, int(n)) for a, n in _RES1.findall(text)]
    if len({(a, n) for a, n in ones}) >= 2:
        out += ones
    return out


def fasta_records(text: str) -> list[str]:
    """The sequences of a FASTA text, in order (entity A, B, ... of the stoichiometry)."""
    return [re.sub(r"[^A-Za-z]", "", body).upper() for _, body in re.findall(r">([^\n]*)\n([^>]*)", text or "")]


def entity_by_sequence(tokens: list[tuple[str, int]], seqs: list[str]) -> str | None:
    """The one entity whose sequence has every residue at its number (K49 R56 -> the chain with K at 49 and R at 56)."""
    if not tokens or not seqs:
        return None
    hits = [chr(65 + i) for i, sq in enumerate(seqs) if all(0 < n <= len(sq) and sq[n - 1] == aa for aa, n in tokens)]
    return hits[0] if len(hits) == 1 else None


def chain_ids(sto: str) -> list[str]:
    """A2B1 -> ['A', 'B', 'C']: the chain ids an engine model has, in input order (A, B, ..., Z, AA, AB, ...)."""
    n = expected_chains(sto) or 1
    return [chr(65 + i) if i < 26 else chr(65 + i // 26 - 1) + chr(65 + i % 26) for i in range(n)]


def entity_chains(sto: str) -> dict[str, list[str]]:
    """A2B1C1 -> {'A': ['A', 'B'], 'B': ['C'], 'C': ['D']}: which model chains are copies of which entity."""
    base = sto.split("@")[0]
    parts = re.findall(r"([A-Z])(\d+)", base) if re.fullmatch(r"([A-Z]\d+)+", base) else [("A", "1")]
    ids, out, i = chain_ids(sto), {}, 0
    for ent, n in parts:
        out[ent] = ids[i:i + int(n)]
        i += int(n)
    return out


def _to_chains(sel: str, ents: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Site selections are usually written per entity (the antigen, the heavy chain); the gate needs model chains.
    An entity letter whose first chain has another id is mapped to that first copy, and said so."""
    model_chains = {c for cs in ents.values() for c in cs}
    out, said = [], []
    for ch, nums in parse_select(sel).items():
        first = ents.get(ch, [ch])[0]
        if ch in model_chains:
            if first != ch:                            # B is both a model chain and an entity whose first chain differs
                said.append(f"{ch} kept as model chain {ch}; if it meant entity {ch}, use chain {first}")
            first = ch
        elif first != ch:
            said.append(f"{ch} read as entity {ch} -> its first chain {first}")
        out.append(f"{first}:{','.join(map(str, sorted(nums)))}" if nums else first)
    return ",".join(out), said


def draft_spec(d, sto: str) -> dict:
    """A first gate spec from what the dossier knows: a contact gate per site it can read and an assembly sanity gate
    for complexes. Every gate carries a "note" saying where it came from and what to check. Generic symmetry and
    clash gates are not drafted: on CASP17's released natives they removed good models (see the notes they leave)."""
    chains, ents = chain_ids(sto), entity_chains(sto)
    entity_of = {c: e for e, cs in ents.items() for c in cs}
    has_ligand = any(re.search(r"(?<![a-z])(ligand|ccd|smiles|cofactor|metal|ions?)(?![a-z])", k, re.I) for k in d.evidence)
    gates, notes, names = [], [], set()
    for key in sorted(k for k in d.evidence if SITE_KEYS.search(k)):
        if not d.evidence[key]:
            continue
        e = d.evidence[key][-1]
        text = e.get("value") if isinstance(e.get("value"), str) else json.dumps(e.get("value"))
        skip = target_ids(d.target)
        sel, guessed, how_read = site_from_value(e.get("value"), skip)
        if not sel:
            notes.append(f"evidence {key} names no residues I can read: {text[:120]}")
            continue
        how = ""
        if guessed:                                    # residues without a chain: which entity has them?
            nums = sel.split(":", 1)[1]
            ent = entity_by_sequence(residue_tokens(text, skip), fasta_records(d.sequence.get("text", "")))
            hint = re.findall(r"\bchain ([A-Z])\b", text)
            if ent:
                sel, guessed, how = f"{ent}:{nums}", False, f"; entity {ent} chosen because its sequence has these residues"
            elif len(set(hint)) == 1:
                sel, guessed, how = f"{hint[0]}:{nums}", False, f"; chain {hint[0]} as the text says"
        sel, mapped = _to_chains(sel, ents)
        unknown = sorted(set(parse_select(sel)) - set(chains))
        if unknown:
            notes.append(f"evidence {key} names chain(s) {', '.join(unknown)} that a {sto} model does not have "
                         f"(chains {', '.join(chains)}): {text[:100]}")
            continue
        site_ents = {entity_of.get(c, c) for c in parse_select(sel)}
        partner = [c for c in chains if entity_of.get(c) not in site_ents]       # other proteins, not other copies
        if not partner:
            if has_ligand:
                partner = ["ligand"]
            else:
                notes.append(f"evidence {key} ({sel}) has no other entity to contact it: say what should (another "
                             "copy for an interface, a ligand for a pocket) and add the gate by hand")
                continue
        name = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_") or "site"
        base, k = name, 2
        while name in names | {"complete", "census"}:
            name, k = f"{base}_{k}", k + 1
        names.add(name)
        src = source_of(e.get("value"))
        gates.append({"name": name, "type": "contact", "site": sel, "partner": ",".join(partner), "within": 6.0,
                      "note": f"from evidence {key} ({how_read}): {text[:100]}" + how
                              + ("; chain guessed as A: check it" if guessed else "")
                              + "".join(f"; {m}" for m in mapped) + "; check the partner (chains, or ligand)"
                              + (f"; source {src}" if src else "; no source recorded: say which paper, structure or "
                                 "experiment puts the partner here, so the threshold can be checked")})
    rings = [(ent, cs) for ent, cs in ents.items() if len(cs) >= 3]
    if rings:                                          # not drafted: lattices, fibrils, cages and dihedral rings fail it
        notes.append("if the copies form one closed Cn ring, add a symmetry gate, e.g. "
                     + json.dumps({"name": f"symmetry_{rings[0][0]}", "type": "symmetry", "chains": ",".join(rings[0][1]), "max": 3.0}))
    if len(chains) > 1:
        gates.append({"name": "assembled", "type": "interface", "a": chains[0], "b": ",".join(chains[1:]), "min_contacts": 10,
                      "note": f"chain {chains[0]} touches the rest: a sanity gate; replace it with the interface you expect"})
    notes.append("no clash gate: relaxation fixes clashes at packaging, and on CASP17 pools the best models of some "
                 "targets clashed most (templated af3 models of H2340: 77-117 atom pairs under 1.5 A at DockQ 0.77). Add "
                 "one only when a clash means a wrong assembly, e.g. chains through each other")
    for key in ("public_structures", "templates"):
        if key in d.evidence:
            notes.append(f"evidence {key} may hold a reference for an rmsd gate (a rigid part) or an interface gate with "
                         "ref + min_fraction: add its file path if so")
    spec = {"gates": gates, "notes": notes + ["`complete` (every chain present) is always checked"]}
    return check_spec(spec)


def init_cmd(cfg, d, sto: str, actor: str) -> str:
    """Write a new draft; an existing draft (perhaps edited by hand) is never overwritten."""
    gdir = _gdir(cfg, d)
    gdir.mkdir(parents=True, exist_ok=True)
    spec = draft_spec(d, sto)
    out, k = gdir / f"draft_{sto}.json", 2
    while out.exists():
        out, k = gdir / f"draft_{sto}_{k}.json", k + 1
    out.write_text(json.dumps(spec, indent=1))
    d.note(actor, f"drafted gates for {sto}: {', '.join(g['name'] for g in spec['gates'])} -> {out}")
    return (json.dumps(spec, indent=1) + f"\n\nwrote {out}. Edit it (sites, partners, thresholds, references), then "
            f"`lantern gate {d.target} --spec {out} --sto {sto} --name <name>`")


def latest_results(d) -> list[dict]:
    """The most recent `gates` evidence per (gate set, stoichiometry), ordered by when each was last shown."""
    seen: dict = {}
    for e in d.evidence.get("gates", []):
        v = e.get("value") or {}
        seen.pop((v.get("name"), v.get("sto")), None)
        seen[(v.get("name"), v.get("sto"))] = v
    return list(seen.values())


def _num(x, fmt: str) -> str:
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return str(x)


def describe(g: dict) -> str:
    """One line per gate, for reports: what it checks and its threshold (defaults included)."""
    t = g.get("type")
    if t == "rmsd":
        return f"CA RMSD of {g.get('select')} to {g.get('ref', 'a reference')} ≤ {g.get('max')} Å"
    if t == "interface":
        parts = []
        if "min_contacts" in g:
            parts.append(f"≥ {g['min_contacts']} residue contacts")
        if "min_fraction" in g:
            parts.append(f"≥ {_num(g['min_fraction'], '.0%')} of {g.get('ref', 'the reference')}'s contacts")
        return f"interface {g.get('a')} : {g.get('b')} within {g.get('cutoff', 5.0)} Å, " + " and ".join(parts)
    if t == "contact":
        n = int(g.get("min_residues", 1))
        return f"{g.get('partner')} within {g.get('within')} Å of {g.get('site')}" + (f" (≥ {n} site residues)" if n > 1 else "")
    if t == "clash":
        return f"≤ {g.get('max', 5)} inter-chain atom pairs under {g.get('cutoff', 2.0)} Å"
    if t == "stacking":
        return (f"median nearest-neighbour chain spacing (CA centroids) of {g.get('chains') or 'all chains'} within "
                f"{g.get('min', 4.4)}–{g.get('max', 5.3)} Å (cross-β stacking)")
    if t == "symmetry":
        ring = str(g.get("chains", ""))
        return f"cyclic symmetry of {ring}: the ring onto itself shifted by one copy ≤ {g.get('max')} Å"
    return str(t or g.get("name") or "?")


def result_csv(ddir: Path, sto: str, name: str | None) -> Path | None:
    """The gate result for `sto`: the one named, or the most recent; None when there is none (or name is "none")."""
    if name == "none":
        return None
    gdir = ddir / "gates"
    if name:
        c = gdir / f"{name}_{sto}.csv"
        if not c.exists():
            raise FileNotFoundError(f"no gate result {c.name}: run `lantern gate T --spec ... --sto {sto} --name {name}`")
        return c
    found = sorted(gdir.glob(f"*_{sto}.csv"), key=lambda p: p.stat().st_mtime) if gdir.is_dir() else []
    found = [f for f in found if re.fullmatch(r"[A-Za-z0-9_]+", f.name[: -len(f"_{sto}.csv")] or "-")]
    return found[-1] if found else None


def slate_failures(ddir: Path, models: list[str]) -> dict[str, list[str]]:
    """model -> the gates it fails, from the most recent gate result of each model's stoichiometry."""
    out: dict = {}
    for sto in {m.split("__")[0] for m in models}:
        c = result_csv(ddir, sto, None)
        if c:
            f = failures(c)
            out.update({m: f[m] for m in models if f.get(m)})
    return out


def proposal_failures(ddir: Path, proposal: dict) -> dict[str, list[str]]:
    """The gate failures of a recorded slate proposal, judged by the gates it was made with: the gate_fail recorded
    for a chosen slate, else the gate sets named in its `gated` notes, else the most recent result per stoichiometry."""
    models = proposal.get("models", [])
    if proposal.get("chosen") and "gate_fail" in proposal:   # recorded when the slate was chosen (--models)
        return {m: g for m, g in (proposal.get("gate_fail") or {}).items() if m in models and g}
    gated = proposal.get("gated") or {}
    if gated:
        out: dict = {}
        for sto, note in gated.items():
            c = result_csv(ddir, sto, note.get("gates"))
            if c:
                f = failures(c)
                out.update({m: f[m] for m in models if m.startswith(sto + "__") and f.get(m)})
        return out
    return slate_failures(ddir, models)


def for_slate(ddir: Path, sto: str, name: str | None, site: str | None) -> tuple[set[str] | None, dict | None]:
    """The passing model keys for `sto` and a note for the slate, or (None, None) when there is no gate result."""
    c = result_csv(ddir, sto, name)
    if c is None:
        if site:
            raise ValueError("--site needs a gate result with a census (`lantern gate` with \"census\" in the spec)")
        return None, None
    keep, total = load_pass(c, site)
    if not keep:
        raise ValueError(f"no {sto} model passes {c.name}" + (f" at site {site}" if site else "")
                         + ": loosen a gate, or rerun with --gates none and say why in the selection decision")
    return keep, {"gates": c.name[: -len(f"_{sto}.csv")], "site": site, "pass": len(keep), "of": total}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lantern.gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--sto", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    s = run(json.loads(a.spec.read_text()), a.pool, a.sto, a.out, a.workers)
    print(json.dumps({k: v for k, v in s.items() if k != "spec"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
