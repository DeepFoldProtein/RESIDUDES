"""`residudes structures T`: the public-structure sweep the intake skill asks for on every target.

An exact-sequence search of the PDB per chain through RCSB's public search API, each hit with its
method, resolution and release date, and the age of the local template mirrors beside it — a
prediction only sees what its template database saw.

This is a step of its own, with its own evidence key (`public_structures`), because a template
search inside the predictor does not replace it. AlphaFold 3 filters near-identical templates out,
so a 100 % match is exactly what it will not show. The CASP17 sweep found 47 targets with a
near-identical public structure, and 8ARV, 7QV6, 3RIQ and 8YIV turned up only after the predictions
had been run.

    residudes structures T                      every chain, >= 90 % identity
    residudes structures T --identity 0.5       remote homologues too
    residudes structures T --foldseek model.cif the same question asked of a structure
    residudes structures T --hhsearch           and of a profile, when there is no model yet
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

from residudes.dossier import chain_letter

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
UA = {"User-Agent": "residudes-structures/1.0", "Content-Type": "application/json"}

NEAR_IDENTITY = 0.95        # identity at or above this, over most of the chain, is "the structure exists"
NEAR_COVERAGE = 0.90
TEMPLATE_COMPONENTS = ("PDB mmCIF (templates)", "foldseek PDB", "foldseek PDB 2026 (newer copy)")


# ----------------------------------------------------------------------------------- the target's chains

def chains(d) -> list[tuple[str, str, str]]:
    """[(letter, name, sequence)] from the dossier's FASTA, in file order: A, B, C …"""
    fasta = d.sequence.get("fasta")
    text = Path(fasta).read_text() if fasta and Path(fasta).exists() else d.sequence.get("text", "")
    if not text.strip():
        raise ValueError(f"{d.target}: no sequence in the dossier (residudes target add … --fasta)")
    recs = re.findall(r">([^\n]*)\n([^>]+)", text) if ">" in text else [("", text)]
    out = []
    for i, (header, body) in enumerate(recs):
        seq = re.sub(r"[^A-Za-z]", "", body).upper()
        if seq:
            out.append((chain_letter(i), header.strip().split()[0] if header.strip() else f"chain {i + 1}", seq))
    if not out:
        raise ValueError(f"{d.target}: no sequence in the dossier")
    return out


# ----------------------------------------------------------------------------------- RCSB (public, no key)

def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=UA, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def search(seq: str, identity: float = 0.9, rows: int = 100) -> list[dict]:
    """Polymer entities whose sequence matches this one, best first.

    identity/coverage come from the alignment RCSB reports, not from the request, so a hit that
    covers half the chain at 100 % is visible as such instead of passing as an exact match."""
    payload = {
        "query": {"type": "terminal", "service": "sequence",
                  "parameters": {"sequence_type": "protein", "value": seq,
                                 "identity_cutoff": identity, "evalue_cutoff": 1}},
        "request_options": {"scoring_strategy": "sequence", "results_verbosity": "verbose",
                            "paginate": {"start": 0, "rows": rows}},
        "return_type": "polymer_entity",
    }
    hits = []
    for r in _post(SEARCH_URL, payload).get("result_set") or []:
        ident = str(r.get("identifier", ""))
        ctx = {}
        for svc in r.get("services") or []:
            for node in svc.get("nodes") or []:
                for c in node.get("match_context") or []:
                    ctx = c
                    break
        qlen = ctx.get("query_length") or len(seq)
        alen = ctx.get("alignment_length") or 0
        hits.append({"entry": ident.split("_")[0].upper(), "entity": ident.partition("_")[2],
                     "identity": ctx.get("sequence_identity"), "evalue": ctx.get("evalue"),
                     "coverage": round(alen / qlen, 3) if qlen else None,
                     "query_range": [ctx.get("query_beg"), ctx.get("query_end")]})
    return hits


def entry_meta(entry: str) -> dict:
    """method, resolution, release date and title of one PDB entry; empty values if it cannot be read."""
    try:
        d = _get_json(ENTRY_URL.format(entry))
    except Exception:
        return {"method": None, "resolution": None, "released": None, "title": None}
    info = d.get("rcsb_entry_info") or {}
    res = (info.get("resolution_combined") or [None])[0]
    method = info.get("experimental_method") or ((d.get("exptl") or [{}])[0].get("method"))
    return {"method": method, "resolution": res,
            "released": ((d.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10] or None,
            "title": (d.get("struct") or {}).get("title")}


def is_near_identical(hit: dict) -> bool:
    return (hit.get("identity") or 0) >= NEAR_IDENTITY and (hit.get("coverage") or 0) >= NEAR_COVERAGE


# ----------------------------------------------------------------------------------- the sweep

def sweep(d, identity: float = 0.9, rows: int = 100, limit: int = 25, with_meta: bool = True) -> dict:
    """{chain letter: [hit, …]} for every chain, each hit annotated with its entry metadata."""
    out: dict[str, list[dict]] = {}
    seen: dict[str, dict] = {}
    for letter, _name, seq in chains(d):
        hits = search(seq, identity=identity, rows=rows)
        for h in hits[:limit] if with_meta else []:
            h.update(seen.get(h["entry"]) or seen.setdefault(h["entry"], entry_meta(h["entry"])))
        out[letter] = hits
    return out


def template_db_dates(cfg=None) -> list[dict]:
    """The local template mirrors and how old they are — what a template search would actually see."""
    from residudes import catalog
    try:
        rows = catalog.check_all(paths=(cfg.extra.get("databases") or {}) if cfg else {})
    except Exception:
        return []
    return [r for r in rows if r["name"] in TEMPLATE_COMPONENTS]


def sweep_cmd(cfg, d, actor: str, identity: float = 0.9, rows: int = 100, limit: int = 25,
              chain: str | None = None) -> str:
    from residudes.dossier import now
    found = sweep(d, identity=identity, rows=rows, limit=limit)
    if chain:
        found = {k: v for k, v in found.items() if k == chain.upper()}
        if not found:
            raise ValueError(f"{d.target}: no chain {chain} (chains: {', '.join(c[0] for c in chains(d))})")

    value = {k: [{kk: vv for kk, vv in h.items() if kk != "query_range"} for h in v[:limit]] for k, v in found.items()}
    near = {k: [h for h in v if is_near_identical(h)] for k, v in found.items()}
    d.evidence.setdefault("public_structures", []).append({
        "time": now(), "by": actor,
        "value": {"identity_cutoff": identity, "per_chain": value,
                  "near_identical": {k: [h["entry"] for h in v] for k, v in near.items() if v}}})

    lines: list[str] = []
    for letter, hits in found.items():
        name = next((n for ln, n, _ in chains(d) if ln == letter), letter)
        lines.append(f"chain {letter} ({name}): {len(hits)} entities at >= {identity:.0%} identity"
                     + (f", showing {limit}" if len(hits) > limit else ""))
        if not hits:
            lines.append("    none — no public structure of this sequence")
        lines.append(f"    {'entry':8s} {'ident':>6s} {'cover':>6s} {'res Å':>6s} {'released':10s} method / title")
        for h in hits[:limit]:
            mark = "*" if is_near_identical(h) else " "
            res = f"{h['resolution']:.2f}" if isinstance(h.get("resolution"), (int, float)) else "-"
            ident_s = f"{h['identity']:.0%}" if h.get("identity") is not None else "-"
            cov_s = f"{h['coverage']:.0%}" if h.get("coverage") is not None else "-"
            title = (h.get("title") or "")[:60]
            lines.append(f"  {mark} {h['entry']:8s} {ident_s:>6s} {cov_s:>6s} {res:>6s} "
                         f"{h.get('released') or '-':10s} {h.get('method') or '-'} {title}")
        lines.append("")

    hot = [f"{k}: {', '.join(h['entry'] for h in v[:10])}" + (f" (+{len(v) - 10} more)" if len(v) > 10 else "")
           for k, v in near.items() if v]
    if hot:
        lines.append(f"* near-identical ({NEAR_IDENTITY:.0%} identity over {NEAR_COVERAGE:.0%} of the chain) — "
                     f"{'; '.join(hot)}")
        lines.append("  AlphaFold 3 filters these out of its own template search, so pass them in as templates "
                     "deliberately, or say in the dossier why not.")
    else:
        lines.append("* no near-identical public structure for any chain at this cutoff")

    for r in template_db_dates(cfg):
        age = ""
        if r.get("date"):
            try:
                age = f", {(dt.date.today() - dt.date.fromisoformat(r['date'])).days} days old"
            except ValueError:
                age = ""
        lines.append(f"  template db: {r['name']} — {r['status']} ({r.get('date') or 'no date'}{age})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- foldseek: the structural half

FOLDSEEK_COLUMNS = ("query", "target", "fident", "alnlen", "qlen", "tlen", "evalue", "bits", "alntmscore", "prob")
CLOSE_TM = 0.80             # at or above this the fold is the same one, whatever the sequence says
_DB_PARTS = ("_ca", "_h", "_ss")
_ENTRY = re.compile(r"^(?:pdb)?([0-9][A-Za-z0-9]{3})\b")


def db_prefix(path: str | Path) -> Path:
    """A foldseek database is a path prefix, not a file. Accept either the prefix
    (…/pdb, with …/pdb.dbtype beside it) or a directory holding exactly one."""
    p = Path(path)
    if Path(str(p) + ".dbtype").exists():
        return p
    if p.is_dir():
        stems = {f.name[: -len(".dbtype")] for f in p.glob("*.dbtype")}
        found = sorted(s for s in stems if not s.endswith(_DB_PARTS))
        if len(found) == 1:
            return p / found[0]
        if not found:
            raise FileNotFoundError(f"no foldseek database in {p} (no *.dbtype)")
        raise ValueError(f"{p} holds several foldseek databases ({', '.join(found)}): name one with --db")
    raise FileNotFoundError(f"no foldseek database at {p}")


def foldseek_db(cfg, explicit: str | None = None) -> tuple[Path, str]:
    """(database prefix, where it came from). The newest configured database wins, because the
    whole point of the sweep is seeing what a 2024 mirror cannot (CASP17: 8ARV, 7QV6, 3RIQ)."""
    if explicit:
        return db_prefix(explicit), "--db"
    dbs = (cfg.extra.get("databases") if cfg else None) or {}
    for key in ("foldseek_pdb_2026", "foldseek_pdb"):
        if dbs.get(key):
            return db_prefix(dbs[key]), f"[databases] {key}"
    from residudes import catalog
    try:
        components = catalog.load(paths=dbs)
    except Exception:
        components = []
    for c in components:
        if c["name"] not in TEMPLATE_COMPONENTS or c.get("unset") or "foldseek" not in c["name"].lower():
            continue
        for p in (c.get("paths") or []) + (c.get("any_of") or []):
            if Path(p).exists():
                return db_prefix(p), f"catalog: {c['name']}"
    raise FileNotFoundError("no foldseek database configured: set [databases] foldseek_pdb_2026 in residudes.toml "
                            "(or pass --db). A 2024 mirror hides anything released since.")


def foldseek_cmd(cfg, d, model: str, actor: str, db: str | None = None, evalue: float = 1e-3) -> str:
    """Submit a structural search of one model against the database, through sbatch."""
    from residudes import ops
    from residudes.dossier import Dossier, now
    from residudes.msa import tool
    m = Path(model)
    if not m.exists():
        raise FileNotFoundError(f"no model at {m}")
    prefix, how = foldseek_db(cfg, db)
    exe = tool(cfg, "foldseek")
    if not exe:
        raise FileNotFoundError("foldseek not found: put it on PATH or set [tools] foldseek in residudes.toml")
    out = Dossier.path(cfg.targets_dir, d.target).parent / "structures"
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / f"foldseek_{m.stem}.tsv"
    cmd = (f"{exe} easy-search {m} {prefix} {tsv} {out}/tmp_{m.stem} "
           f"--format-output {','.join(FOLDSEEK_COLUMNS)} --alignment-type 1 -e {evalue}")
    job = ops._sbatch(cmd, f"residudes-foldseek-{d.target}", out / f"foldseek_{m.stem}.%j.log", gpu=False, hours=4)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm",
                   "what": f"foldseek easy-search {m.name}", "state": "submitted", "out": str(tsv)})
    d.note(actor, f"foldseek search of {m.name} against {prefix.name} ({how}): job {job}")
    return (f"foldseek easy-search {m.name} against {prefix} ({how})\njob {job} -> {tsv}\n"
            f"when it is done: `residudes structures {d.target} --read-foldseek {tsv}`")


def _entry_of(target: str) -> tuple[str, str | None]:
    """('6A5S', 'A') from the many spellings of a foldseek target: 6a5s_A, pdb6a5s.ent.gz_A, 6A5S.cif.gz_A."""
    name = Path(target).name
    chain = None
    m = re.search(r"_([A-Za-z0-9]{1,4})$", name)
    if m:
        chain, name = m.group(1), name[: m.start()]
    for suf in (".gz", ".cif", ".pdb", ".ent", ".mmcif"):
        while name.endswith(suf):
            name = name[: -len(suf)]
    e = _ENTRY.match(name)
    return (e.group(1).upper() if e else name.upper()), chain


def read_foldseek(tsv: str | Path, columns: tuple[str, ...] = FOLDSEEK_COLUMNS) -> list[dict]:
    """The rows of an easy-search table, best bits first, one per entry (its best chain)."""
    rows: list[dict] = []
    for line in Path(tsv).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(columns):
            continue
        r = dict(zip(columns, parts))
        entry, chain = _entry_of(r["target"])

        def _f(k: str) -> float | None:
            try:
                return float(r[k])
            except (KeyError, TypeError, ValueError):
                return None
        qlen = _f("qlen") or 0
        rows.append({"entry": entry, "chain": chain, "query": r["query"], "fident": _f("fident"),
                     "tm": _f("alntmscore"), "evalue": _f("evalue"), "bits": _f("bits"), "prob": _f("prob"),
                     "coverage": round((_f("alnlen") or 0) / qlen, 3) if qlen else None})
    rows.sort(key=lambda r: (-(r["bits"] or 0)))
    best: dict[str, dict] = {}
    for r in rows:
        best.setdefault(r["entry"], r)
    return list(best.values())


def _sequence_sweep_entries(d) -> set[str]:
    for ev in reversed(d.evidence.get("public_structures", [])):
        per_chain = (ev.get("value") or {}).get("per_chain")
        if per_chain:
            return {h["entry"] for hits in per_chain.values() for h in hits}
    return set()


def read_foldseek_cmd(cfg, d, tsv: str, actor: str, limit: int = 25) -> str:
    from residudes.dossier import now
    hits = read_foldseek(tsv)
    for h in hits[:limit]:
        h.update(entry_meta(h["entry"]))
    close = [h for h in hits if (h.get("tm") or 0) >= CLOSE_TM]
    already = _sequence_sweep_entries(d)
    new = [h for h in close if h["entry"] not in already]

    d.evidence.setdefault("public_structures", []).append({"time": now(), "by": actor, "value": {
        "source": "foldseek", "table": str(tsv), "close_tm": CLOSE_TM,
        "hits": [{k: v for k, v in h.items() if k != "query"} for h in hits[:limit]],
        "close": [h["entry"] for h in close],
        "new_to_the_sequence_sweep": [h["entry"] for h in new]}})

    lines = [f"foldseek: {len(hits)} entries, {len(close)} at TM >= {CLOSE_TM:.2f} ({tsv})",
             f"    {'entry':8s} {'TM':>5s} {'ident':>6s} {'cover':>6s} {'res Å':>6s} {'released':10s} title"]
    for h in hits[:limit]:
        res = f"{h['resolution']:.2f}" if isinstance(h.get("resolution"), (int, float)) else "-"
        tm = f"{h['tm']:.2f}" if h.get("tm") is not None else "-"
        fid = f"{h['fident']:.0%}" if h.get("fident") is not None else "-"
        cov = f"{h['coverage']:.0%}" if h.get("coverage") is not None else "-"
        mark = "*" if (h.get("tm") or 0) >= CLOSE_TM else " "
        lines.append(f"  {mark} {h['entry']:8s} {tm:>5s} {fid:>6s} {cov:>6s} {res:>6s} "
                     f"{h.get('released') or '-':10s} {(h.get('title') or '')[:58]}")
    if new:
        lines.append(f"\nfound by structure, not by sequence: {', '.join(h['entry'] for h in new[:10])}"
                     + (f" (+{len(new) - 10} more)" if len(new) > 10 else ""))
        lines.append("  this is the case the sequence sweep cannot make: look at these before settling the templates.")
    elif close and already:
        lines.append("\nevery close structural hit was already in the sequence sweep")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- HHpred: profile against profile

HHR_HIT = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*?)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)\s+"
                     r"(\d+)-(\d+)\s+(\d+)-(\d+)\s*\((\d+)\)\s*$")
HH_PROB = 70.0              # hhsearch's own "probably homologous" line


def hh_db(cfg, explicit: str | None = None) -> tuple[Path, str]:
    """(database prefix, where it came from) for hhsearch, which also takes a prefix rather than a file."""
    if explicit:
        cand, how = Path(explicit), "--db"
    else:
        dbs = (cfg.extra.get("databases") if cfg else None) or {}
        key = next((k for k in ("hhsearch_db", "hh_pdb70", "pdb70") if dbs.get(k)), None)
        if not key:
            raise FileNotFoundError(
                "no HH-suite database configured: set [databases] hhsearch_db in residudes.toml (the prefix of a "
                "pdb70/pdb100-style HHM database). Without one, use --foldseek, which needs a model but no profile db.")
        cand, how = Path(dbs[key]), f"[databases] {key}"
    if not any(Path(str(cand) + s).exists() for s in ("_hhm.ffdata", "_a3m.ffdata", "_cs219.ffdata")):
        raise FileNotFoundError(f"no HH-suite database at {cand} (expected {cand.name}_hhm.ffdata beside it)")
    return cand, how


def hh_query(cfg, d, chain: str | None = None) -> tuple[Path, str]:
    """(query file, what it is): the chain's own MSA when the target has one, else its bare sequence.
    A profile search on a single sequence is weak — say which it was rather than hide it."""
    from residudes.dossier import Dossier
    from residudes.msa import target_msas
    letters = [c[0] for c in chains(d)]
    want = (chain or letters[0]).upper()
    if want not in letters:
        raise ValueError(f"{d.target}: no chain {want} (chains: {', '.join(letters)})")
    for p in target_msas(cfg, d):
        if re.search(rf"(^|[^A-Za-z]){want}([^A-Za-z]|$)", p.stem) or p.stem.upper() == want:
            return p, f"MSA {p.name}"
    out = Dossier.path(cfg.targets_dir, d.target).parent / "structures"
    out.mkdir(parents=True, exist_ok=True)
    seq = next(s for ln, _n, s in chains(d) if ln == want)
    q = out / f"query_{want}.fasta"
    q.write_text(f">{d.target}_{want}\n{seq}\n")
    return q, "the bare sequence (no MSA yet: a profile search on one sequence is weak)"


def hhsearch_cmd(cfg, d, actor: str, chain: str | None = None, db: str | None = None, evalue: float = 1e-3) -> str:
    """Submit a profile-profile search of one chain against an HH-suite database, through sbatch."""
    from residudes import ops
    from residudes.dossier import Dossier, now
    from residudes.msa import tool
    prefix, how = hh_db(cfg, db)
    exe = tool(cfg, "hhsearch")
    if not exe:
        raise FileNotFoundError("hhsearch not found: put hh-suite on PATH or set [tools] hhsearch in residudes.toml")
    query, what = hh_query(cfg, d, chain)
    letter = (chain or chains(d)[0][0]).upper()
    out = Dossier.path(cfg.targets_dir, d.target).parent / "structures"
    out.mkdir(parents=True, exist_ok=True)
    hhr = out / f"hhsearch_{letter}.hhr"
    cmd = f"{exe} -i {query} -d {prefix} -o {hhr} -cpu 8 -e {evalue} -p 20 -Z 250 -B 250"
    job = ops._sbatch(cmd, f"residudes-hhsearch-{d.target}", out / f"hhsearch_{letter}.%j.log", gpu=False, hours=4)
    d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm",
                   "what": f"hhsearch chain {letter}", "state": "submitted", "out": str(hhr)})
    d.note(actor, f"hhsearch chain {letter} from {what} against {prefix.name} ({how}): job {job}")
    return (f"hhsearch chain {letter} from {what}, against {prefix} ({how})\njob {job} -> {hhr}\n"
            f"when it is done: `residudes structures {d.target} --read-hhsearch {hhr}`")


def read_hhsearch(hhr: str | Path) -> list[dict]:
    """The hit table of an .hhr, one row per entry (its best hit), best probability first."""
    rows: list[dict] = []
    for line in Path(hhr).read_text(errors="ignore").splitlines():
        m = HHR_HIT.match(line)
        if not m:
            continue
        (_no, hit, _desc, prob, ev, _pv, score, _ss, cols,
         qs, qe, _ts, _te, tlen) = m.groups()
        entry, chain = _entry_of(hit)
        rows.append({"entry": entry, "chain": chain, "prob": float(prob), "evalue": float(ev),
                     "score": float(score), "cols": int(cols), "query_range": [int(qs), int(qe)],
                     "template_length": int(tlen)})
    rows.sort(key=lambda r: -r["prob"])
    best: dict[str, dict] = {}
    for r in rows:
        best.setdefault(r["entry"], r)
    return list(best.values())


def read_hhsearch_cmd(cfg, d, hhr: str, actor: str, limit: int = 25) -> str:
    from residudes.dossier import now
    hits = read_hhsearch(hhr)
    for h in hits[:limit]:
        h.update(entry_meta(h["entry"]))
    likely = [h for h in hits if h["prob"] >= HH_PROB]
    already = _sequence_sweep_entries(d)
    new = [h for h in likely if h["entry"] not in already]

    d.evidence.setdefault("public_structures", []).append({"time": now(), "by": actor, "value": {
        "source": "hhsearch", "table": str(hhr), "min_probability": HH_PROB,
        "hits": hits[:limit], "likely": [h["entry"] for h in likely],
        "new_to_the_sequence_sweep": [h["entry"] for h in new]}})

    lines = [f"hhsearch: {len(hits)} entries, {len(likely)} at prob >= {HH_PROB:.0f}% ({hhr})",
             f"    {'entry':8s} {'prob':>5s} {'E':>9s} {'cols':>5s} {'res Å':>6s} {'released':10s} title"]
    for h in hits[:limit]:
        res = f"{h['resolution']:.2f}" if isinstance(h.get("resolution"), (int, float)) else "-"
        mark = "*" if h["prob"] >= HH_PROB else " "
        lines.append(f"  {mark} {h['entry']:8s} {h['prob']:5.1f} {h['evalue']:9.1e} {h['cols']:5d} {res:>6s} "
                     f"{h.get('released') or '-':10s} {(h.get('title') or '')[:58]}")
    if new:
        lines.append(f"\nfound by profile, not by sequence: {', '.join(h['entry'] for h in new[:10])}"
                     + (f" (+{len(new) - 10} more)" if len(new) > 10 else ""))
        lines.append("  a remote homologue is a template candidate, not a match: check the alignment before using it.")
    return "\n".join(lines)
