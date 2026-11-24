"""`residudes report T`: the target analysis as a document, and which priors each decision rests on.

The agent's analysis lives in the dossier as evidence entries (construct, identity, public structures, MSA depth,
stoichiometry evidence, QA, gates ...) and decisions that cite them. This module turns that into one readable
analysis.md:
  - the target and every prior found, at the stage it was found;
  - every decision with the priors it rests on and the alternatives it rejected;
  - a prior → decision map, so it is visible that the analysis actually reached the decisions;
  - what was asked of a person, and the answers.
`links()` is also what docs/figures/make_figures.py draws.

A decision cites a prior when one of its evidence lines names an evidence key (whole word: `msa`, `gates_A3.csv`
-> gates_A3, but not af3_msa -> msa), uses a known alias (`n-sweep` -> stoichiometry_evidence), points to an earlier decision (`D3`), or quotes
a person (`OPERATOR`, an answer).
"""
from __future__ import annotations

import json
import re

from residudes.dossier import tkey, Dossier

ALIASES = {"n-sweep": "stoichiometry_evidence", "nsweep": "stoichiometry_evidence"}
PERSON_RX = re.compile(r"\bOPERATOR\b|\banswer(ed)?\b|\bperson\b", re.I)
HIDDEN = ("retrospective", "native")                 # native-derived: only with with_retro


def stage_at(d: Dossier, time: str) -> str:
    """The dossier's stage at a given time, from the stage moves in the log."""
    stage = "NEW"
    for x in d.log:
        if tkey(x["time"]) >= tkey(time):           # a move in the same second comes after the work it closes
            break
        m = re.match(r"stage (\w+) -> (\w+)", x["note"])
        if m:
            stage = m.group(2)
    return stage


def links(d: Dossier) -> dict[str, dict]:
    """Per decision id: the evidence keys, earlier decisions and person notes it cites."""
    keys = [k for k in d.evidence if k not in HIDDEN]
    ids = [x["id"] for x in d.decisions]
    out = {}
    for dec in d.decisions:
        text = " ".join(dec["evidence"])
        cited = [k for k in keys if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text)]      # msa, not af3_msa
        cited += [v for a, v in ALIASES.items() if a in text.lower() and v in keys and v not in cited]
        deps = [i for i in ids if i != dec["id"] and re.search(rf"\b{i}\b", text) and ids.index(i) < ids.index(dec["id"])]
        both = text + " " + dec.get("chose", "")
        cps = [c["id"] for c in d.checkpoints if re.search(rf"\b{c['id']}\b", both)]          # a checkpoint conversation
        out[dec["id"]] = {"evidence": cited, "decisions": deps, "checkpoints": cps,
                          "person": bool(PERSON_RX.search(text) or cps), "stage": stage_at(d, dec["time"])}
    return out


def _value(v, width: int = 400) -> str:
    """One line, for a table cell or a slide."""
    if isinstance(v, dict):
        parts = [f"**{k}**: {_value(x, 160)}" for k, x in v.items()]
        return "; ".join(parts)[:width * 2]
    if isinstance(v, list):
        return ", ".join(_value(x, 120) for x in v)[:width]
    s = str(v).replace("\n", " ")
    return s if len(s) <= width else s[:width - 1] + "…"


def _scalars(v) -> bool:
    return isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)


def _records(v) -> bool:
    """A list of like-shaped records, which reads as a table rather than as prose."""
    if not isinstance(v, list) or len(v) < 2 or not all(isinstance(x, dict) for x in v):
        return False
    cols = {k for x in v for k in x}
    return len(cols) <= 8 and all(len(set(x) & cols) >= len(cols) - 2 for x in v)


def _table(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    out = ["| " + " | ".join(c.replace("_", " ") for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_value(r.get(c, ""), 300).replace("|", "/") or "–" for c in cols) + " |")
    return out


def _block(v, depth: int = 0) -> list[str]:
    """Evidence as nested bullets, with a table where the value is a list of like-shaped records. Nothing is
    truncated at this level: the analysis is the document a person reads, and a cut-off claim is a lost claim."""
    pad = "  " * depth
    if isinstance(v, dict):
        out: list[str] = []
        for k, x in v.items():
            name = f"**{k.replace('_', ' ')}**"
            if _records(x):
                out += [f"{pad}- {name}:", ""] + _table(x) + [""]
            elif isinstance(x, (dict, list)) and x and not _scalars(x):
                out.append(f"{pad}- {name}:")
                out += _block(x, depth + 1)
            else:
                out.append(f"{pad}- {name}: {_value(x, 1200) if x or x == 0 else 'none'}")
        return out
    if _records(v):
        return [""] + _table(v) + [""]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, dict):                 # one record per group: its first field leads, the rest hang off it
                sub = _block(x, depth + 1)
                out += sub[:1] + ["  " + s for s in sub[1:]]
            elif isinstance(x, list):
                out += _block(x, depth + 1)
            else:
                out.append(f"{pad}- {_value(x, 1200)}")
        return out
    return [f"{pad}{_value(v, 4000)}"]


SOURCE_KEY = re.compile(r"source|reference|^refs?$|citation|pmid|pmc|doi|paper|read_from|from_file|^method$|found_by", re.I)
_CITE_ID = re.compile(r"PMC\d{4,}|PMID[:\s]*\d{4,}|10\.\d{4,}/[^\s,;)\]]+"
                      r"|[\w.-]+\.(?:pdf|xml|txt|cif|pdb)\b", re.I)          # a paper, or the local file it was read from
NO_SOURCE_SKIP = ("report", "pool", "gates", "slate_proposal", "results_review", "msa_depth", "jobs")


def _leaves(v) -> list[str]:
    """Every scalar under a value that a source-ish key points at: one source may be recorded as a string, as a
    list of them ("UniProt feature table", "alignment done here"), or as a dict of fields."""
    if isinstance(v, (str, int, float)):
        return [str(v).strip()] if str(v).strip() else []
    if isinstance(v, list):
        return [s for x in v for s in _leaves(x)]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _leaves(x)]
    return []


def _walk_sources(v, out: list[str]) -> None:
    if isinstance(v, dict):
        for k, x in v.items():
            if SOURCE_KEY.search(k):
                out += _leaves(x)
            else:
                _walk_sources(x, out)
    elif isinstance(v, list):
        for x in v:
            _walk_sources(x, out)
    elif isinstance(v, str) and _CITE_ID.search(v):
        out.append(v.strip())


def sources(d: Dossier, with_retro: bool = False) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """The bibliography: every source the evidence records, in reading order, each with the evidence it supports;
    and the evidence keys that record no source at all. Two mentions of one paper merge on their identifier
    (a PMC id, a PMID, a DOI), so the same citation written two ways is one entry."""
    groups: list[dict] = []
    by_id: dict[str, dict] = {}
    unsourced = []
    for k, entries in d.evidence.items():
        if not entries or (k in HIDDEN and not with_retro) or k in NO_SOURCE_SKIP:
            continue
        hits: list[str] = []
        _walk_sources(entries[-1].get("value"), hits)
        if not hits:
            unsourced.append(k)
        for s in hits:
            found = {re.sub(r"[:\s]+", "", m.group(0)).lower(): m.group(0) for m in _CITE_ID.finditer(s)}
            g = next((by_id[i] for i in found if i in by_id), None)     # one shared identifier is the same source
            if g is None:
                g = next((x for x in groups if not found and not x["ids"] and x["text"] == s), None)
            if g is None:
                g = {"text": s, "keys": [], "ids": {}}
                groups.append(g)
            for i, orig in found.items():
                g["ids"].setdefault(i, orig)
                by_id.setdefault(i, g)
            if len(s) > len(g["text"]):                 # keep the fullest wording of the same citation
                g["text"] = s
            if k not in g["keys"]:
                g["keys"].append(k)
    out = []
    for g in groups:                                    # an identifier only the other wording carried is kept too
        extra = [o for o in g["ids"].values() if o.lower() not in g["text"].lower()]
        out.append((g["text"] + (f" ({'; '.join(extra)})" if extra else ""), g["keys"]))
    return out, unsourced


def _note(ddir) -> list[str]:
    """The agent's own written target note (`notes/*.md` in the dossier), embedded with its headings demoted and
    its links made relative to the dossier."""
    from pathlib import Path
    nd = Path(ddir) / "notes"
    if not nd.is_dir():
        return []
    out: list[str] = []
    for f in sorted(nd.glob("*.md")):
        body = relativise(f.read_text()[:60000], Path(ddir)).replace("](../", "](").splitlines()
        while body and (not body[0].strip() or body[0].startswith("# ")):      # its title repeats the heading above
            body.pop(0)
        out += ["", f"<!-- {f.name} -->", ""] + ["#" + x if x.startswith("#") else x for x in body]
    return out


def relativise(text: str, ddir) -> str:
    """Links written as absolute paths inside the dossier, made relative to it.

    A note is written in the dossier and then read somewhere else - embedded in analysis.md, copied into an
    exported package, rendered on a git host - and an absolute path survives none of those. The file it points at
    travels with it, so the link should too."""
    from pathlib import Path
    d = str(Path(ddir).resolve())
    return text.replace(d + "/", "").replace(d, ".")


# a PDB id has a digit first and at least one letter, so a year (2026) or a count is not read as an entry
_PDB = re.compile(r"\b([1-9](?=[A-Za-z0-9]{3}(?![A-Za-z0-9]))(?=[0-9]*[A-Za-z])[A-Za-z0-9]{3})(?:[_:]([A-Za-z0-9]{1,2}))?\b")
_IDENT = re.compile(r"\d{1,3}(?:\s*[-\u2013]\s*\d{1,3})?\s*%\s*id[a-z]*|fident[ =]*[01]\.\d+"
                    r"|exact[- ]sequence match|exact match", re.I)   # a search threshold is not a hit's identity
_COVER = re.compile(r"qcov[ ~=]*[0-9]+\s*/\s*[0-9]+|qcov[ ~=]*[01]?\.\d+|coverage[ ~=]*[01]?\.\d+"
                    r"|aligned query [0-9]+\s*[-\u2013]\s*[0-9]+|~?[0-9]+-col coverage of [0-9]+", re.I)
_ORGANISM = ("organism", "species", "taxon", "lineage", "header", "kingdom", "strain", "source_org")
_FAMILY = ("family", "fold", "superfamily", "domain", "pfam", "interpro", "architecture", "class", "ec_number")
_KNOWN = ("uniprot", "pdb", "exact", "basis", "metadata", "alias", "identity_to")
_CONSTRUCT = ("tag", "native", "term", "note", "mutation", "truncat", "signal", "length", "numbering",
              "subunit", "residues", "chain")
_MSA_NOTES = ("effective", "assessment", "note", "default_msa", "regime", "kept", "famil", "crawl",
              "built", "requested", "hmmscan", "depth_after", "verdict", "thin_region")
_COMP = (("charged D/E/K/R", "DEKR"), ("hydrophobic A/V/I/L/M/F/W/Y", "AVILMFWY"), ("G/P", "GP"), ("S/T", "ST"))


def _latest(ev: dict, key: str):
    v = ev.get(key) or []
    return v[-1].get("value") if v else None


def _fields(val, names: tuple[str, ...], prefix: str = "") -> list[tuple[str, str]]:
    """(field, text) for each field whose name contains one of `names`, descending into per-chain nesting: the
    skills record a complex as {chain: {organism, family, ...}}, so the field is `identity.B.family`."""
    if not isinstance(val, dict):
        return []
    out: list[tuple[str, str]] = []
    for k, v in val.items():
        if isinstance(v, dict):
            out += _fields(v, names, f"{prefix}{k}.")
        elif (not names or any(n in k.lower() for n in names)) and v not in (None, "", [], {}):
            out.append((f"{prefix}{k}", _value(v, 700)))
    return out


def _chains(d: Dossier) -> list[tuple[str, str]]:
    """(name, sequence) per chain, from the target FASTA the dossier keeps."""
    txt = (d.sequence.get("text") or "").strip()
    if not txt:
        return []
    if not txt.startswith(">"):
        return [("sequence", "".join(txt.split()))]
    out: list[tuple[str, str]] = []
    for blk in txt.split(">")[1:]:
        head, *rest = blk.splitlines()
        s = "".join(x.strip() for x in rest)
        if s:
            out.append((head.strip().rstrip("|").strip() or f"chain {len(out) + 1}", s))
    return out


def _profile(d: Dossier, ev: dict, ddir=None) -> list[str]:
    """What the target is. `identity` and `construct` exist to describe it, so every field they carry belongs here;
    the groups only decide the order, so a field nobody thought to name is still shown."""
    rows: list[tuple[str, str]] = []
    chains = _chains(d)
    if chains:
        rows.append(("chains", ", ".join(f"{n} — {len(s)} aa" for n, s in chains)))
    rows.append(("stoichiometry given by CASP", str(d.stoichiometry.get("given") or "not given")))
    for key in ("identity", "construct"):
        found = _fields(_latest(ev, key), ())
        rank = {}
        for i, group in enumerate((_ORGANISM, _FAMILY, _KNOWN, _CONSTRUCT)):
            for f, _ in found:
                name = f.rsplit(".", 1)[-1].lower()
                if f not in rank and any(n in name for n in group):
                    rank[f] = i
        rows += [(f"`{key}.{f}`", v) for f, v in sorted(found, key=lambda x: rank.get(x[0], len(rank) + 4))
                 if f.rsplit(".", 1)[-1].lower() not in ("source", "sources")]
    seen: set[str] = set()
    out = ["## What the target is", "", "| | recorded |", "|---|---|"]
    for label, text in rows:
        if label in seen:
            continue
        seen.add(label)
        out.append(f"| {label} | {text} |")
    if len(out) <= 4:
        return []
    from pathlib import Path
    fig = Path(ddir) / "report" / "figs" / "chain_map.svg" if ddir else None
    if fig is not None and fig.exists():
        out += ["", "![Each chain to scale, with the domains that were recorded and the residues a gate would use]"
                    "(report/figs/chain_map.svg)"]
    return out + [""]


def _composition(d: Dossier) -> list[str]:
    """Composition per chain, counted from the FASTA."""
    from itertools import groupby
    chains = _chains(d)
    if not chains:
        return []
    out = ["## The sequence itself", "",
           "Counted from the dossier's FASTA, so this is arithmetic and not a prediction. A repetitive window or a "
           "long run of one residue is where an MSA and a template are most often thin.", "",
           "| chain | residues | " + " | ".join(n for n, _ in _COMP) + " | Cys | His | longest run | "
           "most repetitive 30-mer |",
           "|---|---|" + "---|" * (len(_COMP) + 4)]
    for name, s in chains:
        n = len(s)
        comp = " | ".join(f"{round(100 * sum(s.count(a) for a in grp) / n)}%" for _, grp in _COMP)
        runs = [(len(list(g)), k) for k, g in groupby(s)]
        run, run_aa = max(runs) if runs else (0, "?")
        rep, at = 0, 0
        for i in range(max(0, n - 29)):
            w = s[i:i + 30]
            c = max(w.count(a) for a in set(w))
            if c > rep:
                rep, at = c, i + 1
        out.append(f"| {name} | {n} | {comp} | {s.count('C')} | {s.count('H')} | {run}× {run_aa} | "
                   + (f"{rep}/30 identical, from residue {at}" if rep else "–") + " |")
    return out + [""]


_MSA_FIELDS = {"n": "sequences", "n_seq": "sequences", "sequences": "sequences", "nseq": "sequences",
               "n_cov50": "≥50% coverage", "cov50": "≥50% coverage", "length": "columns", "neff": "Neff",
               "coverage_median": "median coverage", "coverage_min": "min coverage",
               "median_identity": "median identity"}
_MSA_ORDER = ("sequences", "≥50% coverage", "columns", "Neff", "median coverage", "min coverage", "median identity")


def _msa_rows(val, prefix: str = "") -> list[tuple[str, dict]]:
    """(label, counts) for every node that carries MSA counts, at whatever depth it was recorded: per file, or per
    chain and then per database."""
    if not isinstance(val, dict):
        return []
    mine = {_MSA_FIELDS[k.lower()]: v for k, v in val.items()
            if k.lower() in _MSA_FIELDS and isinstance(v, (int, float))}
    if mine:
        return [(prefix.rstrip(" /"), mine)]
    nums = {k: v for k, v in val.items() if isinstance(v, (int, float))}
    if nums and len(nums) == len([v for v in val.values() if not isinstance(v, dict)]):
        return [(f"{prefix}{k}", {"sequences": v}) for k, v in nums.items()]      # database -> count
    out: list[tuple[str, dict]] = []
    for k, v in val.items():
        step = "" if k.lower() in ("depth", "msas", "databases", "per_db", "files") else f"{k} / "
        out += _msa_rows(v, f"{prefix}{step}")
    return out


def _msas(ev: dict, ddir=None) -> list[str]:
    """MSA depth as it was measured, the agent's verdict on it, and the coverage plot if one was drawn."""
    out: list[str] = []
    rows: list[tuple[str, dict]] = []
    for key in ("msa_depth", "msa"):
        rows += _msa_rows(_latest(ev, key))
    if rows:
        cols = [c for c in _MSA_ORDER if any(c in r for _, r in rows)]
        out += ["| MSA | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for label, r in rows:
            out.append(f"| `{label}` | " + " | ".join(f"{r[c]:,}" if isinstance(r.get(c), int) else str(r.get(c, "–"))
                                                      for c in cols) + " |")
        out.append("")
    for key in ("msa", "msa_depth", "msa_crawl"):
        out += [f"- **`{key}.{f}`:** {t}" for f, t in _fields(_latest(ev, key), _MSA_NOTES)]
    from pathlib import Path
    fig = Path(ddir) / "report" / "figs" / "msa_coverage.svg" if ddir else None
    if fig is not None and fig.exists():
        out += ["", "![Coverage per residue, one line per MSA](report/figs/msa_coverage.svg)"]
    return ["## How much the MSAs cover", ""] + out + [""] if out else []


def _templates(ev: dict) -> list[str]:
    """One row per structure a search returned, with the identity as the search reported it."""
    rows: list[str] = []
    for key in ("templates", "public_structures"):
        val = _latest(ev, key)
        if val in (None, "", {}, []):
            continue
        items: list[tuple[str, str]] = []
        if isinstance(val, dict):
            for f, v in val.items():
                items += [(f, x if isinstance(x, str) else _value(x, 700)) for x in (v if isinstance(v, list) else [v])]
        else:
            items.append(("", _value(val, 700)))
        for f, text in items:
            if text.strip().lower() in ("true", "false", "none", "–", "-"):
                continue
            m = _PDB.search(text)
            entry = (m.group(1).lower() + (f"_{m.group(2)}" if m.group(2) else "")) if m and not m.group(1).isdigit() else ""
            ident, cover = _IDENT.search(text), _COVER.search(text)
            rows.append(f"| {('`' + entry + '`') if entry else '–'} | {ident.group(0) if ident else '–'} | "
                        f"{cover.group(0) if cover else '–'} | `{key}.{f}` | {text} |")
    if not rows:
        return []
    return ["## Templates and public structures", "",
            "Identity and coverage are quoted as the search reported them, not recomputed; a dash means that "
            "search did not give one.", "",
            "| entry | identity | coverage | search | what it reported |", "|---|---|---|---|---|"] + rows + [""]


def _fig(ddir, name: str, caption: str) -> list[str]:
    """A figure `residudes report` drew, linked relative to the dossier, or nothing when it was not drawn."""
    from pathlib import Path
    if ddir and (Path(ddir) / "report" / "figs" / f"{name}.svg").exists():
        return ["", f"![{caption}](report/figs/{name}.svg)", ""]
    return []


def _literature(d: Dossier, ddir=None) -> list[str]:
    """What was read, claim by claim, with the source and the priors that share an identifier with it.

    The claims are in the raw record below as well. They are here because a claim without its source, and without
    what in this dossier rests on it, is the part of an analysis that is hardest to check later."""
    from residudes import figures
    val = figures._latest(d, "literature", "background", "reading")
    claims = figures.claims(val)
    if not claims:
        return []
    skip = figures.target_ids(d.target)
    out = ["## What the literature says, claim by claim", "",
           "Each claim as it was recorded, with its source. The last column is drawn, not asserted: a prior or a "
           "decision appears there when it names the same paper, structure, accession, residue or motif as the "
           "claim.", "",
           "| # | claim | source | shares an identifier with |", "|---|---|---|---|"]
    for i, c in enumerate(claims, 1):
        on = figures.rests_on(d, figures.tokens(c["text"] + " " + c["source"], skip))
        label = f"**{c['label']}** " if c["label"] else ""
        out.append(f"| {i} | {label}{_value(c['text'], 700).replace('|', '/')} | "
                   f"{_value(c['source'], 300).replace('|', '/') or '**none recorded**'} | "
                   + (", ".join(on) or "–") + " |")
    for label, rx in (("Not settled by anything read", figures._UNKNOWN_F), ("Searched", figures._SEARCHED_F)):
        for t in figures._fields_matching(val, rx):
            out += ["", f"**{label}:** {_value(t, 1200)}"]
    return out + _fig(ddir, "literature", "Every claim, its source, and the priors and decisions that share it") + [""]


def _sites(d: Dossier, ddir=None) -> list[str]:
    """Each recorded site, as `residudes gate --init` reads it, with the residue identities from the FASTA."""
    from residudes import figures
    cs = _chains(d)
    if not cs:
        return []
    letters = {chr(ord("A") + i): len(s) for i, (_, s) in enumerate(cs)}
    seqs = {chr(ord("A") + i): s for i, (_, s) in enumerate(cs)}
    skipped: list[str] = []
    recs = figures.site_records(d, letters, skipped)
    mods, mod_skipped = figures.modifications(d)
    if not recs and not mods:
        return []
    out = ["## The sites the analysis found", "",
           "The residues are the ones a gate would use: read by the same function as `residudes gate --init`, which "
           "prefers the numbering recorded as a field over the prose beside it. The residue letters come from the "
           "target FASTA, so a site on the wrong numbering shows up as the wrong residue.", ""]
    if recs:
        out += ["| evidence | chain | residues | read as | what it is | source |", "|---|---|---|---|---|---|"]
        for r in recs:
            res = ", ".join(f"{seqs[r['chain']][x - 1]}{x}" for x in r["residues"])
            out.append(f"| `{r['key']}` | {r['chain']} | {res} | {r['how']} | "
                       f"{_value(r['what'], 400).replace('|', '/') or '–'} | "
                       f"{_value(r['source'], 200).replace('|', '/') or '**none recorded**'} |")
    if mods:
        out += ["", "Chemistry the FASTA does not carry, placed on a residue: "
                + "; ".join(f"**{ch}{r}** {lab}" for ch, r, lab in mods) + "."]
    for t in sorted(set(skipped + mod_skipped)):
        out += ["", f"Not placed: {t}"]
    return out + _fig(ddir, "sites", "Each recorded site in its own sequence, with what the record says it does") + [""]


def _motifs(d: Dossier, ddir=None) -> list[str]:
    """The motifs the analysis named, scanned against the sequence: where each fits, and which one was recorded."""
    from residudes import figures
    cs = _chains(d)
    ms = figures.motifs(d)
    if not cs or not ms:
        return []
    seqs = {chr(ord("A") + i): s for i, (_, s) in enumerate(cs)}
    rows: list[str] = []
    for m in ms:
        for ch in ([m["chain"]] if m["chain"] in seqs else sorted(seqs)):
            hits = figures.hits(m["pattern"], seqs[ch])
            if not hits:
                continue
            anchor = next((j for j, (t, _) in enumerate(m["pattern"]) if t[:1] == "p"), 0)
            at = ", ".join(f"{seqs[ch][h + anchor - 1]}{h + anchor}" for h in hits[:12])
            rec = ", ".join(f"{seqs[ch][x - 1]}{x}" for x in m["recorded"] if x <= len(seqs[ch]))
            matched = [h for h in hits if h + anchor in m["recorded"]]
            rows.append(f"| `{m['token']}` | {ch} | {len(hits)} | {at} | {rec or '–'} | "
                        f"{'yes' if matched else 'no'} | `{m['key']}` |")
    if not rows:
        return []
    out = ["## The motifs, and every place they can sit", "",
           "Each pattern the analysis wrote down, matched against the chain it was written about. This is "
           "arithmetic on the FASTA, not a claim: it says where the pattern fits, and whether the residue the "
           "analysis recorded as the site is one of those places.", "",
           "| pattern | chain | places it fits | at | recorded site | the record's residue is one of them | from |",
           "|---|---|---|---|---|---|---|"] + rows
    cands = figures.candidates(d)
    if cands:
        out += ["", "Candidates the analysis weighed: "
                + "; ".join(f"**{k}** {_value(v, 200)}" for k, v in cands.items()) + "."]
    return out + _fig(ddir, "motif", "Every place the recorded motifs can sit in the sequence") + [""]


def _clusters(d: Dossier, ddir=None) -> list[str]:
    """The pool as residudes-qa clustered it: cluster sizes, and the best model of each by both scores."""
    from residudes import figures
    pts = figures.cluster_points(ddir) if ddir else []
    if not pts:
        return []
    scores = figures.pool_scores(ddir)
    slate = figures.slate_models(ddir)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for p in pts:
        groups.setdefault((p["sto"], p.get("run", p["variant"]), p["cluster"]), []).append(p)
    out = ["## The pool, clustered", "",
           "residudes-qa clusters the pool by TM-score. One run covers one input variant or several together, and "
           "when several are clustered together the variants column says which models landed in each cluster — a "
           "cluster that holds one variant and nothing else is that variant changing the structure. The slate "
           "takes at most one model per cluster in the first three slots, so these are the alternatives a slate "
           "hedges across.", "",
           "| stoichiometry | cluster | models | variants in it | best confidence | its model | "
           "best predicted DockQ |", "|---|---|---|---|---|---|---|"]
    for (sto, _run, cl), ps in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -len(kv[1]))):
        vals = [(scores.get(p["key"]) or {}).get("rs") for p in ps]
        xgbs = [(scores.get(p["key"]) or {}).get("xgb") for p in ps]
        best = max(((v, p) for v, p in zip(vals, ps) if v is not None), key=lambda t: t[0], default=(None, None))
        name = best[1]["key"].split("__", 1)[-1] if best[1] else "–"
        mark = f" ({slate[best[1]['key']]})" if best[1] and best[1]["key"] in slate else ""
        xs = [x for x in xgbs if x is not None]
        mix: dict[str, int] = {}
        for p in ps:
            mix[p["variant"]] = mix.get(p["variant"], 0) + 1
        out.append(f"| {sto} | {cl} | {len(ps):,} | "
                   + ", ".join(f"{v} {n:,}" for v, n in sorted(mix.items(), key=lambda kv: -kv[1]))
                   + f" | {f'{best[0]:.3f}' if best[0] is not None else '–'} | `{name}`{mark} | "
                   f"{f'{max(xs):.3f}' if xs else '–'} |")
    notes = figures.cluster_notes(pts, scores)
    if notes:
        out += ["", "What the clustering says, from these numbers alone:", ""] + [f"- {t}" for t in notes]
    return out + _fig(ddir, "clusters",
                      "The same models four ways: clusters, predicted DockQ, confidence, and which predictor "
                      "sits where") + [""]


def _slates(ddir) -> list[str]:
    """The slate as a table, with residudes-qa's predicted DockQ beside each predictor's own confidence."""
    from pathlib import Path
    out: list[str] = []
    for f in sorted(Path(ddir).glob("slate_v*.json")) if ddir else []:
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        models = s.get("models") or []
        if not models:
            continue
        bits = [str(s.get("sto") or ""), "chosen by review" if s.get("chosen") else "by the rules"]
        cand = s.get("n_candidates")
        if isinstance(cand, dict):
            bits.append(", ".join(f"{k} {v}" for k, v in cand.items()) + " candidates")
        elif cand:
            bits.append(f"{cand} candidates")
        bits.append("interface scores used" if s.get("used_interface") else "no interface scores")
        bits.append("clusters used" if s.get("used_clusters") else "no clusters")
        out += ["", f"### {f.stem.replace('slate_v', 'slate v')}", "", ", ".join(x for x in bits if x), "",
                "| slot | model | predictor | ranking_score | z within predictor | predicted DockQ | cluster |",
                "|---|---|---|---|---|---|---|"]
        for m in models:
            z = m.get("rs_z_runner")
            x = m.get("xgb_pred")
            rs = m.get("ranking_score")
            out.append(f"| {m.get('slot', '–')} | `{m.get('model', '?')}` | {m.get('runner', '–')} | "
                       f"{f'{rs:.3f}' if isinstance(rs, (int, float)) else '–'} | "
                       f"{f'{z:+.2f}' if isinstance(z, (int, float)) else '–'} | "
                       f"{f'{x:.3f}' if isinstance(x, (int, float)) else '–'} | {m.get('cluster') or '–'} |")
    if not out:
        return []
    return ["## Slates", "",
            "`ranking_score` is each predictor's own confidence, and it is not comparable between predictors, so the "
            "z column is within one predictor. The predicted DockQ is residudes-qa's interface score, fitted against "
            "measured DockQ, so it is the column whose scale means the same thing for every predictor "
            "(residudes/docs/metrics.md). A dash means no interface score was computed for that model."] + out + [""]


def key_points(d: Dossier, ddir=None) -> list[tuple[str, str]]:
    """The analysis in one page: (label, sentence) for each thing that decides something, with its numbers.

    Everything here is read out of the dossier and the pool, never written by hand, and a line that has no
    evidence behind it is left out rather than filled with a hedge. The order is the order a person asks: what is
    it, what is the chemistry nobody can see in the FASTA, how much can be aligned, what structures exist, what
    the pool does, what nobody knows, and what is being asked of them."""
    from residudes import figures
    out: list[tuple[str, str]] = []
    cs = _chains(d)
    ident = _latest(d.evidence, "identity")
    bits = []
    for i, (name, seq) in enumerate(cs):
        letter = chr(ord("A") + i)
        fields = [v for f, v in _fields(ident, ("uniprot", "name", "family", "organism", "fold"))
                  if f.startswith(f"{letter}.") or len(cs) == 1]
        short = name.split(",", 1)[1] if "," in name else name
        bits.append(f"**{letter}** ({len(seq)} aa) {short}" + (f" — {'; '.join(x[:90] for x in fields[:3])}"
                                                               if fields else ""))
    if bits:
        out.append(("What it is", " · ".join(bits) + f". Given stoichiometry **"
                    f"{d.stoichiometry.get('given') or 'not given'}**."))
    chem = _latest(d.evidence, "chemistry")
    mods, _ = figures.modifications(d)
    if chem is not None:
        cands = figures.candidates(d)
        line = "; ".join(f"**{ch}{r}** {lab}" for ch, r, lab in mods) or _value(chem, 300)
        if cands:
            line += " — " + "; ".join(f"{k} {v.split(':', 1)[0]}" for k, v in cands.items())
        out.append(("The chemistry the FASTA does not carry", line))
    rows = _msa_rows(_latest(d.evidence, "msa_depth")) + _msa_rows(_latest(d.evidence, "msa"))
    counted = [(lab, r["sequences"]) for lab, r in rows if isinstance(r.get("sequences"), int)]
    if counted:
        hi = max(counted, key=lambda t: t[1])
        lo = min(counted, key=lambda t: t[1])
        out.append(("MSA depth", f"deepest `{hi[0]}` **{hi[1]:,}** sequences, thinnest `{lo[0]}` {lo[1]:,}, "
                                 f"over {len(counted)} measured alignment{'s' if len(counted) != 1 else ''}"))
    pdbs = {m.group(1).lower() for key in ("templates", "public_structures")
            for _, t in _fields(_latest(d.evidence, key), ()) for m in _PDB.finditer(t)}
    if pdbs:
        out.append(("Structures that exist", f"**{len(pdbs)}** PDB entries named in the record: "
                    + ", ".join(f"`{x}`" for x in sorted(pdbs)[:8])
                    + (" …" if len(pdbs) > 8 else "")))
    pts = [p for p in (figures.cluster_points(ddir) if ddir else []) if "x" in p]
    if pts:
        scores = figures.pool_scores(ddir)
        notes = figures.cluster_notes(pts, scores)
        out.append(("The pool", f"**{len({p['key'] for p in pts}):,}** models in "
                    f"**{len({(p.get('run', ''), p['cluster']) for p in pts})}** clusters"
                    + (f". {notes[0]}" if notes else "")))
        for n in notes[1:3]:
            out.append(("", n))
    val = figures._latest(d, "literature", "background", "reading")
    for t in figures._fields_matching(val, figures._UNKNOWN_F)[:1]:
        out.append(("What nobody knows", _value(t, 420)))
    for c in [c for c in d.checkpoints if not c["resolution"]][-1:]:
        out.append((f"Open, and waiting on a person ({c['id']})",
                    "; ".join(f"({i + 1}) {q.split(':', 1)[0][:90]}" for i, q in enumerate(c["questions"]))))
    return out


def closing_points(d: Dossier, ddir=None) -> list[tuple[str, str]]:
    """What the analysis proposes and what it is asking, for the last page: the plan in one line per decision,
    the best model in the pool by each of the two scales, the open questions, and what is missing from the record
    that a person is about to approve anyway."""
    from residudes import figures
    out: list[tuple[str, str]] = []
    if d.decisions:
        by: dict[str, list[str]] = {}
        for dec in d.decisions:
            by.setdefault(dec["type"], []).append(_value(dec["chose"], 150))
        for t, chosen in by.items():
            out.append((t.replace("_", " "), "; ".join(chosen)))
    scores = figures.pool_scores(ddir) if ddir else {}
    if scores:
        rs = [(v["rs"], k) for k, v in scores.items() if v.get("rs") is not None]
        xg = [(v["xgb"], k) for k, v in scores.items() if v.get("xgb") is not None]
        bits = []
        if rs:
            v, k = max(rs)
            bits.append(f"by confidence `{k}` ({v:.3f})")
        if xg:
            v, k = max(xg)
            bits.append(f"by predicted DockQ `{k}` ({v:.3f})")
        if bits:
            same = len({k for _, k in [max(rs)] + ([max(xg)] if xg else [])}) == 1
            out.append(("Best model in the pool so far",
                        " · ".join(bits) + ("" if same else " — not the same model, which is what the slate's "
                                            "two scales are for")))
    for c in [c for c in d.checkpoints if not c["resolution"]]:
        for i, q in enumerate(c["questions"], 1):
            out.append((f"{c['id']} question {i}", _value(q, 400)))
        for x in c["thread"][-2:]:
            who = "the agent" if x["by"] in ("agent", "residudes") else "a person"
            out.append((f"{c['id']} thread, {who}", _value(x["text"], 400)))
    if ddir:
        from residudes.consult import plan_gaps
        gaps = plan_gaps(d, ddir)
        if gaps:
            out.append(("Not recorded, and being approved anyway", "; ".join(gaps)))
    return out


def build(d: Dossier, with_retro: bool = False, ddir=None) -> str:
    lk = links(d)
    ev = {k: v for k, v in d.evidence.items() if with_retro or k not in HIDDEN}
    lines = [f"# {d.target}: target analysis", "",
             f"Stage **{d.stage}**. {len(ev)} kinds of evidence and {len(d.decisions)} decisions, recorded by "
             + (", ".join(sorted({e.get('by', '?') for v in ev.values() for e in v} | {x.get('actor', '?') for x in d.decisions})) or "nobody yet") + ".", ""]
    cs = _chains(d)
    if cs:                                   # count residues, not the characters of the FASTA file
        total = sum(len(s) for _, s in cs)
        per = f" in {len(cs)} chains ({', '.join(str(len(s)) for _, s in cs)})" if len(cs) > 1 else ""
        lines += [f"Sequence: {total} residues{per}. Given stoichiometry: "
                  f"{d.stoichiometry.get('given') or 'not given'}.", ""]

    kp = key_points(d, ddir)
    if kp:                                   # the answer first; the record that produced it follows
        lines += ["## In one page", ""]
        lines += [f"- **{label}.** {text}" if label else f"  - {text}" for label, text in kp] + [""]

    note = _note(ddir) if ddir else []
    if note:
        lines += ["## The target, written up", ""] + note + [""]

    lines += _profile(d, ev, ddir) + _composition(d) + _literature(d, ddir) + _sites(d, ddir) + _motifs(d, ddir) \
        + _msas(ev, ddir) + _templates(ev) + _clusters(d, ddir)

    lines += ["## Every prior, as it was recorded", "",
              "The sections above are read out of these entries. This is the whole record, including the fields they "
              "leave out, each at the stage it was found.", ""]
    for k, v in ev.items():
        if k in ("slate_proposal", "gates"):
            continue
        last = v[-1]
        lines.append(f"### {k}")
        lines.append(f"*{stage_at(d, last.get('time', ''))}, by {last.get('by', '?')}"
                     + (f", {len(v)} entries (latest shown)" if len(v) > 1 else "") + "*")
        lines += [""] + _block(last.get("value")) + [""]

    lines += ["## Decisions and the priors they rest on", "",
              "| id | stage | type | chose | rests on | confidence | rejected |", "|---|---|---|---|---|---|---|"]
    for dec in d.decisions:
        l = lk[dec["id"]]
        rests = ", ".join([f"`{k}`" for k in l["evidence"]] + l["decisions"]
                          + ([f"a person ({', '.join(l['checkpoints'])})" if l["checkpoints"] else "a person's note"] if l["person"] else [])) or "free text only"
        chose = dec["chose"].replace("|", "/")
        alts = "; ".join(a.replace("|", "/")[:90] for a in dec.get("alternatives", [])) or "–"
        lines.append(f"| {dec['id']} | {l['stage']} | {dec['type']} | {chose[:160]} | {rests} | {dec.get('confidence', '')} | {alts[:300]} |")

    cited = [k for k in ev if k not in ("slate_proposal",)]
    lines += ["", "## Prior → decision map", "",
              "Each column is a decision; ● means the decision cites that prior.", "",
              "| prior | " + " | ".join(x["id"] for x in d.decisions) + " |",
              "|---|" + "---|" * len(d.decisions)]
    for k in cited:
        lines.append(f"| `{k}` | " + " | ".join("●" if k in lk[x["id"]]["evidence"] else "" for x in d.decisions) + " |")
    lines.append("| a person's note | " + " | ".join("●" if lk[x["id"]]["person"] else "" for x in d.decisions) + " |")
    unused = [k for k in cited if not any(k in v["evidence"] for v in lk.values())]
    if unused:
        lines += ["", f"Found but not cited by any decision: {', '.join(f'`{k}`' for k in unused)}."]

    if d.checkpoints:
        lines += ["", "## Checkpoints: where the agent and a person worked it out", ""]
        for c in d.checkpoints:
            res = c["resolution"]
            lines.append(f"### {c['id']} ({c['kind']}) at {c['stage']}: "
                         + (f"{res['outcome']} by {res['by']}" if res else "open"))
            lines += [""] + [f"{i + 1}. {q}" for i, q in enumerate(c["questions"])] + [""]
            for x in c["thread"]:
                who = "agent" if x["by"] in ("agent", "residudes") else "person"
                lines.append(f"- {x['time'][11:16]} **{who}**: {x['text'][:600]}")
            if res and res.get("note"):
                lines.append(f"- {res['time'][11:16]} **person** resolved ({res['outcome']}): {res['note'][:400]}")
            lines.append("")
    asks = [x for x in d.log if x["note"].startswith(("NEEDS HUMAN", "FYI"))]
    people = [x for x in d.log if x["actor"] not in ("agent", "residudes")]
    if asks or people:
        lines += ["", "## Between the agent and a person", ""]
        for x in sorted(asks + people, key=lambda x: x["time"]):
            who = "agent" if x["actor"] in ("agent", "residudes") else "person"
            lines.append(f"- {x['time'][11:16]} **{who}**: {x['note'][:400]}")

    from residudes import gate as gatemod
    for g in gatemod.latest_results(d):
        lines += ["", f"## Structural gates `{g.get('name')}` at {g.get('sto')}", "",
                  f"{g.get('n_pass', 0)} of {g.get('n', 0)} models pass every gate.", "",
                  "| gate | checks | models passing |", "|---|---|---|"]
        spec = {x.get("name"): x for x in (g.get("spec") or {}).get("gates", [])}
        for name, n in (g.get("gates") or {}).items():
            what = "every chain of the stoichiometry is present" if name == "complete" else gatemod.describe(spec.get(name, {}))
            lines.append(f"| {name} | {what} | {n} |")
        lines += ["", "| predictor | models | pass | best passing model (ranking_score) |", "|---|---|---|---|"]
        for r, v in sorted((g.get("per_runner") or {}).items()):
            bp = f"{v['best_pass'][0]} ({v['best_pass'][1]:.3f})" if v.get("best_pass") else "–"
            lines.append(f"| {r} | {v.get('n')} | {v.get('pass')} | {bp} |")
        if g.get("census"):
            lines += ["", f"Site census (probe {(g.get('spec') or {}).get('census', {}).get('probe', '?')}): "
                      + "; ".join(f"{r}: " + ", ".join(f"{k} {n}" for k, n in c.items()) for r, c in sorted(g["census"].items())) + "."]
        if g.get("incomplete"):
            lines += ["", "Runs with models missing a chain: " + ", ".join(f"{k} ({n})" for k, n in g["incomplete"].items()) + "."]

    table = _slates(ddir)
    props = d.evidence.get("slate_proposal", [])
    if table:
        lines += [""] + table
    elif props:
        lines += ["", "## Slates", ""]
        for p in props:
            v = p["value"]
            lines.append(f"- v{v.get('version')} ({'chosen by review' if v.get('chosen') else 'rules'}, {p.get('by')}): "
                         + ", ".join(v.get("models", [])))
            if v.get("relaxed"):
                lines.append("  - relaxed for want of candidates: " + "; ".join(f"{m} ({w})" for m, w in v["relaxed"].items()))

    cp = closing_points(d, ddir)
    if cp:
        lines += ["", "## What this proposes, and what it is asking", "",
                  "The decisions above in one line each, the two models the pool's two scales each prefer, and the "
                  "questions a person has to answer before anything runs.", ""]
        lines += [f"- **{label}.** {text}" for label, text in cp]

    srcs, unsourced = sources(d, with_retro)
    if srcs or unsourced:
        lines += ["", "## References", "",
                  "What the analysis read, and which priors rest on it. A prior whose source is not written down "
                  "cannot be checked against the paper later, and `residudes retro` cannot score it.", ""]
        if srcs:
            lines += ["| # | source | supports |", "|---|---|---|"]
            for i, (text, keys) in enumerate(srcs, 1):
                lines.append(f"| {i} | {_value(text, 500).replace('|', '/')} | " + ", ".join(f"`{k}`" for k in keys) + " |")
        if ddir:
            from pathlib import Path
            got: dict[str, list] = {}
            for p in sorted((Path(ddir) / "papers").glob("*")) if (Path(ddir) / "papers").is_dir() else []:
                if p.is_file():
                    g = got.setdefault(p.stem, [[], 0])
                    g[0].append(p.suffix.lstrip(".") or "file")
                    g[1] = max(g[1], p.stat().st_size)
            if got:                                     # a refused download leaves a few hundred bytes of error page
                lines += ["", "Files in `papers/`: " + ", ".join(
                    f"`{s}` ({'/'.join(x)}{'' if n >= 2048 else ', under 2 kB: an error page rather than the paper'})"
                    for s, (x, n) in got.items()) + "."]
        if unsourced:
            lines += ["", f"**No source recorded**: {', '.join(f'`{k}`' for k in unsourced)}. Say where each came "
                      "from (a paper, a structure, an experiment, or a calculation run here)."]
    return "\n".join(lines) + "\n"


def run(d: Dossier, out_path, with_retro: bool = False, ddir=None) -> str:
    notes: list[str] = []
    if ddir:                                        # draw the figures first, so the document can link them
        from residudes import figures
        drawn, problems = figures.draw_all(d, ddir)
        notes = [f"figures: {', '.join(sorted(drawn)) or 'none drawn'}"] + [f"not drawn - {p}" for p in problems]
    text = build(d, with_retro, ddir)
    if out_path:
        from pathlib import Path
        Path(out_path).write_text(text)
        return "\n".join([f"wrote {out_path}"] + notes)
    return text


def links_json(d: Dossier) -> str:
    return json.dumps(links(d), indent=1)
