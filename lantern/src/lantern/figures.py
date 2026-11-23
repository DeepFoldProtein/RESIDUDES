"""Figures drawn from the dossier alone, with no job to run: one per kind of evidence the analysis records.

  chain_map   each chain to scale, with its domains and the residues a gate would use
  literature  every claim that was read, its source, and which of this dossier's priors and decisions share it
  sites       each recorded site in its own sequence, residue by residue, with what the record says it does
  motif       the motifs the analysis named, and every place in the sequence they can sit
  clusters    the pool in the clustering lantern-qa computed (UMAP), one panel per variant

Two rules keep them honest, and they are the same two the chain map started with. Everything drawn comes from a
number that was recorded - a chain and its length from the target FASTA, a site from the `select` or `residues`
field `lantern gate` reads, a cluster and its coordinates from lantern-qa's own output - and anything the parser
could not place is named on the figure rather than left out silently. A figure with nothing to draw returns None,
so the report links what exists and nothing else.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from lantern.dossier import Dossier
from lantern.gate import SITE_KEYS, parse_select, site_from_value, target_ids
from lantern.report import _chains as chains

COLS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8c564b", "#7570b3"]
INK, MUTED, RULE, PANEL, FLAG = "#1d1d1f", "#6e6e73", "#d1d1d6", "#f2f2f7", "#D55E00"
_FONT = "DejaVu Sans, Helvetica, Arial"
_MONO = "DejaVu Sans Mono, Menlo, monospace"

_RANGE = re.compile(r"(\d{1,5})\s*[-–]\s*(\d{1,5})")
_DOMAIN_FIELD = re.compile(r"domain|region|repeat|helix|strand|landmark|lobe|propeller|barrel|groove|tag", re.I)
_NOT_INPUT = re.compile(r"native|maps_to|precursor|uniprot|offset|numbering|identity", re.I)

# ------------------------------------------------------------------------------------------------ text and canvas
_NARROW = set("iljtfr.,:;'!|()[]{}·  ")
_WIDE = set("mwMW@%&")


def tw(text, size: float, bold: bool = False) -> float:
    """Text width in px, estimated per character. DejaVu Sans has no metrics in the stdlib, so this errs wide:
    a box sized from it is never too small for the text inside, which is the failure that shows."""
    w = 0.0
    for ch in str(text):
        w += 0.33 if ch in _NARROW else 0.92 if ch in _WIDE else 0.70 if (ch.isupper() or ch.isdigit()) else 0.60
    return w * size * (1.08 if bold else 1.0)


def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(text, px: float, size: float, lines: int = 99) -> list[str]:
    """`text` broken into lines that fit `px`, at most `lines` of them (the last one ellipsised)."""
    out: list[str] = []
    cur = ""
    for word in str(text).split():
        while tw(word, size) > px and len(word) > 4:                     # one unbreakable token: hard-split it
            cut = max(4, int(len(word) * px / max(1.0, tw(word, size))))
            out.append(word[:cut])
            word = word[cut:]
        nxt = f"{cur} {word}".strip()
        if cur and tw(nxt, size) > px:
            out.append(cur)
            cur = word
        else:
            cur = nxt
    if cur:
        out.append(cur)
    if len(out) > lines:
        out = out[:lines]
        while out and tw(out[-1] + " …", size) > px:
            out[-1] = out[-1][:-2]
        out[-1] += " …"
    return out or [""]


class Canvas:
    """The bit of SVG these figures need: text, boxes, lines, dots. Keeps nothing but a list of strings."""

    def __init__(self, width: int):
        self.w = width
        self.p: list[str] = []

    def text(self, x: float, y: float, t, size: float = 18, fill: str = INK, anchor: str = "start",
             weight: str = "normal", mono: bool = False) -> None:
        if str(t) == "":
            return
        self.p.append(f'<text x="{x:.0f}" y="{y:.0f}" font-family="{_MONO if mono else _FONT}" '
                      f'font-size="{size:g}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
                      f"{esc(t)}</text>")

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none", stroke: str | None = None,
             sw: float = 1.5, rx: float = 0, opacity: float | None = None) -> None:
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, w):.1f}" height="{max(0.0, h):.1f}" '
                      f'fill="{fill}"' + (f' fill-opacity="{opacity}"' if opacity is not None else "")
                      + (f' stroke="{stroke}" stroke-width="{sw}"' if stroke else "")
                      + (f' rx="{rx}"' if rx else "") + "/>")

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = RULE, sw: float = 1.5,
             dash: str | None = None) -> None:
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                      f'stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    def dot(self, cx: float, cy: float, r: float, fill: str, stroke: str | None = None, sw: float = 1.2,
            opacity: float | None = None) -> None:
        self.p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"'
                      + (f' fill-opacity="{opacity}"' if opacity is not None else "")
                      + (f' stroke="{stroke}" stroke-width="{sw}"' if stroke else "") + "/>")

    def chip(self, x: float, y: float, t, size: float = 15, fill: str = PANEL, ink: str = INK) -> float:
        """A labelled pill, left edge at x, baseline-ish centred on y. Returns its width."""
        w = tw(t, size) + 16
        self.rect(x, y - size, w, size * 1.62, fill=fill, rx=5)
        self.text(x + 8, y + size * 0.26, t, size, ink)
        return w

    def save(self, out, height: float) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{height:.0f}" '
                f'viewBox="0 0 {self.w} {height:.0f}"><rect width="{self.w}" height="{height:.0f}" fill="white"/>')
        out.write_text(head + "".join(self.p) + "</svg>")
        return out


def _leaves(val, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(val, dict):
        out: list[tuple[str, str]] = []
        for k, v in val.items():
            out += _leaves(v, f"{prefix}{k}.")
        return out
    if isinstance(val, list):
        return [(prefix.rstrip("."), " ".join(str(x) for x in val))]
    return [(prefix.rstrip("."), str(val))] if val not in (None, "") else []


def _latest(d: Dossier, *keys: str):
    for k in keys:
        if d.evidence.get(k):
            return d.evidence[k][-1].get("value")
    return None


def _text_of(val) -> str:
    return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ the chain map
def domains(d: Dossier, letters: dict[str, int]) -> tuple[list[tuple[str, int, int, str]], list[str]]:
    """(chain, first, last, label) per domain, and the fields whose ranges could not be placed.

    A range counts as a domain when its field name says so and both ends fit the chain. A field about the native
    sequence is skipped whole: `construct.native_region` on T1416 reads "res 9-414 = CotH residues ~186-591 ...
    N-terminal 1-185", three ranges in two numbering frames, and 1-185 fits the 414-residue input while meaning
    something else entirely. Those are listed under the figure instead of drawn."""
    drawn: list[tuple[str, int, int, str]] = []
    skipped: list[str] = []
    for key in ("identity", "construct", "domains", "architecture"):
        for e in d.evidence.get(key, [])[-1:]:
            for field, text in _leaves(e.get("value")):
                if not _DOMAIN_FIELD.search(field):
                    continue
                if _NOT_INPUT.search(field):
                    if _RANGE.search(text):
                        skipped.append(f"`{key}.{field}` names residue ranges in the native numbering")
                    continue
                letter = next((c for c in letters if f".{c}." in f".{field}." or field.startswith(f"{c}.")), None)
                for a, b in _RANGE.findall(text):
                    first, last = int(a), int(b)
                    fits = [c for c, L in letters.items() if last <= L and first >= 1 and (letter in (None, c))]
                    if first >= last or not fits:
                        skipped.append(f"`{key}.{field}` {first}-{last}")
                        continue
                    drawn.append((fits[0], first, last, f"{field.split('.')[-1]} {first}-{last}"))
    return drawn, sorted(set(skipped))


def sites(d: Dossier, letters: dict[str, int]) -> tuple[list[tuple[str, int, str]], list[str]]:
    """(chain, residue, key) per marked residue.

    Read the way `lantern gate --init` reads the same evidence - `site_from_value`, which prefers the numbering
    recorded as a field over the prose beside it - so the marks are the residues a drafted gate would use. A chain
    the text does not name is taken from a `chain` field, and a residue past the end of that chain is almost always
    native numbering, so it is reported, not drawn."""
    marks: list[tuple[str, int, str]] = []
    skipped: list[str] = []
    for rec in site_records(d, letters, skipped):
        marks += [(rec["chain"], r, rec["key"]) for r in rec["residues"]]
    return marks, sorted(set(skipped))


def site_records(d: Dossier, letters: dict[str, int], skipped: list[str] | None = None) -> list[dict]:
    """One record per site evidence key: {key, chain, residues, what, source, how}, residues in the chain's own
    numbering and only those that fit it. `skipped` collects what could not be placed, in words."""
    out: list[dict] = []
    notes = skipped if skipped is not None else []
    for key in sorted(k for k in d.evidence if SITE_KEYS.search(k)):
        if not d.evidence[key]:
            continue
        val = d.evidence[key][-1].get("value")
        text = _text_of(val)
        sel, guessed, how = site_from_value(val, target_ids(d.target))
        if not sel:
            notes.append(f"`{key}` names no residues that can be read")
            continue
        if guessed:
            hint = (val.get("chain") if isinstance(val, dict) else None) \
                or next(iter(re.findall(r"\bchain ([A-Z])\b", text)), None)
            if not hint:
                notes.append(f"`{key}` names residues but no chain: {sel.split(':', 1)[1][:60]}")
                continue
            sel = f"{hint}:{sel.split(':', 1)[1]}"
        for chain, res in parse_select(sel).items():
            L = letters.get(chain)
            if L is None:
                notes.append(f"`{key}` is on chain {chain}, which this FASTA does not have")
                continue
            if res is None:
                notes.append(f"`{key}` selects the whole of chain {chain}")
                continue
            keep = sorted(r for r in res if r <= L)
            notes += [f"`{key}` has {chain}:{r}, past residue {L} - native numbering?" for r in sorted(res) if r > L]
            if keep:
                out.append({"key": key, "chain": chain, "residues": keep, "how": how,
                            "what": _field(val, r"what|note|description|role|why|detail"),
                            "source": _field(val, r"source|reference|citation|pmid|doi")})
    return out


def _field(val, pattern: str) -> str:
    """The first field of a recorded value whose name matches, as text; "" when there is none."""
    rx = re.compile(pattern, re.I)
    if isinstance(val, dict):
        for k, v in val.items():
            if rx.search(str(k)) and v not in (None, "", [], {}):
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return ""


def chain_map_svg(d: Dossier, out, width: int = 1400, ddir=None) -> Path | None:
    """One row per chain: the sequence as a bar, the domains on it, the sites that become gates marked under it."""
    cs = chains(d)
    if not cs:
        return None
    letters = {chr(ord("A") + i): len(s) for i, (_, s) in enumerate(cs)}
    doms, dom_skipped = domains(d, letters)
    marks, site_skipped = sites(d, letters)
    if len(cs) < 2 and not doms and not marks:
        return None                                   # one chain and nothing on it is a ruler, not a figure
    keys = sorted({k for _, _, k in marks})
    colour = {k: COLS[i % len(COLS)] for i, k in enumerate(keys)}
    C = Canvas(width)

    x0, w = 250, width - 330
    row = 150 if marks else 104              # the label band under a bar is only needed when something is marked

    def _n(k):
        c = sum(1 for _, _, kk in marks if kk == k)
        return f"{k} ({c} residue{'s' if c != 1 else ''})"

    notes = ([f"sites: {', '.join(_n(k) for k in keys)}"] if keys else []) \
        + ([f"not placed: {'; '.join(dom_skipped + site_skipped)}"] if dom_skipped or site_skipped else [])
    for i, (name, seq) in enumerate(cs):
        letter, L = chr(ord("A") + i), len(seq)
        y = 70 + row * i
        fx = lambda r: x0 + w * (r - 1) / max(1, L - 1)                              # noqa: B023 - per-chain scale
        short = name.split(",", 1)[1] if name.startswith(f"{d.target},") and "," in name else name
        C.text(x0 - 16, y + 26, f"{letter} · {short}", 21, INK, "end", "bold")
        C.text(x0 - 16, y + 52, f"{L} aa", 18, MUTED, "end")
        C.rect(x0, y, w, 34, fill=PANEL, stroke=RULE)
        for first, last, label in [(a, b, lab) for c, a, b, lab in doms if c == letter]:
            C.rect(fx(first), y, max(2.0, fx(last) - fx(first)), 34, fill=COLS[0], stroke=COLS[0], opacity=0.18)
            C.text((fx(first) + fx(last)) / 2, y - 10, label, 17, COLS[0], "middle")
        step = 50 if L > 200 else (10 if L > 40 else 2)
        ticks = sorted(set(range(1, L + 1, step)) | {L})
        ticks = [r for j, r in enumerate(ticks) if r == L or ticks[j + 1] - r > step * 0.4]  # no crowding at the end
        for r in ticks:
            C.line(fx(r), y + 34, fx(r), y + 41, "#aeaeb2")
            C.text(fx(r), y + 62, r, 17, MUTED, "middle")
        here = sorted(((r, k) for c, r, k in marks if c == letter))
        last_x, level = -1e9, 0
        for r, key in here:
            c, x = colour[key], fx(r)
            level = level + 1 if x - last_x < 110 else 0                      # stagger only what would collide
            last_x = x
            C.line(x, y - 6, x, y + 40, c, 3)
            C.text(x, y + 86 + 22 * (level % 3), f"{seq[r - 1]}{r}", 17, c, "middle", "bold")
    for j, text in enumerate(notes):
        C.text(x0, 96 + row * len(cs) + 26 + 34 * j, text, 18, MUTED)
    return C.save(out, 96 + row * len(cs) + 34 * len(notes))


# ------------------------------------------------------------------------------------------------ the literature
_CLAIM_TEXT = re.compile(r"^(claim|finding|fact|statement|what|text|note|summary|says)$", re.I)
_SOURCE_F = re.compile(r"source|reference|^refs?$|citation|pmid|pmc|doi|paper", re.I)
_UNKNOWN_F = re.compile(r"not[_ ]?known|unknown|not[_ ]?found|no[_ ]?paper|gap|uncertain|open[_ ]?question", re.I)
_SEARCHED_F = re.compile(r"search|queried|looked", re.I)
_PDB_ID = re.compile(r"\b[1-9](?=[A-Za-z0-9]{3}\b)(?=[0-9]*[A-Za-z])[A-Za-z0-9]{3}\b")
_IDS = re.compile(r"PMID[:\s]*\d{4,}|PMC\d{4,}|10\.\d{4,}/[^\s,;)\]]+"
                  r"|\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b"
                  r"|\bPF\d{5}\b|\bIPR\d{6}\b", re.I)
_RESIDUE = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,3})\b")
_MOTIF = re.compile(r"\b(?:p?[A-Z]|\[[A-Z/]{2,8}\]|[Xx])(?:-(?:p?[A-Z]|\[[A-Z/]{2,8}\]|[Xx])){2,8}\b")


def tokens(text: str, skip: set[str] = frozenset()) -> set[str]:
    """The identifiers a claim and a prior can share: a paper, an accession, a PDB entry, a residue, a motif.
    Words are deliberately not shared - two entries both saying "dimer" is not a link worth drawing."""
    t = str(text)
    out = {m.group(0).upper().replace(" ", "").replace("PMID:", "PMID") for m in _IDS.finditer(t)}
    out |= {m.group(0).upper() for m in _PDB_ID.finditer(t)}
    out |= {f"{a}{n}" for a, n in _RESIDUE.findall(t)}
    out |= {m.group(0) for m in _MOTIF.finditer(t)}
    return {x for x in out if x not in skip and x.upper() not in {s.upper() for s in skip}}


def claims(val) -> list[dict]:
    """Every claim a literature-ish record carries: {label, text, source}. The skills record claims as a list of
    {claim, source}, but one-per-topic dicts and bare strings happen too, and all three read the same way."""
    out: list[dict] = []

    def one(x, label: str = "") -> None:
        if isinstance(x, dict):
            text = next((v for k, v in x.items() if _CLAIM_TEXT.match(str(k)) and isinstance(v, str)), "")
            if not text:
                strs = [v for k, v in x.items() if isinstance(v, str) and not _SOURCE_F.search(str(k))]
                text = max(strs, key=len) if strs else ""
            src = "; ".join(str(v) for k, v in x.items() if _SOURCE_F.search(str(k)) and v)
            rest = {k: v for k, v in x.items()
                    if not _SOURCE_F.search(str(k)) and (not isinstance(v, str) or v != text)}
            if text or rest:
                out.append({"label": label, "text": text or json.dumps(rest, ensure_ascii=False),
                            "source": src, "extra": rest})
        elif isinstance(x, str) and x.strip():
            out.append({"label": label, "text": x.strip(), "source": "", "extra": {}})

    if isinstance(val, dict):
        for k, v in val.items():
            if _UNKNOWN_F.search(str(k)) or _SEARCHED_F.search(str(k)):
                continue
            if isinstance(v, list):
                for x in v:
                    one(x)
            else:
                one(v, str(k))
    elif isinstance(val, list):
        for x in val:
            one(x)
    else:
        one(val)
    return [c for c in out if c["text"]]


def _fields_matching(val, rx: re.Pattern) -> list[str]:
    out: list[str] = []
    if isinstance(val, dict):
        for k, v in val.items():
            if rx.search(str(k)) and v not in (None, "", [], {}):
                out.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            else:
                out += _fields_matching(v, rx)
    elif isinstance(val, list):
        for x in val:
            out += _fields_matching(x, rx)
    return out


def rests_on(d: Dossier, claim_tokens: set[str]) -> list[str]:
    """Which of this dossier's other priors and decisions share an identifier with a claim. A shared PMID, PDB
    entry, accession, residue or motif is a link a reader can check; a shared word is not."""
    skip = target_ids(d.target)
    hits: list[str] = []
    for k, entries in d.evidence.items():
        if not entries or re.search(r"literature|background|reading|report", k, re.I):
            continue
        if claim_tokens & tokens(_text_of(entries[-1].get("value")), skip):
            hits.append(f"`{k}`")
    for dec in d.decisions:
        text = " ".join(dec.get("evidence", []) + [dec.get("chose", "")] + list(dec.get("alternatives", [])))
        if claim_tokens & tokens(text, skip):
            hits.append(dec["id"])
    return hits


def literature_svg(d: Dossier, out, width: int = 1400, ddir=None) -> Path | None:
    """One row per claim: what was read, where it came from, and which priors and decisions share an identifier
    with it. Under them, what the reading says is *not* known - the claim a plan is most often wrong about."""
    val = _latest(d, "literature", "background", "reading")
    cl = claims(val)
    if not cl:
        return None
    skip = target_ids(d.target)
    unknown = _fields_matching(val, _UNKNOWN_F)
    searched = _fields_matching(val, _SEARCHED_F)
    C = Canvas(width)
    x0, num_w, claim_w = 40, 44, int(width * 0.55)
    chip_x = x0 + num_w + claim_w + 40
    y = 58
    C.text(x0, y, f"{len(cl)} claim{'s' if len(cl) != 1 else ''} recorded under "
                  f"`{'literature' if d.evidence.get('literature') else 'background'}`, with the source each rests "
                  "on", 19, INK, weight="bold")
    C.text(chip_x, y, "shares an identifier with", 17, MUTED)
    y += 18
    C.line(x0, y, width - 40, y, RULE)
    y += 34
    for i, c in enumerate(cl, 1):
        tk = tokens(c["text"] + " " + c["source"], skip)
        on = rests_on(d, tk)
        body = wrap(c["text"], claim_w, 17, 5)
        src = wrap(c["source"] or "no source recorded", claim_w, 15, 2)
        top = y - 22
        C.text(x0 + 4, y, i, 19, MUTED, weight="bold")
        if c["label"]:
            C.text(x0 + num_w, y, c["label"], 17, COLS[0], weight="bold")
            y += 24
        for ln in body:
            C.text(x0 + num_w, y, ln, 17, INK)
            y += 23
        for ln in src:
            C.text(x0 + num_w, y, ln, 15, MUTED if c["source"] else FLAG)
            y += 20
        cy, cx = top + 22, chip_x
        for chip in on[:8] or ["nothing else names it"]:
            wch = tw(chip, 15) + 16
            if cx + wch > width - 40:
                cx, cy = chip_x, cy + 28
            cx += C.chip(cx, cy, chip, 15, PANEL if on else "white", INK if on else MUTED) + 8
        y = max(y, cy + 26) + 16
        if i < len(cl):
            C.line(x0, y - 12, width - 40, y - 12, "#ebebf0", 1)
    y += 4
    for ln in wrap("A claim is linked to a prior or a decision when both name the same paper, structure, "
                   "accession, residue or motif — never when they merely use the same word.", width - x0 - 60, 15):
        C.text(x0, y, ln, 15, MUTED)
        y += 21
    for label, texts, colour in (("Not settled by anything read", unknown, FLAG), ("Searched", searched, MUTED)):
        if not texts:
            continue
        y += 10
        C.text(x0, y, label, 18, colour, weight="bold")
        y += 26
        for t in texts:
            for ln in wrap(t, width - x0 - 60, 16, 4):
                C.text(x0 + 4, y, ln, 16, MUTED)
                y += 22
        y += 6
    return C.save(out, y + 20)


# ------------------------------------------------------------------------------------------------ the sites
def modifications(d: Dossier) -> tuple[list[tuple[str, int, str]], list[str]]:
    """(chain, residue, label) for each chemistry the FASTA does not carry, and the records that name no residue.
    Read from `chemistry`-ish evidence: "phosphoserine (CCD SEP) on chain A residue 7" or "A:7"."""
    out: list[tuple[str, int, str]] = []
    unplaced: list[str] = []
    for key in sorted(k for k in d.evidence if re.search(r"chemistry|modification|ptm|ligand", k, re.I)):
        if not d.evidence[key]:
            continue
        for field, text in _leaves(d.evidence[key][-1].get("value")) or []:
            if re.search(r"source|basis|reference|candidates|run_as|why", field, re.I):
                continue
            m = re.search(r"chain\s+([A-Z])\b[^.;]{0,40}?\bresidues?\s+(\d{1,4})", text, re.I) \
                or re.search(r"\b([A-Z]):(\d{1,4})\b", text)
            label = re.split(r"\s+on\s+|\s+at\s+", text, maxsplit=1)[0][:40]
            if m:
                out.append((m.group(1), int(m.group(2)), label))
            elif re.search(r"phospho|acetyl|methyl|glyco|ccd|sep|tpo|ptr|metal|ion|ligand", text, re.I):
                unplaced.append(f"`{key}.{field}` names chemistry but no residue: {text[:70]}")
    return out, sorted(set(unplaced))


def _groups(res: list[int], L: int, pad: int = 6, gap: int = 12) -> list[tuple[int, int, list[int]]]:
    """Residues collected into windows: (first, last, residues). Two sites 12 apart share one strip; further
    apart they get one each, because the sequence between them is not what the record is about."""
    out: list[tuple[int, int, list[int]]] = []
    for r in sorted(res):
        if out and r - out[-1][2][-1] <= gap:
            first, _, rs = out.pop()
            out.append((first, min(L, r + pad), rs + [r]))
        else:
            out.append((max(1, r - pad), min(L, r + pad), [r]))
    return out


def sites_svg(d: Dossier, out, width: int = 1400, ddir=None) -> Path | None:
    """Each recorded site in its own sequence: the residues as letters, the site marked, what the record says it
    does, and where that came from. The numbers are the ones a gate would use (`lantern gate --init`)."""
    cs = chains(d)
    if not cs:
        return None
    letters = {chr(ord("A") + i): len(s) for i, (_, s) in enumerate(cs)}
    seqs = {chr(ord("A") + i): s for i, (_, s) in enumerate(cs)}
    names = {chr(ord("A") + i): n for i, (n, _) in enumerate(cs)}
    skipped: list[str] = []
    recs = site_records(d, letters, skipped)
    if not recs:
        return None
    mods, mod_skipped = modifications(d)
    C = Canvas(width)
    x0, box, bh, text_w = 56, 30, 38, width - 120
    y = 56
    C.text(x0, y, f"{len(recs)} site{'s' if len(recs) != 1 else ''} recorded, residue by residue in the "
                  "construct's own numbering", 19, INK, weight="bold")
    y += 16
    C.line(x0, y, width - 40, y, RULE)
    y += 34
    for i, rec in enumerate(recs):
        colour = COLS[i % len(COLS)]
        seq, L = seqs[rec["chain"]], letters[rec["chain"]]
        nm = names[rec["chain"]]
        nm = nm.split(",", 1)[1] if nm.startswith(f"{d.target},") and "," in nm else nm
        head = f"{rec['key']} — chain {rec['chain']} ({nm}), " \
               f"{', '.join(f'{seq[r - 1]}{r}' for r in rec['residues'])}"
        C.text(x0, y, head, 19, colour, weight="bold")
        C.text(width - 40, y, f"read as: {rec['how']}", 15, MUTED, "end")
        y += 14
        for first, last, rs in _groups(rec["residues"], L):
            y += 30
            C.text(x0, y + bh / 2 + 6, f"{rec['chain']}:{first}", 15, MUTED, "end")
            x = x0 + 10
            last_label = -1e9
            for r in range(first, last + 1):
                hit = r in rs
                mod = next((m for m in mods if m[0] == rec["chain"] and m[1] == r), None)
                C.rect(x, y, box - 2, bh, fill=colour if hit else PANEL, stroke=colour if hit else RULE,
                       opacity=1.0 if hit else None, rx=3)
                C.text(x + (box - 2) / 2, y + bh / 2 + 7, seq[r - 1], 19, "white" if hit else INK, "middle",
                       "bold" if hit else "normal", mono=True)
                if hit:                       # neighbouring numbers would run together: drop the second one a line
                    mid = x + (box - 2) / 2
                    low = mid - tw(r, 15) / 2 < last_label + 6
                    C.text(mid, y + bh + (40 if low else 20), r, 15, colour, "middle", "bold")
                    if not low:
                        last_label = mid + tw(r, 15) / 2
                if mod:
                    C.text(x + (box - 2) / 2, y - 8, "✱", 17, FLAG, "middle")
                x += box
            C.text(x + 10, y + bh / 2 + 6, f"…{L}" if last < L else f"end ({L})", 15, MUTED)
            y += bh + 46
        y += 22                                  # the residue numbers hang under the strip: leave them their line
        for ln in wrap(rec["what"] or "no description recorded", text_w, 17, 4):
            C.text(x0 + 10, y, ln, 17, INK if rec["what"] else FLAG)
            y += 23
        for ln in wrap(f"source: {rec['source']}" if rec["source"] else "no source recorded", text_w, 15, 2):
            C.text(x0 + 10, y, ln, 15, MUTED if rec["source"] else FLAG)
            y += 20
        y += 18
        if i < len(recs) - 1:
            C.line(x0, y - 14, width - 40, y - 14, "#ebebf0", 1)
    marks = [f"✱ {lab} on {ch}{r}" for ch, r, lab in mods if ch in letters]
    for t in marks + [f"not placed: {x}" for x in sorted(set(skipped + mod_skipped))]:
        for ln in wrap(t, width - x0 - 60, 16, 3):
            C.text(x0, y, ln, 16, MUTED)
            y += 22
    return C.save(out, y + 16)


# ------------------------------------------------------------------------------------------------ the motif
def _pattern(tok: str) -> list[tuple[str, set[str]]]:
    """'pS-X-P' -> [('pS', {'S'}), ('X', any), ('P', {'P'})]; an empty set means any residue."""
    out: list[tuple[str, set[str]]] = []
    for t in tok.split("-"):
        if t.startswith("[") and t.endswith("]"):
            out.append((t, {c for c in t[1:-1] if c.isalpha()}))
        elif t.upper() == "X":
            out.append((t, set()))
        else:
            out.append((t, {t[-1].upper()}))
    return out


def motifs(d: Dossier) -> list[dict]:
    """Every motif the analysis wrote down, with where it came from and the residue it was recorded on:
    {token, pattern, key, chain, recorded, text}."""
    out: list[dict] = []
    seen: set[str] = set()
    keys = [k for k in d.evidence if re.search(r"motif|chemistry|binding|site|literature|epitope", k, re.I)]
    for key in sorted(keys, key=lambda k: (not re.search(r"motif", k, re.I), k)):
        if not d.evidence[key]:
            continue
        val = d.evidence[key][-1].get("value")
        text = _text_of(val)
        # a site record is about one chain and says so; a claim out of a paper mentions chains in passing, so a
        # pattern read there is scanned against every chain and the figure says which key it came from
        site = bool(SITE_KEYS.search(key)) or bool(re.search(r"chemistry|modification", key, re.I))
        chain = (val.get("chain") if isinstance(val, dict) else None) \
            or (next(iter(re.findall(r"\bchain ([A-Z])\b", text)), None) if site else None)
        rec = val.get("residues") if isinstance(val, dict) else None
        for m in _MOTIF.finditer(text):
            tok = m.group(0)
            if len(tok.split("-")) < 3 or tok in seen:      # the same pattern in a paper and in the site record
                continue                                    # is one pattern: the site record names the chain
            seen.add(tok)
            out.append({"token": tok, "pattern": _pattern(tok), "key": key, "chain": chain,
                        "recorded": [int(r) for r in rec or [] if isinstance(r, int)],
                        "text": text})
    return out


def hits(pattern: list[tuple[str, set[str]]], seq: str) -> list[int]:
    """Every start position (1-based) where the pattern matches the sequence."""
    n = len(pattern)
    return [i + 1 for i in range(len(seq) - n + 1)
            if all(not allowed or seq[i + j].upper() in allowed for j, (_, allowed) in enumerate(pattern))]


def candidates(d: Dossier) -> dict[str, str]:
    """{'S7': 'chosen: pS-X-P motif ...'} from a candidates/alternatives field: the residues the analysis
    considered and what it decided about each."""
    out: dict[str, str] = {}
    for key in sorted(d.evidence):
        if not d.evidence[key] or not isinstance(d.evidence[key][-1].get("value"), dict):
            continue
        for k, v in d.evidence[key][-1]["value"].items():
            if re.search(r"candidate|alternative|option|considered", str(k), re.I) and isinstance(v, dict):
                for res, why in v.items():
                    if re.fullmatch(r"[A-Z]\d{1,4}", str(res)) and isinstance(why, str):
                        out.setdefault(str(res), why)
    return out


def motif_svg(d: Dossier, out, width: int = 1400, ddir=None) -> Path | None:
    """The motifs the analysis named, scanned against the sequence they were named on: where each can sit, which
    position was recorded as the site, and - from the candidates field - what was decided about the others.

    This is the arithmetic behind a sentence like "S7 is in pS-X-P and S5 is not", done here rather than trusted."""
    cs = chains(d)
    ms = motifs(d)
    if not cs or not ms:
        return None
    letters = {chr(ord("A") + i): len(s) for i, (_, s) in enumerate(cs)}
    seqs = {chr(ord("A") + i): s for i, (_, s) in enumerate(cs)}
    cands = candidates(d)
    rows: list[dict] = []
    nowhere: list[dict] = []
    for m in ms:
        found = False
        for ch in ([m["chain"]] if m["chain"] in seqs else sorted(seqs)):
            h = hits(m["pattern"], seqs[ch])
            if h:
                rows.append({**m, "chain": ch, "hits": h})
                found = True
        if not found:
            nowhere.append(m)                   # a consensus the target does not satisfy is worth one line: on
    if not rows:                                # H1388 the peptide is a partial mode-I match, and that is why
        return None
    C = Canvas(width)
    x0, box = 56, 30
    y = 56
    C.text(x0, y, f"{len(rows)} motif/sequence pair{'s' if len(rows) != 1 else ''}: every place the recorded "
                  "pattern can sit, counted from the FASTA", 19, INK, weight="bold")
    y += 16
    C.line(x0, y, width - 40, y, RULE)
    y += 40
    for i, r in enumerate(rows):
        colour = COLS[i % len(COLS)]
        seq, L = seqs[r["chain"]], letters[r["chain"]]
        n = len(r["pattern"])
        # the anchor is the position the record is about: the modified one if the pattern marks one (pS-X-P), else
        # the first. A match is "the recorded site" only when its anchor is the residue that was recorded - on
        # H1388, S-A-S starting at S5 covers S7 without being about it.
        anchor = next((j for j, (t, _) in enumerate(r["pattern"]) if t[:1] == "p"), 0)
        chosen = [h for h in r["hits"] if h + anchor in r["recorded"]]
        C.text(x0, y, f"{r['token']} on chain {r['chain']} ({L} aa) — from `{r['key']}`", 19, colour, weight="bold")
        C.text(width - 40, y, f"{len(r['hits'])} place{'s' if len(r['hits']) != 1 else ''} it fits"
               + (f", recorded on {', '.join(f'{seq[x - 1]}{x}' for x in r['recorded'])}" if r["recorded"] else ""),
               15, MUTED, "end")
        y += 18
        windows = _groups(sorted({h for h in r["hits"]} | set(r["recorded"])), L, pad=3, gap=10)
        full = L <= (width - x0 - 160) // box
        windows = [(1, L, sorted(set(r["hits"]) | set(r["recorded"])))] if full else windows[:4]
        for first, last, _ in windows:
            y += 28
            C.text(x0, y + 19 + 6, f"{r['chain']}:{first}", 15, MUTED, "end")
            x = x0 + 10
            for pos in range(first, last + 1):
                start = next((h for h in r["hits"] if h <= pos < h + n), None)
                here = start is not None
                strong = start is not None and start in chosen
                fill = colour if strong else (PANEL if not here else "white")
                C.rect(x, y, box - 2, 38, fill=fill, stroke=colour if here else RULE, rx=3,
                       sw=2.0 if here else 1.2)
                C.text(x + (box - 2) / 2, y + 26, seq[pos - 1], 19, "white" if strong else INK, "middle",
                       "bold" if here else "normal", mono=True)
                if pos in r["recorded"] or (here and pos == start + anchor):
                    C.text(x + (box - 2) / 2, y + 58, pos, 15, colour if here else MUTED, "middle",
                           "bold" if pos in r["recorded"] else "normal")
                x += box
            if last < L:
                C.text(x + 10, y + 25, f"…{L}", 15, MUTED)
            y += 82                              # 38 of box, then the position numbers under it, then air
        shown = {h for first, last, _ in windows for h in r["hits"] if first <= h <= last}
        extra = [h for h in r["hits"] if h not in shown]
        lines = ["pattern: " + " · ".join(f"{t} = {'any residue' if not a else '/'.join(sorted(a))}"
                                          for t, a in r["pattern"])]
        if chosen:
            lines.append("filled: the match whose "
                         + ("modified position" if anchor or r["token"][:1] == "p" else "first residue")
                         + " is the residue the analysis recorded as the site. Outlined: the same pattern "
                           "elsewhere in this chain, which the record does not claim.")
        else:
            lines.append("nothing is filled: no match here starts at a residue this evidence recorded as the site.")
        if extra:
            lines.append(f"{len(extra)} further match{'es' if len(extra) != 1 else ''} outside these windows: "
                         + ", ".join(f"{seq[h - 1]}{h}" for h in extra[:12]))
        for res, why in (cands.items() if i == 0 else []):           # the candidate list is about the residue,
            pos = int(res[1:])                                       # not about one of the patterns: say it once
            if pos <= L and seqs[r["chain"]][pos - 1] == res[0]:
                verdict = why.split(":", 1)[0].strip().lower()
                mark = "✓" if verdict.startswith(("chosen", "yes", "keep")) else "✗" if verdict else "·"
                lines.append(f"{mark} {res}: {why}")
        for t in lines:
            for ln in wrap(t, width - x0 - 60, 16, 3):
                C.text(x0 + 10, y, ln, 16, MUTED)
                y += 22
        y += 20
        if i < len(rows) - 1:
            C.line(x0, y - 14, width - 40, y - 14, "#ebebf0", 1)
    if nowhere:
        C.line(x0, y - 14, width - 40, y - 14, "#ebebf0", 1)
        for ln in wrap("Fits nowhere in any chain: "
                       + ", ".join(f"{m['token']} (from `{m['key']}`)" for m in nowhere)
                       + " — a consensus the target matches only in part.", width - x0 - 60, 16, 3):
            C.text(x0, y, ln, 16, MUTED)
            y += 22
    return C.save(out, y + 10)


# ------------------------------------------------------------------------------------------------ the clustering
_CLUSTER_NAME = re.compile(r"^(?P<runner>[a-z0-9_]+?)_seed(?P<seed>\d+)_sample(?P<sample>\d+)$", re.I)


def cluster_points(ddir, sto: str | None = None) -> list[dict]:
    """Every model lantern-qa clustered, with the coordinates it placed them at: {sto, variant, method, runner,
    seed, sample, cluster, x, y, key}. Read from `qa/cluster_<sto>*/<sto>/<variant>/<method>/results/
    *clustering.csv`, which carries the cluster, the membership columns and a 2-D UMAP of the TM-score space."""
    out: list[dict] = []
    qa = Path(ddir) / "qa"
    if not qa.is_dir():
        return out
    for cd in sorted(qa.glob(f"cluster_{sto}*" if sto else "cluster_*")):
        if not cd.is_dir():
            continue
        for p in sorted(cd.rglob("*clustering.csv")):
            rel = p.relative_to(cd).parts                     # (<sto>, <run>, <method>, results, file)
            s = rel[0] if rel else (sto or "?")
            run, method = (rel[1], rel[2]) if len(rel) >= 5 else ("?", "?")
            try:
                rows = list(csv.DictReader(p.read_text().splitlines()))
            except OSError:
                continue
            for r in rows:
                raw = (r.get("structure") or r.get("name") or "").removesuffix(".pdb").removesuffix(".cif")
                # one variant at a time: `af3_seed1_sample0` under a directory named for the variant. Several
                # variants clustered together: `af3_seed1_sample0__af3_msa_pS548` under `pool-<hash>`, and then
                # the name is the only place the variant is written.
                name, _, suffix = raw.partition("__")
                variant = suffix or run
                m = _CLUSTER_NAME.fullmatch(name)
                x, y = r.get("umap1"), r.get("umap2")
                if not m or r.get("cluster") in (None, ""):
                    continue
                try:
                    pt = {"x": float(x), "y": float(y)} if x not in (None, "") and y not in (None, "") else {}
                except ValueError:
                    pt = {}
                try:                       # how strongly this model belongs to its cluster: the highest value is
                    pt["mem"] = float(r["max_membership"])     # the cluster's medoid, the pose worth looking at
                except (KeyError, TypeError, ValueError):
                    pass
                from lantern.slate import key as model_key          # the name the pool, the slate and QA share
                out.append({"sto": s, "run": run, "variant": variant, "method": method,
                            "runner": m["runner"].lower(), "seed": int(m["seed"]), "sample": int(m["sample"]),
                            "cluster": str(r["cluster"]),
                            "key": model_key(s, variant, m["runner"].lower(), m["seed"], m["sample"]), **pt})
    return out


def pool_scores(ddir) -> dict[str, dict]:
    """Per model: its own confidence from pool.csv and, when lantern-qa has scored the interfaces, its predicted
    DockQ. Both are keyed by the model name the slate uses, so a point on the clustering can be looked up."""
    from lantern.slate import _model_key, load_interface
    out: dict[str, dict] = {}
    p = Path(ddir) / "pool.csv"
    if not p.exists():
        return out
    try:
        rows = list(csv.DictReader(p.read_text().splitlines()))
    except OSError:
        return out
    for r in rows:
        try:
            out[_model_key(r)] = {"rs": float(r["ranking_score"]) if r.get("ranking_score") else None,
                                  "iptm": float(r["iptm"]) if r.get("iptm") else None, "xgb": None,
                                  "path": r.get("path") or ""}
        except (KeyError, ValueError):
            continue
    qa = Path(ddir) / "qa"
    if qa.is_dir():
        for sto in sorted({r.split("__")[0] for r in out}):
            for k, v in load_interface(qa, sto).items():
                if k in out:
                    out[k]["xgb"] = v
    return out


def slate_models(ddir) -> dict[str, str]:
    """{model name: slot label} for the latest slate, so the clustering can show where the slate sits."""
    out: dict[str, str] = {}
    for f in sorted(Path(ddir).glob("slate_v*.json"))[-1:]:
        try:
            s = json.loads(f.read_text())
        except (OSError, ValueError):
            return out
        for m in s.get("models") or []:
            out[str(m.get("model", ""))] = f"M{m.get('slot', '?')}"
    return out


VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]


def ramp(t: float, stops: list[str] = VIRIDIS) -> str:
    """A perceptual-ish sequential colour for t in 0..1 (viridis stops, interpolated in sRGB)."""
    t = 0.0 if t != t else max(0.0, min(1.0, t))                       # noqa: PLR0124 - NaN guard
    i = min(int(t * (len(stops) - 1)), len(stops) - 2)
    f = t * (len(stops) - 1) - i
    a, b = stops[i].lstrip("#"), stops[i + 1].lstrip("#")
    rgb = [round(int(a[j:j + 2], 16) * (1 - f) + int(b[j:j + 2], 16) * f) for j in (0, 2, 4)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


def cluster_table(pts: list[dict], scores: dict[str, dict]) -> list[dict]:
    """One row per cluster, ordered the way a person reads them - biggest first, and everything that decides a
    slate on the row: how many models, which variants and predictors are in it, the best confidence and the best
    and mean predicted DockQ, and the model at its centre (the highest cluster membership = the medoid)."""
    by: dict[tuple[str, str], list[dict]] = {}
    for p in pts:
        by.setdefault((p.get("run", ""), p["cluster"]), []).append(p)
    rows: list[dict] = []
    for (run, cl), ps in by.items():
        rs = [((scores.get(p["key"]) or {}).get("rs"), p) for p in ps]
        rs = [(v, p) for v, p in rs if v is not None]
        xgb = [((scores.get(p["key"]) or {}).get("xgb"), p) for p in ps]
        xgb = [(v, p) for v, p in xgb if v is not None]
        mem = [p for p in ps if "mem" in p]
        rows.append({
            "run": run, "cluster": cl, "n": len(ps), "points": ps,
            "variants": _count(ps, "variant"), "runners": _count(ps, "runner"),
            "best_rs": max(rs, key=lambda t: t[0]) if rs else (None, None),
            "best_xgb": max(xgb, key=lambda t: t[0]) if xgb else (None, None),
            "mean_xgb": (sum(v for v, _ in xgb) / len(xgb)) if xgb else None,
            "medoid": max(mem, key=lambda p: p["mem"]) if mem else (ps[0] if ps else None),
        })
    return sorted(rows, key=lambda r: (-r["n"], r["cluster"]))


def _count(ps: list[dict], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in ps:
        out[p[field]] = out.get(p[field], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def cluster_notes(pts: list[dict], scores: dict[str, dict]) -> list[str]:
    """What the clustering says, in sentences, from the numbers alone: a cluster only one variant reaches, where
    the predictors agree, and whether the biggest cluster is the one the scores like. These are the lines a person
    would otherwise have to work out from the picture."""
    rows = cluster_table(pts, scores)
    if not rows:
        return []
    out: list[str] = []
    variants = {p["variant"] for p in pts}
    for r in rows:
        if len(variants) > 1 and len(r["variants"]) == 1 and r["n"] >= 10:
            v = next(iter(r["variants"]))
            share = sum(1 for p in pts if p["variant"] == v)
            out.append(f"cluster {r['cluster']} is {v} and nothing else ({r['n']:,} of the {share:,} models that "
                       f"variant has): that input reaches a shape the others do not")
    agree = [r for r in rows if len(r["runners"]) >= 3]
    if agree:
        best = agree[0]
        out.append(f"{len(agree)} cluster{'s' if len(agree) != 1 else ''} hold models from three or more "
                   f"predictors; the largest is {best['cluster']} ("
                   + ", ".join(f"{k} {v:,}" for k, v in best["runners"].items()) + ")")
    alone = [r for r in rows if len(r["runners"]) == 1 and r["n"] >= 10]
    if alone:
        out.append("one predictor only: " + ", ".join(f"cluster {r['cluster']} ({next(iter(r['runners']))}, "
                                                      f"{r['n']:,})" for r in alone[:4])
                   + " — a shape no other predictor found")
    ranked = [r for r in rows if r["mean_xgb"] is not None]
    if ranked:
        top = max(ranked, key=lambda r: r["mean_xgb"])
        if top["cluster"] != rows[0]["cluster"]:
            out.append(f"the best cluster by predicted DockQ ({top['cluster']}, mean {top['mean_xgb']:.2f}) is not "
                       f"the biggest one ({rows[0]['cluster']}, {rows[0]['n']:,} models): size is not quality")
        else:
            out.append(f"the biggest cluster is also the best by predicted DockQ (mean {top['mean_xgb']:.2f})")
    return out


def cluster_medoids(ddir, limit: int = 8) -> list[dict]:
    """The model at the centre of each of the biggest clusters, with the numbers that describe its cluster:
    {label, cluster, n, key, path, conf, xgb, variants, runners}.

    A cluster is a claim that these models are the same shape; the medoid is the one to look at to see what that
    shape is. Ordered by size, because that is the order the ranking figure uses."""
    pts = [p for p in cluster_points(ddir) if "x" in p]
    if not pts:
        return []
    scores = pool_scores(ddir)
    out: list[dict] = []
    for r in cluster_table(pts, scores)[:limit]:
        m = r["medoid"]
        if not m:
            continue
        s = scores.get(m["key"]) or {}
        if not s.get("path"):
            continue
        out.append({"label": f"C{r['cluster']}", "cluster": r["cluster"], "n": r["n"], "key": m["key"],
                    "path": s["path"], "sto": m["sto"], "conf": s.get("rs"), "xgb": s.get("xgb"),
                    "mean_xgb": r["mean_xgb"], "variants": r["variants"], "runners": r["runners"],
                    "best_conf": r["best_rs"][0]})
    return out


def _panel(C: Canvas, px: float, py: float, pw: float, ph: float, ps: list[dict],
           colour, title: str, right: str = "", dim: list[dict] | None = None,
           ring: set[str] = frozenset(), star: dict[str, str] | None = None) -> None:
    """One UMAP panel: `dim` points in light grey behind, `ps` coloured by `colour(p)`, the models in `ring`
    circled and the ones in `star` marked and labelled."""
    if right and tw(title, 17, bold=True) + tw(right, 14) + 30 <= pw:
        C.text(px, py - 10, title, 17, INK, weight="bold")
        C.text(px + pw, py - 10, right, 14, MUTED, "end")       # both fit on the header line
    else:
        C.text(px, py - 30, title, 17, INK, weight="bold")      # otherwise the note gets its own line
        C.text(px, py - 11, wrap(right, pw, 14, 1)[0], 14, MUTED)
    C.rect(px, py, pw, ph, fill="#fbfbfd", stroke=RULE, sw=1.2)
    allp = (dim or []) + ps
    xs = [p["x"] for p in allp if "x" in p]
    ys = [p["y"] for p in allp if "y" in p]
    if not xs:
        C.text(px + pw / 2, py + ph / 2, "no coordinates", 15, MUTED, "middle")
        return
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda v: px + 13 + (pw - 26) * (v - x0) / (x1 - x0 or 1)
    sy = lambda v: py + ph - 13 - (ph - 26) * (v - y0) / (y1 - y0 or 1)
    for p in dim or []:
        if "x" in p:
            C.dot(sx(p["x"]), sy(p["y"]), 3.0, "#e5e5ea")
    for p in ps:
        if "x" in p:
            C.dot(sx(p["x"]), sy(p["y"]), 3.8, colour(p), opacity=0.8)
    for p in ps:
        if p["key"] in ring and "x" in p:
            C.dot(sx(p["x"]), sy(p["y"]), 7.0, "none", stroke=INK, sw=1.8)
    for p in ps:
        if star and p["key"] in star and "x" in p:
            C.text(sx(p["x"]), sy(p["y"]) + 7, "★", 20, FLAG, "middle")
            C.text(sx(p["x"]) + 12, sy(p["y"]) + 6, star[p["key"]], 14, FLAG, weight="bold")


def _colourbar(C: Canvas, x: float, y: float, h: float, lo: float, hi: float, label: str) -> None:
    """A vertical ramp with its two ends labelled: a continuous colour means nothing without its scale."""
    n = 40
    for i in range(n):
        C.rect(x, y + h - (i + 1) * h / n, 13, h / n + 0.6, fill=ramp(i / (n - 1)))
    C.rect(x, y, 13, h, fill="none", stroke=RULE, sw=1)
    C.text(x + 18, y + 11, f"{hi:.2f}", 13, MUTED)
    C.text(x + 18, y + h, f"{lo:.2f}", 13, MUTED)
    C.text(x + 6, y - 9, label, 13, MUTED, "middle")


def clusters_svg(d: Dossier, out, width: int = 1400, ddir=None) -> Path | None:
    """The pool as lantern-qa clustered it, in the four views that decide a slate:

      · the clusters themselves, with the medoid of each ringed and any slate model starred;
      · the same points coloured by predicted DockQ, and by the predictor's own confidence - so a cluster that is
        big, a cluster that is confident and a cluster that is good can be told apart, and they are often not the
        same cluster;
      · where each predictor's models sit, which is the "do they agree" question drawn rather than argued;
      · the clusters ranked by mean predicted DockQ, with what is in each.

    The axes are UMAP coordinates of a TM-score distance matrix: they have no units, so none are drawn."""
    pts = [p for p in (cluster_points(ddir) if ddir else []) if "x" in p]
    if not pts:
        return None
    scores = pool_scores(ddir) if ddir else {}
    slate = slate_models(ddir) if ddir else {}
    rows = cluster_table(pts, scores)
    colour = {(r["run"], r["cluster"]): (COLS[i % len(COLS)] if i < 7 else "#aeaeb2") for i, r in enumerate(rows)}
    ring = {r["medoid"]["key"] for r in rows if r["medoid"]}
    xg = [v for v in ((scores.get(p["key"]) or {}).get("xgb") for p in pts) if v is not None]
    cf = [v for v in ((scores.get(p["key"]) or {}).get("rs") for p in pts) if v is not None]
    runners = sorted({p["runner"] for p in pts}, key=lambda r: -sum(1 for p in pts if p["runner"] == r))

    pad, pw, ph, gap = 40, 400, 330, 46
    bar_h = 26
    width = pad * 2 + pw * 3 + gap * 2
    head = 86
    C = Canvas(width)
    sto = sorted({p["sto"] for p in pts})[0]
    C.text(pad, 48, f"{len({p['key'] for p in pts}):,} {sto} models, {len(rows)} clusters "
                    f"({sorted({p['method'] for p in pts})[0]}-score), {len({p['variant'] for p in pts})} input "
                    f"variants, {len(runners)} predictors", 20, INK, weight="bold")
    for ln in wrap("Every panel is the same UMAP of the same models; only the colouring changes. ● medoid of its "
                   "cluster (highest membership) ringed; ★ a model already on a slate. Size, confidence and "
                   "predicted DockQ rarely pick the same cluster, which is why the slate takes one model from "
                   "each of the first three.", width - pad * 2, 15):
        C.text(pad, head - 12, ln, 15, MUTED)
        head += 20
    head += 44

    # row 1: the three colourings of the whole pool
    _panel(C, pad, head, pw, ph, pts, lambda p: colour[(p.get("run", ""), p["cluster"])],
           "Clusters", f"{len(rows)} clusters", ring=ring, star=slate)
    if xg:
        lo, hi = min(xg), max(xg)
        px2 = pad + pw + gap
        _panel(C, px2, head, pw - 46, ph, [p for p in pts if (scores.get(p["key"]) or {}).get("xgb") is not None],
               lambda p: ramp(((scores[p["key"]]["xgb"]) - lo) / ((hi - lo) or 1)),
               "Predicted DockQ (lantern-qa)", f"{len(xg):,} scored",
               dim=[p for p in pts if (scores.get(p["key"]) or {}).get("xgb") is None], star=slate)
        _colourbar(C, px2 + pw - 40, head + 16, ph - 42, lo, hi, "DockQ")
    if cf:
        lo, hi = min(cf), max(cf)
        px3 = pad + (pw + gap) * 2
        _panel(C, px3, head, pw - 46, ph, [p for p in pts if (scores.get(p["key"]) or {}).get("rs") is not None],
               lambda p: ramp(((scores[p["key"]]["rs"]) - lo) / ((hi - lo) or 1)),
               "The predictor's own confidence", "not comparable between predictors",
               dim=[p for p in pts if (scores.get(p["key"]) or {}).get("rs") is None], star=slate)
        _colourbar(C, px3 + pw - 40, head + 16, ph - 42, lo, hi, "rank")
    head += ph + 58

    # row 2: one small panel per predictor - where each one lives
    sw = (width - pad * 2 - gap * (len(runners) - 1)) / max(1, len(runners))
    sh = min(230.0, sw * 0.8)
    C.text(pad, head - 16, "Where each predictor's models sit", 17, INK, weight="bold")
    C.text(width - pad, head - 16, "grey: the rest of the pool", 14, MUTED, "end")
    head += 34
    for i, run in enumerate(runners):
        mine = [p for p in pts if p["runner"] == run]
        cl = _count(mine, "cluster")
        _panel(C, pad + i * (sw + gap), head, sw, sh, mine,
               lambda p: colour[(p.get("run", ""), p["cluster"])],
               f"{run} · {len(mine):,}", f"{len(cl)} cluster{'s' if len(cl) != 1 else ''}",
               dim=[p for p in pts if p["runner"] != run])
    head += sh + 52

    # row 3: the clusters ranked, with what is in each
    rank = sorted(rows, key=lambda r: (-(r["mean_xgb"] if r["mean_xgb"] is not None else -1),
                                       -(r["best_rs"][0] or 0)))[:9]
    metric = "mean predicted DockQ" if any(r["mean_xgb"] is not None for r in rank) else "best confidence"
    C.text(pad, head, f"Clusters ranked by {metric}", 17, INK, weight="bold")
    C.text(width - pad, head, "bar: mean · ◆ best in the cluster", 14, MUTED, "end")
    head += 26
    lab_w, bar_w = 130, 300
    top = max(max(r["mean_xgb"] or 0, (r["best_xgb"][0] or 0) if r["mean_xgb"] is not None else 0,
                  r["best_rs"][0] or 0 if r["mean_xgb"] is None else 0) for r in rank) or 1.0
    for r in rank:
        v = r["mean_xgb"] if r["mean_xgb"] is not None else (r["best_rs"][0] or 0)
        best = r["best_xgb"][0] if r["mean_xgb"] is not None else r["best_rs"][0]
        c = colour[(r["run"], r["cluster"])]
        C.text(pad + lab_w - 10, head + 18, f"cluster {r['cluster']}", 15, INK, "end", "bold")
        C.rect(pad + lab_w, head + 2, bar_w, bar_h, fill=PANEL, rx=3)
        C.rect(pad + lab_w, head + 2, bar_w * v / top, bar_h, fill=c, rx=3)
        if best is not None and best > v:
            C.dot(pad + lab_w + bar_w * best / top, head + 15, 4.5, "none", stroke=INK, sw=1.6)
        C.text(pad + lab_w + bar_w + 56, head + 19, f"{v:.3f}", 14, INK, "end")      # a column, not at the bar
        if best is not None and best > v:                                            # end, where the ◆ sits
            C.text(pad + lab_w + bar_w + 118, head + 19, f"◆ {best:.3f}", 13, MUTED, "end")
        call = ""                      # a callout on the row that decides something, the way an annotated chart
        if len({p["variant"] for p in pts}) > 1 and len(r["variants"]) == 1:    # earns its place
            call = f"only {next(iter(r['variants']))}"
        elif len(r["runners"]) >= 3:
            call = f"{len(r['runners'])} predictors agree here"
        elif len(r["runners"]) == 1 and r["n"] >= 10:
            call = f"{next(iter(r['runners']))} alone"
        elif r is rank[0]:
            call = "best by this score"
        if call:
            C.text(pad + lab_w + 10, head + 19, call, 13, "white" if bar_w * v / top > tw(call, 13) + 20 else INK,
                   weight="bold")
        bits = [f"{r['n']:,} models"]
        if len(r["variants"]) > 1 or len({p["variant"] for p in pts}) > 1:
            bits.append(" / ".join(f"{k} {n:,}" for k, n in r["variants"].items()))
        bits.append(" / ".join(f"{k} {n:,}" for k, n in r["runners"].items()))
        if r["best_rs"][1] is not None:
            bits.append(f"best {r['best_rs'][1]['key'].split('__', 1)[-1]} ({r['best_rs'][0]:.3f})")
        C.text(pad + lab_w + bar_w + 134, head + 19,
               wrap(" · ".join(bits), width - pad * 2 - lab_w - bar_w - 140, 14, 1)[0], 14, MUTED)
        head += bar_h + 12
    head += 10

    for t in cluster_notes(pts, scores):
        for ln in wrap("— " + t, width - pad * 2, 15, 2):
            C.text(pad, head + 14, ln, 15, INK)
            head += 21
        head += 4
    return C.save(out, head + 22)

# ------------------------------------------------------------------------------------------------ all of them
FIGURES: tuple[tuple[str, str], ...] = (
    ("chain_map", "Each chain to scale, with the domains that were recorded and the residues a gate would use"),
    ("literature", "Every claim the analysis read, its source, and the priors and decisions that share it"),
    ("sites", "Each recorded site in its own sequence, with what the record says it does"),
    ("motif", "Every place the recorded motifs can sit in the sequence they were recorded on"),
    ("clusters", "The pool clustered, in the four views that decide a slate: the clusters, the predicted DockQ, the confidence, and where each predictor sits"),
)
_DRAW = {"chain_map": chain_map_svg, "literature": literature_svg, "sites": sites_svg, "motif": motif_svg,
         "clusters": clusters_svg}


def draw_all(d: Dossier, ddir, names: tuple[str, ...] | None = None) -> tuple[dict[str, Path], list[str]]:
    """Draw every figure the dossier has the evidence for into `<ddir>/report/figs/`, and remove the ones it no
    longer has: a figure left behind from an earlier run would be linked as if it were current.

    Returns (drawn, problems). A figure that raises is a problem reported, never a failed report."""
    figs = Path(ddir) / "report" / "figs"
    drawn: dict[str, Path] = {}
    problems: list[str] = []
    for name, _ in FIGURES:
        if names and name not in names:
            continue
        out = figs / f"{name}.svg"
        try:
            got = _DRAW[name](d, out, ddir=ddir)
        except Exception as exc:                        # a figure is never worth failing the report for
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            got = None
        if got is None and out.exists():
            out.unlink()
        if got is not None:
            drawn[name] = got
    return drawn, problems
