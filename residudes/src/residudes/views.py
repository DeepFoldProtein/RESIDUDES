"""What the deck shows before the models: the MSAs as ColabFold draws them, and the templates as structures.

Both come from files the run already has - the a3m files under msas/, and the templates the prediction inputs name
or the MSA search found - so every deck has them without anyone asking for them. A reader looks at the depth of an
MSA and at the closest structures before anything else about a target; a sentence saying "10,377 sequences" or
"6S9K, 100 %" is a poorer version of either picture.

  msa_figure(ddir, d, out)            figs/msa_depth.svg + figs/msa_depth.json (per chain and source: N, Neff)
  template_entries(cfg, d, ddir)      report/templates/<ID>_<chain>.cif, trimmed to the chain and what touches it
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import struct
import urllib.request
import zlib
from pathlib import Path

from residudes.figures import INK, MUTED, RULE, Canvas
from residudes.msa import _match, read_a3m

# numpy's BLAS takes every core it sees, and the deck is built on a login node: measured on H1388's 10,377-row MSA,
# 4 threads finish in the same 2.5 s that 40 do, at a tenth of the CPU. Only effective before numpy's first import.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

# --------------------------------------------------------------------------------------------------- MSAs
# ColabFold's "Sequence coverage" plot, which anyone who has run a structure prediction can read: one row per
# sequence, coloured by its identity to the query (rainbow_r: red far, violet close), white where it has a gap,
# sorted by identity, and the number of sequences covering each position as a black line. Two additions: the rows
# come in blocks by where they came from - ColabFold's UniRef search, its environmental databases (BFD, MGnify,
# MetaEuk, SMAG), paired rows - each block labelled with its count and Neff; and the chain's Neff is in the title.
AA = "ACDEFGHIKLMNPQRSTVWY"
GROUPS = ("paired", "UniRef", "environmental")              # top to bottom
NEFF_ID = 0.8                                              # the usual threshold (trRosetta, RoseTTAFold, AF2's Neff)
NEFF_EXACT_MAX = 3000                                      # above this, estimated from a sample of rows
NEFF_SAMPLE = 2000


def msa_files(ddir: Path, d) -> dict[tuple[str, str], dict[str, Path]]:
    """{(source, chain): {"unpaired": a3m, "paired": a3m}} under msas/, one or two levels down: residudes-engine
    writes msas/<source>/msa/unpaired_<chain>.a3m, residudes's own MSAs sit at msas/<name>/<file>.a3m."""
    roots = [Path(ddir) / "msas"]
    if d.sequence.get("source_dir"):
        roots.append(Path(d.sequence["source_dir"]) / "msas")
    out: dict = {}
    for r in roots:
        if not r.is_dir():
            continue
        for p in sorted([*r.glob("*/*.a3m"), *r.glob("*/*/*.a3m")]):
            if p.name.startswith("templates") or any(part.startswith("_") for part in p.relative_to(r).parts):
                continue                                   # template hits, and ColabFold's raw per-database files
            m = re.match(r"(unpaired|paired)_([A-Za-z0-9]+)\.a3m$", p.name)
            kind, chain = (m.group(1), m.group(2)) if m else ("unpaired", p.stem)
            out.setdefault((p.relative_to(r).parts[0], chain), {}).setdefault(kind, p)
    return out


def source_of(header: str, kind: str) -> str:
    if kind == "paired":
        return "paired"
    return "UniRef" if header.split()[0].startswith("UniRef") else "environmental"


def _codes(query: str, rows: list[str]):
    import numpy as np
    lut = np.full(256, 21, np.uint8)                       # 21: anything else (X, B, Z ...)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    lut[ord("-")] = 20                                     # 20: gap
    L = len(query)
    buf = np.frombuffer("".join(rows).encode("ascii", "replace"), np.uint8) if rows else np.zeros(0, np.uint8)
    return lut[buf].reshape(len(rows), L), lut[np.frombuffer(query.encode(), np.uint8)]


def neff(codes, threshold: float = NEFF_ID, seed: int = 0) -> tuple[float, bool]:
    """Sum over sequences of 1 / (sequences at >= `threshold` identity to it, itself included), identity over the
    query's columns with a gap matching a gap - the weighting trRosetta and RoseTTAFold use. Exact up to
    NEFF_EXACT_MAX rows; above, NEFF_SAMPLE rows are weighed against all of them, which is an unbiased estimate
    and keeps a 10,000-row MSA to seconds. Returns (Neff, estimated)."""
    import numpy as np
    n, L = codes.shape
    if n == 0:
        return 0.0, False
    est = n > NEFF_EXACT_MAX
    pick = np.random.default_rng(seed).choice(n, NEFF_SAMPLE, replace=False) if est else np.arange(n)
    hits = np.zeros(len(pick), np.int64)
    for lo in range(0, len(pick), 500):                   # 500 x N at a time: memory stays near 20 MB per 10k rows
        rows = pick[lo:lo + 500]
        same = np.zeros((len(rows), n), np.float32)
        for a in range(22):
            m = codes == a
            if not m.any():
                continue
            fm = m.astype(np.float32)
            same += fm[rows] @ fm.T
        hits[lo:lo + 500] = (same >= threshold * L - 1e-3).sum(1)
    total = float((1.0 / np.maximum(hits, 1)).sum())
    return (total * n / len(pick) if est else total), est


def _rainbow_r(t):
    """matplotlib's rainbow_r, which ColabFold's plot uses: t = 1 (identical) violet, t = 0 red."""
    import numpy as np
    x = 1.0 - np.clip(t, 0.0, 1.0)
    return np.stack([np.clip(np.abs(2 * x - 0.5), 0, 1), np.clip(np.sin(np.pi * x), 0, 1),
                     np.clip(np.cos(np.pi * x / 2), 0, 1)], -1)


def _png(rgb) -> bytes:
    """An RGB uint8 array as PNG bytes (stdlib zlib; no imaging library)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def chain_panel(files: dict[str, Path]) -> dict | None:
    """Everything one panel needs for one chain: rows grouped by source and sorted by identity, Neff per group."""
    import numpy as np
    query, seqs, groups = None, [], []
    for kind in ("paired", "unpaired"):
        if kind not in files:
            continue
        q, rows = read_a3m(files[kind])
        query = query or q
        for name, s in rows:
            m = _match(s).upper()
            if len(m) == len(q) and q == query:
                seqs.append(m)
                groups.append(source_of(name, kind))
    if query is None or not seqs:
        return None
    codes, qc = _codes(query, seqs)
    L = len(query)
    ident = (codes == qc[None, :]).sum(1) / L              # ColabFold's identity: over the whole query length
    nongap = codes != 20
    blocks = []
    for g in GROUPS:
        idx = np.array([i for i, x in enumerate(groups) if x == g], int)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-ident[idx], kind="stable")]
        nf, est = neff(codes[idx])
        blocks.append({"name": g, "idx": idx, "n": len(idx), "neff": nf, "neff_estimated": est,
                       "n_cov50": int((nongap[idx].sum(1) >= 0.5 * L).sum())})
    order = np.concatenate([b["idx"] for b in blocks])
    nf, est = neff(codes)
    return {"length": L, "n": len(seqs), "neff": nf, "neff_estimated": est, "blocks": blocks,
            "ident": ident[order], "nongap": nongap[order], "coverage": nongap.sum(0)}


def _raster(p: dict, w: int, h: int) -> bytes:
    """The rows binned onto h pixel rows (top = first block, highest identity), columns stretched onto w."""
    import numpy as np
    n, L = p["nongap"].shape
    ng = p["nongap"].astype(np.float32)
    wid = ng * p["ident"][:, None]
    if n >= h:
        edges = (np.arange(h) * n) // h
        cnt = np.diff(np.append(edges, n))[:, None].astype(np.float32)
        frac, idsum = np.add.reduceat(ng, edges, 0) / cnt, np.add.reduceat(wid, edges, 0)
        mean_id = idsum / np.maximum(np.add.reduceat(ng, edges, 0), 1e-6)
    else:
        pick = (np.arange(h) * n) // h
        frac, mean_id = ng[pick], p["ident"][pick][:, None] * np.ones((1, L))
    cols = (np.arange(w) * L) // w
    frac, mean_id = frac[:, cols], mean_id[:, cols]
    rgb = _rainbow_r(mean_id) * frac[..., None] + (1.0 - frac[..., None])     # a gap is white
    return _png((rgb * 255 + 0.5).astype(np.uint8))


def _fmt_neff(v: float, est: bool) -> str:
    return ("≈" if est else "") + (f"{v:,.0f}" if v >= 10 else f"{v:.1f}")


def msa_figure(ddir: Path, d, out: Path) -> dict | None:
    """Draw figs/msa_depth.svg (a ColabFold-style panel per chain and MSA source) and write its numbers beside it
    as msa_depth.json. Returns the numbers, or None when there is no MSA on disk to draw from - in which case an
    existing figure is left as it is (the shipped example has the figure and not the MSAs)."""
    files = msa_files(ddir, d)
    if not files:
        return None
    stamp = {str(p): [p.stat().st_size, int(p.stat().st_mtime)] for f in files.values() for p in f.values()}
    js = out.with_suffix(".json")
    if out.exists() and js.exists():
        try:
            old = json.loads(js.read_text())
            if old.get("files") == stamp:
                return old                                  # nothing changed: a deck build stays seconds
        except (OSError, ValueError):
            pass
    panels, alone = [], []
    for (source, chain), fs in sorted(files.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        p = chain_panel(fs)
        if p is None:
            alone.append(f"{chain} ({source})")
        else:
            panels.append((source, chain, p))
    stats = {"files": stamp, "query_only": alone, "neff_identity": NEFF_ID, "panels": [
        {"source": s, "chain": c, "length": p["length"], "n": p["n"], "neff": round(p["neff"], 1),
         "neff_estimated": p["neff_estimated"],
         "groups": [{k: (round(b[k], 1) if k == "neff" else b[k]) for k in ("name", "n", "n_cov50", "neff", "neff_estimated")}
                    for b in p["blocks"]]} for s, c, p in panels]}
    if not panels:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(stats, indent=1))
        if out.exists():
            out.unlink()                                    # MSAs on disk, none with a homologue: no picture
        return stats
    W, cols = 1400, (1 if len(panels) == 1 else 2)
    pw, ph = W / cols, 560
    C = Canvas(W)
    for k, (source, chain, p) in enumerate(panels):
        px, py = (k % cols) * pw, (k // cols) * ph
        x0, y0, w, h = px + 96, py + 70, pw - 96 - (250 if cols == 1 else 190), 400
        C.text(px + 20, py + 30, f"chain {chain} · {source} · {p['length']} residues", 22, INK, weight="bold")
        C.text(px + 20, py + 56, f"{p['n']:,} sequences · Neff {_fmt_neff(p['neff'], p['neff_estimated'])} "
               f"(≥{NEFF_ID:.0%} identity)", 19, MUTED)
        png = _raster(p, int(w * 2), int(h * 2))
        C.p.append(f'<image x="{x0:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{h:.0f}" preserveAspectRatio="none" '
                   f'href="data:image/png;base64,{base64.b64encode(png).decode()}"/>')
        C.rect(x0, y0, w, h, stroke=INK, sw=1.2)
        n, L = p["n"], p["length"]
        pts = " ".join(f"{x0 + w * (i + 0.5) / L:.1f},{y0 + h - h * c / n:.1f}" for i, c in enumerate(p["coverage"]))
        C.p.append(f'<polyline points="{pts}" fill="none" stroke="black" stroke-width="2.2"/>')
        top = 0                                             # the blocks, top to bottom, each named at its right
        for b in p["blocks"]:
            yb = y0 + h * top / n
            if top:
                C.line(x0, yb, x0 + w, yb, stroke=INK, sw=2)
            mid = y0 + h * (top + b["n"] / 2) / n
            C.text(x0 + w + 10, mid - 4, b["name"], 17, INK, weight="bold")
            C.text(x0 + w + 10, mid + 17, f"{b['n']:,} · Neff {_fmt_neff(b['neff'], b['neff_estimated'])}", 15, MUTED)
            top += b["n"]
        for t in range(5):                                  # sequences, 0 at the bottom as in ColabFold
            v = round(n * t / 4)
            C.line(x0 - 6, y0 + h - h * t / 4, x0, y0 + h - h * t / 4, stroke=INK, sw=1.2)
            C.text(x0 - 10, y0 + h - h * t / 4 + 6, f"{v:,}", 15, MUTED, "end")
        step = max(10, int(round(L / 5 / 10.0)) * 10) if L > 20 else 5
        for pos in [1, *range(step, L + 1, step)]:
            C.line(x0 + w * (pos - 0.5) / L, y0 + h, x0 + w * (pos - 0.5) / L, y0 + h + 6, stroke=INK, sw=1.2)
            C.text(x0 + w * (pos - 0.5) / L, y0 + h + 26, str(pos), 15, MUTED, "middle")
        C.text(x0 + w / 2, y0 + h + 50, "position", 16, MUTED, "middle")
        C.p.append(f'<text x="{px + 26:.0f}" y="{y0 + h / 2:.0f}" font-family="DejaVu Sans, Helvetica, Arial" '
                   f'font-size="16" fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {px + 26:.0f} '
                   f'{y0 + h / 2:.0f})">sequences</text>')
    rows = math.ceil(len(panels) / cols)
    yb = rows * ph + 10                                     # one colour bar for all panels, and what is not drawn
    stops = "".join(f'<stop offset="{t / 10:.1f}" stop-color="rgb({",".join(str(int(v * 255)) for v in _rainbow_r(t / 10))})"/>'
                    for t in range(11))
    C.p.append(f'<defs><linearGradient id="id">{stops}</linearGradient></defs>')
    C.rect(96, yb, 360, 18, fill="url(#id)", stroke=RULE, sw=1)
    C.text(96, yb + 42, "0", 15, MUTED, "middle")
    C.text(456, yb + 42, "1", 15, MUTED, "middle")
    C.text(476, yb + 15, "identity to the query; white = gap; black line = sequences covering each position", 16, MUTED)
    if alone:
        C.text(96, yb + 72, "query alone, no homologue: " + ", ".join(alone), 16, MUTED)
    C.save(out, yb + (92 if alone else 60))
    js.write_text(json.dumps(stats, indent=1))
    return stats


# --------------------------------------------------------------------------------------------------- templates
# The structures a prediction leans on, drawn. Which ones: every template the prediction inputs name (they were
# used), then the template search's hits at >= 50 % identity, to six. Each is the full PDB entry trimmed to the
# template chain and whatever touches it - a bound peptide, a partner, a ligand - because what a high-identity
# template has bound is usually the point (H1388: 6S9K carries the phosphopeptide the target is about).
TEMPLATE_MIN_ID = 0.5
_TID = re.compile(r"(?<![0-9A-Za-z])([0-9][0-9A-Za-z]{3})[_-]([0-9A-Za-z]{1,4})\.cif$", re.I)


def _seq(chain) -> str:
    import gemmi
    return gemmi.one_letter_code([r.name for r in chain])


def _identity(a: str, b: str) -> tuple[float, float]:
    """Identity over the aligned columns, and the coverage of `a` (the target chain), by gemmi's global alignment."""
    import gemmi
    if not a or not b:
        return 0.0, 0.0
    al = gemmi.align_string_sequences(list(a), list(b), [])
    m = al.match_string
    pairs = sum(1 for x, y in zip(al.add_gaps(a, 1), al.add_gaps(b, 2)) if x != "-" and y != "-")
    return (m.count("|") / pairs if pairs else 0.0), pairs / len(a)


def _entry_file(cfg, pdb: str, cache: Path) -> Path | None:
    mirror = (cfg.extra.get("databases") or {}).get("af3_db") if cfg is not None else None
    if mirror and (Path(mirror) / "mmcif_files" / f"{pdb.lower()}.cif").exists():
        return Path(mirror) / "mmcif_files" / f"{pdb.lower()}.cif"
    got = cache / f"{pdb.upper()}.cif"
    if got.exists() and got.stat().st_size > 0:
        return got
    try:
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb.upper()}.cif", timeout=60) as r:
            data = r.read()
    except OSError:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    got.write_bytes(data)
    return got


def _candidates(d, ddir: Path) -> list[dict]:
    """Templates named by the prediction inputs (used), then the MSA search's hits; one per PDB entry."""
    seqs = {}
    from residudes.report import _chains
    from residudes.dossier import chain_letter
    for i, (_, s) in enumerate(_chains(d)):
        seqs[chain_letter(i)] = s
    found: dict[str, dict] = {}
    for js in sorted(Path(ddir).glob("inputs/**/*.json")):
        try:
            spec = json.loads(js.read_text())
        except (OSError, ValueError):
            continue
        for s in spec.get("sequences", []) if isinstance(spec, dict) else []:
            prot = s.get("protein") or {}
            ids = prot.get("id")
            for t in prot.get("templates") or []:
                path = t.get("mmcifPath")
                m = _TID.search(Path(path).name) if path else None
                if not m:
                    continue
                key = m.group(1).upper()
                e = found.setdefault(key, {"pdb": key, "chain": m.group(2), "for": ids[0] if isinstance(ids, list) else ids,
                                           "used": [], "file": path})
                if js.parent.name not in e["used"]:
                    e["used"].append(js.parent.name)
    for m8 in sorted(Path(ddir).glob("msas/**/templates_*.m8")):
        chain = m8.stem.split("_", 1)[1]
        for line in m8.read_text().splitlines():
            f = line.split("\t")
            if len(f) < 10 or "_" not in f[1]:
                continue
            pdb, ch = f[1].split("_", 1)
            ident = float(f[2])
            if ident >= TEMPLATE_MIN_ID and pdb.upper() not in found:
                found[pdb.upper()] = {"pdb": pdb.upper(), "chain": ch, "for": chain, "used": [], "hit_identity": ident}
    return [e | {"query": seqs.get(e["for"], "")} for e in found.values()]


def template_entries(cfg, d, ddir: Path, limit: int = 6) -> list[dict]:
    """Choose, fetch and trim the templates to draw; write report/templates/<ID>_<chain>.cif and templates.json.
    A template that cannot be fetched or read is skipped and named in the json, never a failed deck."""
    import gemmi
    rep = Path(ddir) / "report"
    out_dir, cache = rep / "templates", Path(ddir) / "dl" / "pdb_entries"
    chosen, skipped = [], []
    cands = _candidates(d, ddir)
    for e in cands:                                         # identity first, so the order below means something
        e["identity_hint"] = e.get("hit_identity", 1.0 if e["used"] else 0.0)
    cands.sort(key=lambda e: (not e["used"], -e["identity_hint"], e["pdb"]))
    for e in cands:
        if len(chosen) >= limit:
            break
        src = _entry_file(cfg, e["pdb"], cache)
        try:
            st = gemmi.read_structure(str(src or e.get("file") or ""))
        except (RuntimeError, ValueError, OSError):
            skipped.append(f"{e['pdb']}: not readable")
            continue
        st.setup_entities()
        st.remove_waters()
        model = st[0]
        names = [c.name for c in model]
        ch = e["chain"] if e["chain"] in names else next((c.name for c in model if c.name.upper() == e["chain"].upper()), None)
        if ch is None:
            ch = names[0] if src is None and names else None     # a single-chain template file from the inputs
        if ch is None:
            skipped.append(f"{e['pdb']}: no chain {e['chain']}")
            continue
        ns = gemmi.NeighborSearch(model, st.cell, 5).populate()
        touching: dict[str, int] = {}
        for res in model[ch]:
            for atom in res:
                for mark in ns.find_atoms(atom.pos, "\0", radius=4.5):
                    cra = mark.to_cra(model)
                    if cra.chain.name != ch:
                        touching[cra.chain.name] = touching.get(cra.chain.name, 0) + 1
        # What touches the template chain, minus other copies of it: in a crystal those are mostly packing
        # neighbours (6BYJ F touches three, 161 to 1,432 atoms, and only one of them is the dimer), and the picture
        # is about what the template has bound - a peptide, a partner protein, a ligand.
        own = _seq(model[ch].get_polymer())
        keep = {ch}
        for c, k in touching.items():
            n_res = len(model[c].get_polymer())
            same, cover = _identity(own, _seq(model[c].get_polymer()))
            if same >= 0.9 and cover >= 0.8:                # a copy of the template chain, however short
                continue
            if k >= (100 if n_res > 40 else 5):
                keep.add(c)
        for c in [c.name for c in model if c.name not in keep]:
            model.remove_chain(c)
        while len(st) > 1:
            del st[1]
        ident, cov = _identity(e["query"], _seq(model[ch].get_polymer()))
        partners = []
        for c in model:
            if c.name == ch:
                continue
            poly = c.get_polymer()
            ligs = sorted({r.name for r in c.get_ligands()})
            partners.append({"chain": c.name, "residues": len(poly), "ligands": ligs})
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{e['pdb']}_{ch}.cif"
        st.make_mmcif_document().write_file(str(path))
        chosen.append({"label": f"tpl_{e['pdb']}_{ch}", "pdb": e["pdb"], "chain": ch, "for": e["for"],
                       "used": e["used"], "identity": round(ident, 3), "coverage": round(cov, 3),
                       "resolution": round(st.resolution, 2) if st.resolution else None,
                       "method": (st.info["_exptl.method"].strip("'\" ").lower() if "_exptl.method" in st.info else None),
                       "partners": partners, "path": str(path),
                       "small": [p["chain"] for p in partners if 0 < p["residues"] <= 40]})
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "figs").mkdir(parents=True, exist_ok=True)
    (rep / "figs" / "templates.json").write_text(json.dumps({"templates": chosen, "skipped": skipped}, indent=1))
    return chosen


def template_caption(t: dict) -> str:
    """One line under a template picture: what it is, how close, and what it carries."""
    bits = [f"{t['pdb']} {t['chain']} → {t['for']}", f"{t['identity']:.0%} identity"]
    if t.get("resolution"):
        bits.append(f"{t['resolution']:.2f} Å")
    if t.get("used"):
        bits.append("used: " + ", ".join(t["used"]))
    for p in t.get("partners", []):
        what = f"{p['residues']} aa" if p["residues"] else ""
        if p["ligands"]:
            what = (what + " + " if what else "") + ", ".join(p["ligands"][:3])
        bits.append(f"+ {p['chain']} ({what})" if what else f"+ {p['chain']}")
    return " · ".join(bits)
