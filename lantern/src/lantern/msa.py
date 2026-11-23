"""`lantern msa T`: measure an MSA, and deepen a thin one, the two ways that worked in CASP17.

  lantern msa T --depth [A3M ...]                  depth and coverage of the target's MSAs -> evidence msa_depth
  lantern msa T --crawl --family PF03936 [--family IPR008949 ...] [--db uniref90 --db mgnify] [--max-members 20000]
  lantern msa T --plm [--aligner otalign]          plmMSA (plmmsa.deepfold.org): PLM-embedding search + OTalign

--crawl is the T1331 recipe: the default AF3 search found nothing but the query (depth 1) and the fold came out wrong
(lDDT 0.33); an MSA collected from the protein family (InterPro IPR008949, four Pfam families, hmmsearch over
UniRef90 and MGnify; 26,835 sequences) gave the right fold (model 1 lDDT 0.71, TM 0.86). Steps:
  1. on the login node (light HTTP): each family's Pfam HMM and seed alignment from the InterPro API (an InterPro entry
     expands to its Pfam members), and the family's UniProt members from the UniProt REST API (capped);
  2. one sbatch job per sequence database: hmmsearch of every family HMM against it (UniRef90 72 GB, MGnify 129 GB;
     one job per database so each big file is streamed by one job at a time);
  3. one sbatch job after those (`python -m lantern.msa assemble DIR`): pool the hits, members and seed sequences,
     cluster at 90 % identity (mmseqs), align every cluster representative straight to the query with OTalign
     (PLM embeddings, plmMSA's public /v2/align/pairwise; compute nodes reach it), drop near-duplicates (hhfilter)
     -> msas/crawl/custom.a3m and provenance.json. `--aligner hmm` aligns through the family HMMs instead (hmmalign,
     re-indexed on the query): offline, but for remote families most of the query lands in insert states
     (T1331: 63 of 281 residues covered, against a median of 184 with OTalign).
Then `lantern msa T --depth` records how much deeper it is; the agent adds it as an input variant (lantern-engine custom MSA).

The family is the agent's call: from UniProt/InterPro annotations of the target or its closest relatives, HHpred hits,
foldseek hits of a predicted model, or the literature (T1331: a terpene synthase -> isoprenoid synthase domain).
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Site defaults: where these things are on the DeepFold cluster. Every one of them is a fallback - `lantern.toml`
# wins ([msa] for the databases, [tools] for the binaries and the plmMSA client), and a binary that is on PATH
# wins over a path here that does not exist. Running elsewhere means editing that file, not this one.
DBS = {"uniref90": "/gpfs/database/af3/uniref90_2022_05.fa", "mgnify": "/gpfs/database/af3/mgy_clusters_2022_05.fa",
       "uniprot": "/gpfs/database/af3/uniprot_all_2021_04.fa"}
TOOLS = {"hmmsearch": "/gpfs/deepfold/.local/homebrew/bin/hmmsearch", "hmmalign": "/gpfs/deepfold/.local/homebrew/bin/hmmalign",
         "mmseqs": "/gpfs/deepfold/others/mmseqs/bin/mmseqs", "hhfilter": "/home/deepfold/bin/hhfilter"}
UA = {"User-Agent": "lantern-msa/1.0"}
PLMMSA = "/gpfs/deepfold/casp/casp17/msas/plmmsa/submit_target.sh"


def tool(cfg, name: str) -> str | None:
    p = (cfg.extra.get("tools", {}) if cfg else {}).get(name) or TOOLS.get(name)
    return p if p and Path(p).exists() else shutil.which(name)


# ------------------------------------------------------------------------------------------------- a3m and depth
def read_a3m(path: Path) -> tuple[str, list[tuple[str, str]]]:
    rows, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    rows.append((name, "".join(buf)))
                name, buf = line[1:], []
            elif line and not line.startswith("#"):
                buf.append(line.strip())
    if name is not None:
        rows.append((name, "".join(buf)))
    if not rows:
        raise ValueError(f"{path}: empty a3m")
    return rows[0][1].replace("-", "").upper(), rows[1:]


def _match(seq: str) -> str:
    """The query-length row of an a3m line: drop insertions (lowercase and '.')."""
    return "".join(c for c in seq if not (c.islower() or c == "."))


def depth(path: Path) -> dict:
    query, rows = read_a3m(path)
    L = len(query)
    cov = [0] * L
    n50, ids = 0, []
    for _, s in rows:
        m = _match(s)
        if len(m) != L:
            continue
        aligned = [i for i, c in enumerate(m) if c != "-"]
        for i in aligned:
            cov[i] += 1
        if len(aligned) >= 0.5 * L:
            n50 += 1
        if aligned:
            ids.append(sum(1 for i in aligned if m[i] == query[i]) / len(aligned))
    thin = max(10, int(0.02 * len(rows)))
    gaps, start = [], None                                    # query stretches (>= 15 residues) with almost no coverage
    for i, c in enumerate(cov + [thin]):
        if c < thin and start is None:
            start = i
        elif c >= thin and start is not None:
            if i - start >= 15:
                gaps.append(f"{start + 1}-{i}")
            start = None
    return {"path": str(path), "length": L, "n": len(rows), "n_cov50": n50, "coverage_min": min(cov) if cov else 0,
            "coverage_median": int(statistics.median(cov)) if cov else 0,
            "median_identity": round(statistics.median(ids), 3) if ids else None, "thin_regions": gaps,
            "coverage": cov}


def coverage_svg(stats: list[dict], out: Path, width: int = 1400, height: int = 560) -> Path:
    """Coverage per query position for one or more MSAs (stdlib SVG, drawn large; Typst and GitHub show it as is)."""
    cols = ["#0072B2", "#8e8e93", "#D55E00", "#009E73", "#CC79A7"]
    L = max(s["length"] for s in stats)
    top = max(max(s["coverage"] or [1]) for s in stats) or 1
    x0, y0, w, h = 110, 30, width - 160, 300
    fx = lambda i: x0 + w * i / max(1, L - 1)
    fy = lambda c: y0 + h - h * c / top
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         f'<rect width="{width}" height="{height}" fill="white"/>']
    T = lambda x, y, t, size=20, fill="#1d1d1f", anchor="start", weight="normal": P.append(
        f'<text x="{x:.0f}" y="{y:.0f}" font-family="DejaVu Sans, Helvetica, Arial" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{t}</text>')
    for k in range(5):                                      # y grid: 0 .. max sequences
        c = top * k / 4
        P.append(f'<line x1="{x0}" y1="{fy(c):.0f}" x2="{x0 + w}" y2="{fy(c):.0f}" stroke="#e5e5ea" stroke-width="1.5"/>')
        T(x0 - 12, fy(c) + 7, f"{int(c):,}", 18, "#6e6e73", "end")
    step = 50 if L > 200 else 25
    for pos in range(0, L + 1, step):
        T(fx(max(0, pos - 1)), y0 + h + 30, str(max(1, pos)), 18, "#6e6e73", "middle")
    for k, s in enumerate(stats):
        pts = " ".join(f"{fx(i):.1f},{fy(c):.1f}" for i, c in enumerate(s["coverage"]))
        P.append(f'<polyline points="{pts}" fill="none" stroke="{cols[k % len(cols)]}" stroke-width="3.5"/>')
    T(x0 + w / 2, y0 + h + 62, f"query position (1–{L})", 20, "#6e6e73", "middle")
    for k, s in enumerate(stats):                           # legend under the plot, one line per MSA
        y = y0 + h + 112 + 34 * k
        P.append(f'<line x1="{x0}" y1="{y - 7}" x2="{x0 + 40}" y2="{y - 7}" stroke="{cols[k % len(cols)]}" stroke-width="6"/>')
        name = f'{Path(s["path"]).parent.name}/{Path(s["path"]).name}'
        T(x0 + 54, y, f'{name}: N {s["n"]:,} · ≥50% coverage {s["n_cov50"]:,}'
          + (f' · thin {", ".join(s["thin_regions"])}' if s["thin_regions"] else ""), 21, cols[k % len(cols)], weight="bold")
    P.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(P).replace('height="560"', f'height="{y0 + h + 112 + 34 * len(stats)}"', 1)
                   .replace(f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {y0 + h + 112 + 34 * len(stats)}"', 1))
    return out


def target_msas(cfg, d) -> list[Path]:
    """The MSAs this target already has: the lantern-engine tree (targets_root/T/msas) and lantern's own (dossier/msas)."""
    from lantern.dossier import Dossier
    out = []
    roots = [Dossier.path(cfg.targets_dir, d.target).parent / "msas"]
    if d.sequence.get("source_dir"):
        roots.append(Path(d.sequence["source_dir"]) / "msas")
    for r in roots:
        if r.is_dir():
            for sub in sorted(r.iterdir()):
                if sub.is_dir():
                    a3ms = sorted(sub.glob("*.a3m"))
                    pick = [p for p in a3ms if p.name in ("custom.a3m", "merged.a3m", "uniref90.a3m")] or a3ms[:1]
                    out += pick
    return out


def depth_cmd(cfg, d, paths: list[str], actor: str) -> str:
    from lantern.dossier import Dossier, now
    ps = [Path(p) for p in paths] or target_msas(cfg, d)
    if not ps:
        raise FileNotFoundError(f"no MSAs found for {d.target}: pass a3m paths")
    stats = [depth(p) for p in ps]
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    svg = coverage_svg(stats, ddir / "report" / "figs" / "msa_coverage.svg")
    d.evidence.setdefault("msa_depth", []).append({"time": now(), "by": actor, "value": {
        Path(s["path"]).parent.name + "/" + Path(s["path"]).name: {k: v for k, v in s.items() if k != "coverage"} for s in stats}})
    lines = [f"{'msa':40s} {'N':>8s} {'cov>=50%':>9s} {'cov median':>11s} {'med id':>7s}  thin regions"]
    for s in stats:
        lines.append(f"{Path(s['path']).parent.name + '/' + Path(s['path']).name:40s} {s['n']:8,d} {s['n_cov50']:9,d} "
                     f"{s['coverage_median']:11,d} {str(s['median_identity']):>7s}  {', '.join(s['thin_regions']) or '-'}")
    return "\n".join(lines) + f"\ncoverage plot: {svg}"


# ------------------------------------------------------------------------------------------------- crawl: fetch (login node)
def _get(url: str, timeout: int = 120) -> tuple[bytes, dict]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read(), dict(r.headers)


def _pfam_members(ipr: str) -> list[str]:
    """Pfam entries integrated into an InterPro entry (none for most homologous superfamilies, e.g. IPR008949)."""
    try:
        body, _ = _get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/?integrated={ipr}&page_size=50&format=json")
        return [x["metadata"]["accession"] for x in json.loads(body or b"{}").get("results", [])]
    except Exception:
        return []


def check_families(cfg, query: str, fam: Path, pfams: list[str]) -> dict[str, float | None]:
    """hmmscan the query against the candidate Pfam HMMs (one sequence: light) -> best E-value per family (None = no hit).
    T1331: of seven terpene-synthase candidates only PF19086 hit (E 4.7e-6)."""
    hmmscan = tool(cfg, "hmmscan") or str(Path(tool(cfg, "hmmsearch") or "hmmsearch").with_name("hmmscan"))
    press = str(Path(hmmscan).with_name("hmmpress"))
    db = fam / "candidates.hmm"
    db.write_text("".join((fam / f"{pf}.hmm").read_text() for pf in pfams))
    subprocess.run([press, "-f", str(db)], capture_output=True, check=True)
    (fam / "query.fa").write_text(f">query\n{query}\n")
    subprocess.run([hmmscan, "--tblout", str(fam / "scan.tbl"), "-E", "10", str(db), str(fam / "query.fa")],
                   capture_output=True, check=True)
    best: dict[str, float | None] = {pf: None for pf in pfams}
    for line in (fam / "scan.tbl").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        acc = f[1].split(".")[0]
        if acc in best:
            e = float(f[4])
            best[acc] = e if best[acc] is None else min(best[acc], e)
    return best


def _uniprot_members(query: str, cap: int, out: Path) -> int:
    url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}&format=fasta&size=500"
    n = 0
    with open(out, "w") as fh:
        while url and n < cap:
            body, headers = _get(url)
            text = body.decode()
            fh.write(text)
            n += text.count(">")
            m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or headers.get("link", ""))
            url = m.group(1) if m else None
    return n


def fetch_families(families: list[str], out: Path, max_members: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pfams: list[str] = []
    prov = {"families": families, "pfam": [], "members": {}}
    for f in families:
        if f.upper().startswith("PF"):
            pfams.append(f.upper())
        elif f.upper().startswith("IPR"):
            sub = _pfam_members(f.upper())
            prov.setdefault("interpro", {})[f.upper()] = sub
            pfams += sub[:6]
        else:
            raise ValueError(f"family {f}: give a Pfam (PF…) or InterPro (IPR…) accession")
    pfams = list(dict.fromkeys(pfams))
    for pf in pfams:
        hmm, _ = _get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pf}/?annotation=hmm")
        (out / f"{pf}.hmm").write_bytes(gzip.decompress(hmm))
        try:
            seed, _ = _get(f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pf}/?annotation=alignment:seed")
            (out / f"{pf}.seed.sto").write_bytes(gzip.decompress(seed))
        except Exception as exc:                                              # seeds are a bonus
            prov.setdefault("warnings", []).append(f"{pf} seed: {exc}")
        prov["pfam"].append(pf)
    for f in families:                                                        # UniProt members of each family, capped
        key = f"xref:pfam-{f.upper()}" if f.upper().startswith("PF") else f"xref:interpro-{f.upper()}"
        prov["members"][f.upper()] = _uniprot_members(key, max_members, out / f"members_{f.upper()}.fasta")
    (out / "families.json").write_text(json.dumps(prov, indent=1))
    return prov


def crawl_cmd(cfg, d, families: list[str], dbs: list[str], max_members: int, actor: str, aligner: str = "otalign") -> str:
    from lantern import ops
    from lantern.dossier import Dossier, now
    if not families:
        raise ValueError("give the family: --family PF… and/or --family IPR… (see the lantern-intake skill)")
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    out = ddir / "msas" / "crawl"
    fam = out / "families"
    seq = _query(d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "query.fasta").write_text(f">{d.target}\n{seq}\n")
    prov = fetch_families(families, fam, max_members)
    if not prov["pfam"]:
        raise ValueError(f"no Pfam HMM found for {families} (an InterPro superfamily has none: add the Pfam families too)")
    hits = check_families(cfg, seq, fam, prov["pfam"])
    prov["query_hits"] = hits
    kept = [pf for pf, e in hits.items() if e is not None and e < 1.0]
    if not kept:
        raise ValueError(f"the query hits none of {prov['pfam']} (hmmscan E < 1): {hits}; pick other families")
    if len(kept) < len(prov["pfam"]):
        d.note(actor, f"FYI: MSA crawl keeps {', '.join(kept)}; the query does not hit {', '.join(pf for pf in prov['pfam'] if pf not in kept)}")
    prov["pfam"] = kept
    prov["aligner"] = aligner
    (fam / "families.json").write_text(json.dumps(prov, indent=1))
    hs = tool(cfg, "hmmsearch")
    jobs = []
    (out / "hits").mkdir(exist_ok=True)
    for db in dbs or ["uniref90", "mgnify"]:
        path = (cfg.extra.get("msa", {}) or {}).get(db, DBS.get(db))
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"database {db}: {path} not found")
        cmd = " && ".join(f"{hs} --cpu 8 --noali -o /dev/null -E 1e-3 --domE 1e-3 --incE 1e-3 --incdomE 1e-3 "
                          f"-A {out}/hits/{pf}__{db}.sto {fam}/{pf}.hmm {path}" for pf in prov["pfam"])
        jobs.append(ops._sbatch(cmd, f"lantern-crawl-{d.target}-{db}", out / f"search_{db}.%j.log", gpu=False, hours=8))
    py = sys.executable
    assemble = f"cd {Path(__file__).resolve().parents[1]} && {py} -m lantern.msa assemble {out}"
    job = ops._sbatch(assemble, f"lantern-crawl-{d.target}-assemble", out / "assemble.%j.log", gpu=False, hours=4,
                      after=jobs)
    for j, what in zip(jobs + [job], [f"msa crawl: hmmsearch {db}" for db in (dbs or ["uniref90", "mgnify"])] + ["msa crawl: assemble"]):
        d.runs.append({"time": now(), "by": actor, "job": j, "scheduler": "slurm", "what": what, "state": "submitted", "out": str(out)})
    d.evidence.setdefault("msa_crawl", []).append({"time": now(), "by": actor, "value": {
        "families": families, "pfam": prov["pfam"], "query_hits": prov["query_hits"], "uniprot_members": prov["members"],
        "dbs": dbs or ["uniref90", "mgnify"],
        "out": str(out / "custom.a3m"), "jobs": jobs + [job]}})
    d.note(actor, f"MSA crawl submitted for {', '.join(families)} (Pfam {', '.join(prov['pfam'])}): jobs {', '.join(jobs + [job])}")
    return (f"Pfam HMMs kept {', '.join(prov['pfam'])} (query hits: {prov['query_hits']}); UniProt members {prov['members']}\n"
            f"hmmsearch jobs {', '.join(jobs)}; assemble job {job} -> {out / 'custom.a3m'}\n"
            f"when it is done: `lantern msa {d.target} --depth` to compare it with the default MSA")


def _query(d) -> str:
    fasta = d.sequence.get("fasta")
    text = Path(fasta).read_text() if fasta and Path(fasta).exists() else d.sequence.get("text", "")
    seqs = re.findall(r">[^\n]*\n([^>]+)", text) if ">" in text else [text]
    seq = re.sub(r"[^A-Za-z]", "", seqs[0]).upper() if seqs else ""
    if not seq:
        raise ValueError(f"{d.target}: no sequence in the dossier")
    return seq


# ------------------------------------------------------------------------------------------------- crawl: assemble (sbatch)
def _read_sto(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "//")):
                continue
            name, _, s = line.rstrip("\n").partition(" ")
            seqs[name] = seqs.get(name, "") + s.strip()
    return {k: re.sub(r"[^A-Za-z]", "", v).upper() for k, v in seqs.items()}


def _read_fasta(path: Path) -> dict[str, str]:
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        out[name] = "".join(buf)
    return out


PLMMSA_API = "https://plmmsa.deepfold.org"


OTALIGN_MAX_LEN = 1022                                       # ankh_large's limit on the server (E_SEQ_TOO_LONG)


def otalign_pairwise(query: str, targets: list[tuple[str, str]], chunk: int = 200, api: str = PLMMSA_API,
                     pause: float = 1.0, log=None) -> list[tuple[str, str, float]]:
    """Align each target to the query with OTalign (PLM embeddings, ankh_large) through plmMSA's public
    /v2/align/pairwise endpoint, in chunks. Returns (id, a3m row over the query, score) per kept alignment.
    HMM alignment cannot do this for remote families: on T1331 hmmalign put 218 of 281 query residues in insert
    states and jackhmmer over the family pool covered half the query for 9 sequences; OTalign covered it for 18,361.
    Targets longer than the server's limit are skipped; a target the server rejects is dropped and the chunk retried."""
    import time
    import urllib.error
    long = [t for t, q in targets if len(q) > OTALIGN_MAX_LEN]
    todo = [(t[:250], q) for t, q in targets if len(q) <= OTALIGN_MAX_LEN]
    if long and log:
        log(f"otalign: skipping {len(long)} targets longer than {OTALIGN_MAX_LEN}")
    out: list[tuple[str, str, float]] = []
    i = 0
    while i < len(todo):
        part = todo[i:i + chunk]
        body = json.dumps({"query_id": "query", "query_sequence": query, "aligner": "otalign", "mode": "glocal",
                           "targets": [{"id": t, "sequence": q} for t, q in part], "emit_a3m": True}).encode()
        req = urllib.request.Request(f"{api}/v2/align/pairwise", data=body, method="POST",
                                     headers={**UA, "Content-Type": "application/json"})
        resp, attempt = None, 0
        while resp is None:
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.loads(r.read())
            except urllib.error.HTTPError as exc:
                detail = {}
                try:
                    detail = json.loads(exc.read() or b"{}")
                except ValueError:
                    pass
                bad = (detail.get("detail") or {}).get("target_id") if isinstance(detail.get("detail"), dict) else None
                if exc.code == 400 and bad:                   # one bad target: drop it and send the chunk again
                    if log:
                        log(f"otalign: dropping {bad}: {detail.get('code')}")
                    todo = [x for x in todo if x[0] != bad]
                    part = [x for x in part if x[0] != bad]
                    body = json.dumps({"query_id": "query", "query_sequence": query, "aligner": "otalign", "mode": "glocal",
                                       "targets": [{"id": t, "sequence": q} for t, q in part], "emit_a3m": True}).encode()
                    req = urllib.request.Request(f"{api}/v2/align/pairwise", data=body, method="POST",
                                                 headers={**UA, "Content-Type": "application/json"})
                    continue
                if exc.code in (429, 500, 502, 503, 504) and attempt < 5:
                    attempt += 1
                    time.sleep(15 * attempt)
                    continue
                raise ValueError(f"plmMSA /v2/align/pairwise HTTP {exc.code}: {detail or exc}") from exc
            except (urllib.error.URLError, TimeoutError):
                if attempt >= 5:
                    raise
                attempt += 1
                time.sleep(15 * attempt)
        for al in resp.get("alignments", []):
            tid, row = al.get("target_id"), al.get("a3m_row")
            if tid and row and len(_match(row)) == len(query):        # a3m row: query columns + lowercase insertions
                out.append((tid, row, float(al.get("score") or 0)))
        i += len(part)
        if log:
            log(f"otalign {i}/{len(todo)}: {len(out)} rows")
        time.sleep(pause)
    return out


def sto_to_a3m(sto: Path, query_name: str, out: Path) -> int:
    """A jackhmmer Stockholm alignment that contains the query -> a3m over the query's columns (as AF3 converts its
    own jackhmmer output): columns where the query has a residue are kept, other residues become lowercase insertions."""
    seqs: dict[str, str] = {}
    for line in open(sto):
        if not line.strip() or line.startswith(("#", "//")):
            continue
        n, _, a = line.rstrip("\n").partition(" ")
        seqs[n] = seqs.get(n, "") + a.strip()
    q = seqs.pop(query_name)
    keep = [i for i, c in enumerate(q) if c not in "-."]
    n = 0
    with open(out, "w") as fh:
        fh.write(f">query\n{''.join(q[i].upper() for i in keep)}\n")
        for name, row in seqs.items():
            parts, k = [], set(keep)
            for i, c in enumerate(row):
                if i in k:
                    parts.append(c.upper() if c not in "-." else "-")
                elif c not in "-.":
                    parts.append(c.lower())
            a3 = "".join(parts)
            if sum(1 for c in a3 if c.isupper()) >= 10:
                fh.write(f">{name}\n{a3}\n")
                n += 1
    return n


def a2m_to_a3m(query_row: str, row: str) -> str | None:
    """Re-index a dotless A2M row (match columns uppercase/'-', insertions lowercase) on the query's residues."""
    qpos, qi = [], 0
    for c in query_row:
        if c.isupper():
            qpos.append(qi); qi += 1
        elif c == "-":
            qpos.append(None)
        elif c.islower():
            qi += 1
    L = qi
    cols, ins, k, last = ["-"] * L, {}, 0, -1
    for c in row:
        if c.isupper() or c == "-":
            q = qpos[k] if k < len(qpos) else None
            if q is not None:
                cols[q] = c
                last = q
            elif c != "-":                                      # the query has no residue here: an insertion
                ins[last] = ins.get(last, "") + c.lower()
            k += 1
        elif c.islower():
            ins[last] = ins.get(last, "") + c
    if sum(1 for c in cols if c != "-") < 10:
        return None
    return "".join(cols[i] + ins.get(i, "") for i in range(L))


def assemble(out: Path, cfg=None) -> dict:
    out = Path(out)
    fam = out / "families"
    prov = json.loads((fam / "families.json").read_text())
    query = "".join(l.strip() for l in (out / "query.fasta").read_text().splitlines()[1:])
    pool, origin, families_of = {}, {}, {}
    def add(name, seq, src, pfams):
        if len(seq) < 20:
            return
        key = f"{src}__{name}"
        pool.setdefault(key, seq)
        origin[key] = src
        families_of.setdefault(key, set()).update(pfams)
    for sto in sorted((out / "hits").glob("*.sto")):
        pf, db = sto.stem.split("__", 1)
        for name, seq in _read_sto(sto).items():
            add(name, seq, db, [pf])
    for pf in prov["pfam"]:
        if (fam / f"{pf}.seed.sto").exists():
            for name, seq in _read_sto(fam / f"{pf}.seed.sto").items():
                add(name.replace("/", "_"), seq, "pfamseed", [pf])
    for f, _ in prov["members"].items():
        if f.startswith("PF") and f not in prov["pfam"]:
            continue                                         # a family the query does not hit: its members stay out
        src_pf = [f] if f.startswith("PF") else (prov.get("interpro", {}).get(f) or prov["pfam"])
        pfs = [p for p in src_pf if p in prov["pfam"]] or prov["pfam"]
        for name, seq in _read_fasta(fam / f"members_{f}.fasta").items():
            add(name.replace("|", "_"), seq, "uniprot", pfs)
    with open(out / "pool.fasta", "w") as fh:
        for k, s in pool.items():
            fh.write(f">{k}\n{s}\n")
    mm = tool(cfg, "mmseqs")
    (out / "tmp").mkdir(exist_ok=True)
    subprocess.run([mm, "easy-cluster", str(out / "pool.fasta"), str(out / "clu"), str(out / "tmp"), "--min-seq-id", "0.9",
                    "-c", "0.8", "--cov-mode", "1", "--threads", "8", "-v", "1"], check=True)
    reps = _read_fasta(out / "clu_rep_seq.fasta")
    members: dict = {}
    for line in open(out / "clu_cluster.tsv"):
        rep, mem = line.split()
        members.setdefault(rep, set()).update(families_of.get(mem, set()))
    rows = []
    if prov.get("aligner", "otalign") == "otalign":
        # every cluster representative aligned straight to the query with OTalign (plmMSA): full-length coverage
        # for families too remote for an HMM to align (T1331: HMM 63 of 281 query residues, OTalign median 184)
        log = lambda m: print(m, flush=True)
        for name, row, score in otalign_pairwise(query, list(reps.items()), log=log):
            if sum(1 for c in _match(row) if c != "-") >= 10:
                rows.append((f"clu90__{name} score:{score:.3f}", row))
    for pf in ([] if rows or prov.get("aligner", "otalign") == "otalign" else prov["pfam"]):
        mine = [r for r in reps if pf in members.get(r, families_of.get(r, set()))]
        if not mine:
            continue
        inp = out / f"align_{pf}.fasta"
        with open(inp, "w") as fh:
            fh.write(f">query\n{query}\n" + "".join(f">clu90__{r}\n{reps[r]}\n" for r in mine))
        r = subprocess.run([tool(cfg, "hmmalign"), "--outformat", "A2M", str(fam / f"{pf}.hmm"), str(inp)],
                           capture_output=True, text=True, check=True)
        aln = {}
        name = None
        for line in r.stdout.splitlines():
            if line.startswith(">"):
                name = line[1:].split()[0]; aln[name] = ""
            elif name:
                aln[name] += line.strip()
        q = aln.pop("query")
        for n, s in aln.items():
            a3 = a2m_to_a3m(q, s)
            if a3:
                rows.append((f"{n} {pf}", a3))
    seen, uniq = set(), []
    for n, s in rows:
        if s not in seen:
            seen.add(s); uniq.append((n, s))
    raw = out / "custom.raw.a3m"
    with open(raw, "w") as fh:
        fh.write(f">query\n{query}\n" + "".join(f">{n}\n{s}\n" for n, s in uniq))
    hf = tool(cfg, "hhfilter")
    if hf:
        subprocess.run([hf, "-i", str(raw), "-o", str(out / "custom.a3m"), "-id", "99", "-cov", "20"], check=True)
    else:
        shutil.copy(raw, out / "custom.a3m")
    by_src: dict = {}
    for n, _ in read_a3m(out / "custom.a3m")[1]:
        src = n.split("__")[1] if n.startswith("clu90__") else "?"
        by_src[src] = by_src.get(src, 0) + 1
    st = depth(out / "custom.a3m")
    summary = {"pool": len(pool), "clusters_90": len(reps), "rows_aligned": len(uniq), "final": st["n"],
               "by_source": by_src, "n_cov50": st["n_cov50"], "thin_regions": st["thin_regions"]}
    (out / "provenance.json").write_text(json.dumps({**prov, "assembly": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------- plmMSA
def plm_cmd(cfg, d, aligner: str, actor: str) -> str:
    from lantern.dossier import Dossier, now
    script = (cfg.extra.get("tools", {}) if cfg else {}).get("plmmsa") or PLMMSA
    if not Path(script).exists():
        raise FileNotFoundError(f"plmMSA client not found: {script} (set [tools] plmmsa in lantern.toml)")
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    work = ddir / "msas" / "plm_target"
    work.mkdir(parents=True, exist_ok=True)
    (work / "target.fasta").write_text(f">{d.target}\n{_query(d)}\n")
    r = subprocess.run(["bash", script, f"--aligner={aligner}", str(work)], capture_output=True, text=True, timeout=1800)
    a3m = work / "msas" / "plmmsa" / aligner / "merged.a3m"
    if r.returncode != 0 or not a3m.exists():
        raise ValueError(f"plmMSA failed ({r.returncode}): {r.stderr[-800:]}")
    dest = ddir / "msas" / f"plm_{aligner}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(a3m, dest / "merged.a3m")
    st = depth(dest / "merged.a3m")
    d.evidence.setdefault("msa_plm", []).append({"time": now(), "by": actor, "value": {
        "aligner": aligner, "out": str(dest / "merged.a3m"), **{k: v for k, v in st.items() if k != "coverage"}}})
    return f"plmMSA {aligner}: N {st['n']:,}, >=50% coverage {st['n_cov50']:,}, thin regions {st['thin_regions'] or '-'} -> {dest / 'merged.a3m'}"


if __name__ == "__main__":                                   # python -m lantern.msa assemble DIR   (inside the sbatch job)
    if len(sys.argv) == 3 and sys.argv[1] == "assemble":
        assemble(Path(sys.argv[2]))
    else:
        print(__doc__)
