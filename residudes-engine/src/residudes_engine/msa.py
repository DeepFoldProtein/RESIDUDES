"""MSA search for residudes-engine inputs.

Three sources, same output: one A3M per chain (query first), plus an optional
paired A3M and template hits.

  plmmsa     DeepFold-PLM MSA API (https://plmmsa.deepfold.org): the query is
             embedded with a protein language model, neighbours are retrieved
             from a UniRef50 PLM index (FAISS) and aligned (otalign / plmalign /
             plm_blast). Finds remote homologues sequence search misses; no
             pairing, no templates.
  colabfold  MMseqs2 through the ColabFold MSA server API (public
             https://api.colabfold.com or a self-hosted server): UniRef30 +
             environmental DB, optional species pairing for complexes, PDB70
             template hits (m8).
  mmseqs2    MMseqs2 run locally (scripts/mmseqs2_search.sh — the ColabFold
             recipe used in CASP17, GPU-capable) against UniRef30 / envdb /
             PDB100 databases on this machine.

HTTP goes through curl (the CLI Python may lack ssl). Everything here is
stdlib; the structure-heavy template work lives in core/scripts/template_tools.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PLMMSA_API = "https://plmmsa.deepfold.org"
COLABFOLD_API = "https://api.colabfold.com"
USER_AGENT = "residudes-engine-msa/0.1"
SOURCES = ("plmmsa", "colabfold", "mmseqs2")
PLMMSA_ALIGNERS = ("otalign", "plmalign", "plm_blast")
# Each aligner is calibrated against a different PLM upstream (CASP17 msas/plmmsa/README).
PLMMSA_SCORE_MODEL = {"plm_blast": "prott5", "plmalign": "prott5", "otalign": "ankh_large"}


class MsaError(RuntimeError):
    pass


# ── sequences and stoichiometry ───────────────────────────────────────────

@dataclass
class Chain:
    letter: str          # source letter A, B, C … (FASTA order)
    name: str            # FASTA header (first token)
    seq: str
    copies: int = 1


STO_TOKEN = re.compile(r"([A-Z])(\d+)")


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append((line[1:].split()[0] if len(line) > 1 else f"seq{len(records) + 1}", []))
        elif records:
            records[-1][1].append(line)
        else:
            records.append((f"seq{len(records) + 1}", [line]))
    out = []
    for name, parts in records:
        seq = "".join(parts).replace(" ", "").upper()
        if not seq:
            raise MsaError(f"empty sequence for record {name!r}")
        if not re.fullmatch(r"[A-Z]+", seq):
            raise MsaError(f"record {name!r}: sequence has non-letter characters")
        out.append((name, seq))
    if not out:
        raise MsaError("no sequences in FASTA")
    return out


def parse_sto(spec: str | None, n_chains: int) -> list[int]:
    """`A2B1` → copies per source chain; default one copy each."""
    if not spec:
        return [1] * n_chains
    if not re.fullmatch(r"(?:[A-Z]\d+)+", spec):
        raise MsaError(f"stoichiometry must look like A2B1 (got {spec!r})")
    counts = {}
    for m in STO_TOKEN.finditer(spec):
        if m.group(1) in counts:
            raise MsaError(f"repeated chain {m.group(1)} in {spec!r}")
        counts[m.group(1)] = int(m.group(2))
    letters = [chr(ord("A") + i) for i in range(n_chains)]
    unknown = sorted(set(counts) - set(letters))
    if unknown:
        raise MsaError(f"stoichiometry names chain(s) {unknown} but the FASTA has {n_chains}")
    return [max(1, counts.get(L, 1)) if L in counts else 1 for L in letters]


def chains_from_fasta(path: Path, sto: str | None = None) -> list[Chain]:
    recs = parse_fasta(path.read_text())
    copies = parse_sto(sto, len(recs))
    return [Chain(chr(ord("A") + i), name, seq, copies[i]) for i, (name, seq) in enumerate(recs)]


# ── HTTP via curl ─────────────────────────────────────────────────────────

def _curl(args: list[str], *, timeout: float = 120) -> tuple[int, bytes]:
    if not shutil.which("curl"):
        raise MsaError("curl is required for MSA API calls (CLI Python has no ssl)")
    cmd = ["curl", "-sS", "-L", "--max-time", str(int(timeout)), "-H", f"User-Agent: {USER_AGENT}",
           "-w", "\n%{http_code}", *args]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise MsaError(f"curl failed ({res.returncode}): {res.stderr.decode(errors='replace').strip()[:300]}")
    body, _, code = res.stdout.rpartition(b"\n")
    try:
        return int(code), body
    except ValueError:
        raise MsaError(f"unexpected curl output: {res.stdout[-200:]!r}")


def http_json(url: str, *, method: str = "GET", body: dict | None = None, form: dict | None = None,
              token: str | None = None, timeout: float = 120) -> tuple[int, dict | str]:
    args = ["-X", method]
    if token:
        args += ["-H", f"Authorization: Bearer {token}"]
    if body is not None:
        args += ["-H", "Content-Type: application/json", "--data-binary", json.dumps(body)]
    if form is not None:
        for k, v in form.items():
            args += ["--data-urlencode", f"{k}={v}"]
    args.append(url)
    code, raw = _curl(args, timeout=timeout)
    try:
        return code, json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return code, raw.decode("utf-8", errors="replace")


def http_download(url: str, dest: Path, *, token: str | None = None, timeout: float = 600) -> int:
    args = ["-o", str(dest)]
    if token:
        args += ["-H", f"Authorization: Bearer {token}"]
    args.append(url)
    code, _ = _curl(args, timeout=timeout)
    return code


# ── A3M helpers ───────────────────────────────────────────────────────────

def parse_a3m(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith(">"):
            out.append((line[1:], []))
        elif line.strip() and out:
            out[-1][1].append(line.strip())
    return [(h, "".join(s)) for h, s in out]


def normalise_a3m(text: str, query: str, query_name: str = "query") -> str:
    """Query first (exact sequence, header `>query_name`), then every other row once."""
    rows = parse_a3m(text)
    out = [f">{query_name}", query]
    seen = {query}
    for h, s in rows:
        if s in seen:
            continue
        seen.add(s)
        out += [f">{h}", s]
    return "\n".join(out) + "\n"


def merge_a3ms(texts: list[str], query: str, query_name: str = "query") -> str:
    """Concatenate several A3Ms for the same query, de-duplicated, query first."""
    rows = [f">{query_name}", query]
    seen = {query}
    for t in texts:
        for h, s in parse_a3m(t):
            if s in seen:
                continue
            seen.add(s)
            rows += [f">{h}", s]
    return "\n".join(rows) + "\n"


def a3m_depth(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(">"))


# ── result container ──────────────────────────────────────────────────────

@dataclass
class MsaResult:
    source: str
    unpaired: dict[str, str] = field(default_factory=dict)   # chain letter → a3m text
    paired: dict[str, str] = field(default_factory=dict)     # chain letter → paired a3m (row-aligned across chains)
    templates_m8: dict[str, str] = field(default_factory=dict)  # chain letter → m8 hits
    meta: dict = field(default_factory=dict)


# ── plmMSA (DeepFold-PLM MSA API) ─────────────────────────────────────────

def plmmsa_search(chains: list[Chain], *, api: str = PLMMSA_API, token: str | None = None,
                  aligner: str = "otalign", mode: str = "local", score_model: str | None = None,
                  poll: float = 10.0, timeout: float = 3600.0, log=print) -> MsaResult:
    if aligner not in PLMMSA_ALIGNERS:
        raise MsaError(f"aligner must be one of {PLMMSA_ALIGNERS}")
    score_model = score_model or PLMMSA_SCORE_MODEL[aligner]
    api = api.rstrip("/")
    res = MsaResult("plmmsa", meta={"api": api, "aligner": aligner, "mode": mode, "score_model": score_model})
    done: dict[str, str] = {}
    for ch in chains:
        if ch.seq in done:                       # identical chains share one search
            res.unpaired[ch.letter] = done[ch.seq]
            continue
        body = {"sequences": [ch.seq], "query_ids": [ch.name], "aligner": aligner, "mode": mode,
                "score_model": score_model, "output_format": "a3m", "paired": False}
        code, resp = http_json(f"{api}/v2/msa", method="POST", body=body, token=token)
        if code != 202 or not isinstance(resp, dict) or "job_id" not in resp:
            raise MsaError(f"plmmsa submit failed [{code}]: {str(resp)[:300]}")
        job = resp["job_id"]
        log(f"  plmmsa {ch.letter} ({len(ch.seq)} aa): job {job}")
        t0 = time.time()
        while True:
            code, st = http_json(f"{api}/v2/msa/{job}", token=token)
            if code != 200 or not isinstance(st, dict):
                raise MsaError(f"plmmsa poll failed [{code}]: {str(st)[:300]}")
            status = (st.get("status") or "").lower()
            if status in ("succeeded", "failed", "cancelled", "canceled", "error"):
                break
            if time.time() - t0 > timeout:
                raise MsaError(f"plmmsa job {job} not finished after {timeout:.0f} s (status {status})")
            time.sleep(poll)
        if status != "succeeded":
            raise MsaError(f"plmmsa job {job} for chain {ch.letter} ended in {status!r}: {st.get('error')}")
        result = st.get("result") or {}
        payload = result.get("payload")
        if result.get("format") != "a3m" or not isinstance(payload, str):
            raise MsaError(f"plmmsa job {job}: unexpected result {str(result)[:200]}")
        a3m = normalise_a3m(payload, ch.seq, ch.name)
        done[ch.seq] = a3m
        res.unpaired[ch.letter] = a3m
        log(f"  plmmsa {ch.letter}: {a3m_depth(a3m) - 1} homologues in {time.time() - t0:.0f} s")
    return res


# ── ColabFold MSA server (MMseqs2) ────────────────────────────────────────

_COLABFOLD_FILES = {"uniref": "uniref.a3m", "env": "bfd.mgnify30.metaeuk30.smag30.a3m",
                    "pair": "pair.a3m", "templates": "pdb70.m8"}


def _colabfold_ticket(api: str, endpoint: str, query: str, mode: str, *, poll: float, timeout: float,
                      log=print) -> str:
    """Submit and wait; returns the ticket id (server statuses: PENDING, RUNNING, COMPLETE, ERROR, RATELIMIT, MAINTENANCE)."""
    t0 = time.time()
    while True:
        code, resp = http_json(f"{api}/{endpoint}", method="POST", form={"q": query, "mode": mode})
        if isinstance(resp, dict) and resp.get("status") in ("RATELIMIT", "MAINTENANCE"):
            log(f"  colabfold: {resp['status']}, retrying in 30 s")
            time.sleep(30)
            if time.time() - t0 > timeout:
                raise MsaError("colabfold: rate-limited for too long")
            continue
        if code not in (200, 201) or not isinstance(resp, dict) or "id" not in resp:
            raise MsaError(f"colabfold submit failed [{code}]: {str(resp)[:300]}")
        break
    tid = resp["id"]
    while True:
        code, st = http_json(f"{api}/ticket/{tid}")
        status = st.get("status") if isinstance(st, dict) else None
        if status == "COMPLETE":
            return tid
        if status in ("ERROR", "UNKNOWN"):
            raise MsaError(f"colabfold ticket {tid}: {status}")
        if time.time() - t0 > timeout:
            raise MsaError(f"colabfold ticket {tid} not complete after {timeout:.0f} s ({status})")
        time.sleep(poll)


def _split_server_a3m(text: str, n_queries: int) -> list[str]:
    """The server concatenates per-query A3Ms separated by NUL; headers are >101, >102 …"""
    blocks = [b for b in text.split("\x00") if b.strip()]
    out = [""] * n_queries
    for b in blocks:
        m = re.match(r">(\d+)", b.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 101
        if 0 <= idx < n_queries:
            out[idx] = b
    return out


def colabfold_search(chains: list[Chain], *, api: str = COLABFOLD_API, use_env: bool = True,
                     use_filter: bool = True, use_templates: bool = True, pair: bool = True,
                     pairing: str = "greedy", poll: float = 10.0, timeout: float = 3600.0,
                     workdir: Path | None = None, log=print) -> MsaResult:
    api = api.rstrip("/")
    uniq: list[str] = []
    for ch in chains:
        if ch.seq not in uniq:
            uniq.append(ch.seq)
    query = "".join(f">{101 + i}\n{s}\n" for i, s in enumerate(uniq))
    mode = ("env" if use_env else "all") + ("" if use_filter else "-nofilter")
    res = MsaResult("colabfold", meta={"api": api, "mode": mode, "pairing": pairing if pair else None})
    tmp = Path(workdir or tempfile.mkdtemp(prefix="residudes-engine-colabfold-"))
    tmp.mkdir(parents=True, exist_ok=True)

    log(f"  colabfold: {len(uniq)} unique sequence(s), mode {mode}")
    tid = _colabfold_ticket(api, "ticket/msa", query, mode, poll=poll, timeout=timeout, log=log)
    tgz = tmp / f"msa_{tid}.tar.gz"
    code = http_download(f"{api}/result/download/{tid}", tgz)
    if code != 200:
        raise MsaError(f"colabfold download failed [{code}]")
    files = _untar(tgz, tmp / f"msa_{tid}")
    uniref = _split_server_a3m(files.get(_COLABFOLD_FILES["uniref"], ""), len(uniq))
    env = _split_server_a3m(files.get(_COLABFOLD_FILES["env"], ""), len(uniq)) if use_env else [""] * len(uniq)
    m8 = files.get(_COLABFOLD_FILES["templates"], "") if use_templates else ""
    for ch in chains:
        i = uniq.index(ch.seq)
        res.unpaired[ch.letter] = merge_a3ms([uniref[i], env[i]], ch.seq, ch.name)
        if m8:
            hits = [ln for ln in m8.splitlines() if ln.split("\t")[0] == str(101 + i)]
            if hits:
                res.templates_m8[ch.letter] = "\n".join(hits) + "\n"
        log(f"  colabfold {ch.letter}: {a3m_depth(res.unpaired[ch.letter]) - 1} homologues"
            + (f", {len(res.templates_m8.get(ch.letter, '').splitlines())} template hits" if m8 else ""))

    if pair and len(uniq) > 1:
        pmode = {"greedy": "pairgreedy", "complete": "paircomplete"}.get(pairing, "pair")
        log(f"  colabfold: pairing ({pmode})")
        tid = _colabfold_ticket(api, "ticket/pair", query, pmode, poll=poll, timeout=timeout, log=log)
        tgz = tmp / f"pair_{tid}.tar.gz"
        if http_download(f"{api}/result/download/{tid}", tgz) != 200:
            raise MsaError("colabfold pair download failed")
        files = _untar(tgz, tmp / f"pair_{tid}")
        paired = _split_server_a3m(files.get(_COLABFOLD_FILES["pair"], ""), len(uniq))
        for ch in chains:
            block = paired[uniq.index(ch.seq)]
            if block.strip():
                res.paired[ch.letter] = _relabel_first(block, ch.name)
        if res.paired:
            log(f"  colabfold: paired rows {a3m_depth(next(iter(res.paired.values())))}")
    return res


def _relabel_first(a3m: str, name: str) -> str:
    rows = parse_a3m(a3m)
    if not rows:
        return a3m
    rows[0] = (name, rows[0][1])
    return "\n".join(f">{h}\n{s}" for h, s in rows) + "\n"


def _untar(tgz: Path, dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            out[Path(m.name).name] = f.read().decode("utf-8", errors="replace")
    return out


# ── local MMseqs2 ─────────────────────────────────────────────────────────

def mmseqs2_search(chains: list[Chain], *, out_dir: Path, db_dir: Path, mmseqs: str = "mmseqs",
                   gpu: bool = False, use_env: bool = True, use_templates: bool = True, threads: int = 8,
                   script: Path | None = None, log=print) -> MsaResult:
    """Run scripts/mmseqs2_search.sh (the CASP17 ColabFold recipe) on this machine."""
    script = script or Path(__file__).resolve().parents[2] / "scripts" / "mmseqs2_search.sh"
    if not script.is_file():
        raise MsaError(f"missing {script}")
    work = out_dir / "mmseqs2"
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "query.fasta"
    uniq: list[Chain] = []
    for ch in chains:
        if all(ch.seq != u.seq for u in uniq):
            uniq.append(ch)
    fasta.write_text("".join(f">{c.letter}\n{c.seq}\n" for c in uniq))
    env = dict(os.environ, MMSEQS=mmseqs, DB_DIR=str(db_dir), USE_GPU="1" if gpu else "0",
               USE_ENV="1" if use_env else "0", USE_TEMPLATES="1" if use_templates else "0", THREADS=str(threads))
    log(f"  mmseqs2: {script.name} on {len(uniq)} sequence(s), db {db_dir}, gpu={'on' if gpu else 'off'}")
    with (work / "mmseqs2.log").open("w") as fh:
        rc = subprocess.run(["bash", str(script), str(fasta), str(work)], env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise MsaError(f"mmseqs2_search.sh failed (exit {rc}); see {work / 'mmseqs2.log'}")
    res = MsaResult("mmseqs2", meta={"db_dir": str(db_dir), "gpu": gpu, "env": use_env})
    for ch in chains:
        src = next(u for u in uniq if u.seq == ch.seq)
        a3m = work / "a3m" / f"{src.letter}.a3m"
        if not a3m.is_file():
            raise MsaError(f"mmseqs2: no output for chain {ch.letter} ({a3m})")
        res.unpaired[ch.letter] = normalise_a3m(a3m.read_text(), ch.seq, ch.name)
        m8 = work / "m8" / f"{src.letter}.m8"
        if m8.is_file() and m8.stat().st_size:
            res.templates_m8[ch.letter] = m8.read_text()
    return res


# ── persisting results ────────────────────────────────────────────────────

def write_msa_dir(res: MsaResult, chains: list[Chain], out_dir: Path) -> Path:
    """Lay out <out_dir>/msa/{unpaired_<L>.a3m, paired_<L>.a3m, templates_<L>.m8, msa.json}."""
    d = out_dir / "msa"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"source": res.source, "meta": res.meta, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "chains": {}}
    for ch in chains:
        entry = {"name": ch.name, "length": len(ch.seq), "copies": ch.copies}
        if ch.letter in res.unpaired:
            p = d / f"unpaired_{ch.letter}.a3m"
            p.write_text(res.unpaired[ch.letter])
            entry["unpaired"] = p.name
            entry["depth"] = a3m_depth(res.unpaired[ch.letter]) - 1
        if ch.letter in res.paired:
            p = d / f"paired_{ch.letter}.a3m"
            p.write_text(res.paired[ch.letter])
            entry["paired"] = p.name
        if ch.letter in res.templates_m8:
            p = d / f"templates_{ch.letter}.m8"
            p.write_text(res.templates_m8[ch.letter])
            entry["templates_m8"] = p.name
        manifest["chains"][ch.letter] = entry
    (d / "msa.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return d


def read_msa_dir(d: Path) -> tuple[dict, dict[str, Path], dict[str, Path], dict[str, Path]]:
    man = json.loads((d / "msa.json").read_text())
    unp = {L: d / e["unpaired"] for L, e in man["chains"].items() if e.get("unpaired")}
    par = {L: d / e["paired"] for L, e in man["chains"].items() if e.get("paired")}
    m8 = {L: d / e["templates_m8"] for L, e in man["chains"].items() if e.get("templates_m8")}
    return man, unp, par, m8
