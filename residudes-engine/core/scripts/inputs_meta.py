#!/usr/bin/env python3
"""MSA + template provenance extraction per model.

Each helper takes the model's *native* input file (or a parsed dict) and
returns a structured `inputs_meta` dict for inclusion in summary.json:

    {
      "msa": {
        "<chain>": {
          "unpaired": "<path>",
          "paired":   "<path>",
          "depth_unpaired": <int>,
          "depth_paired":   <int>
        }
      },
      "templates": {
        "<chain>": [
          {"pdb_chain": "7jwb_D", "qs": 146, "qe": 276, "length": <int|None>},
          ...
        ]
      }
    }

Designed so each model's writer can attach an entry to its summary.json
without duplicating the parsing/counting logic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


_HMMSEARCH_HEADER = re.compile(r">([0-9A-Za-z]+)_(\w+)/(\d+)-(\d+)")
_HMMSEARCH_LENGTH = re.compile(r"protein length:(\d+)")
_HMMSEARCH_TEMPLATE_RANGE = re.compile(r"template_range:(\d+)-(\d+)")


def _count_entity_poly_seq(cif_text: str) -> Optional[int]:
    """Count residues in a single-chain mmcif by tallying rows of the
    `_entity_poly_seq` loop. Returns None on parse failure."""
    in_loop = False
    in_eps = False
    cols = 0
    count = 0
    for ln in cif_text.splitlines():
        s = ln.strip()
        if s == "loop_":
            in_loop = True
            in_eps = False
            cols = 0
            continue
        if in_loop and s.startswith("_"):
            if s.startswith("_entity_poly_seq."):
                in_eps = True
            cols += 1
            continue
        if in_loop and cols > 0:
            if s == "" or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                if in_eps:
                    return count if count > 0 else None
                in_loop = False
                in_eps = False
                count = 0
                continue
            if in_eps:
                count += 1
    return count if (in_eps and count > 0) else None


def count_a3m_records(p: Optional[str]) -> int:
    """Count `>` headers in an a3m file. 0 on missing/error."""
    if not p:
        return 0
    try:
        with open(p) as f:
            return sum(1 for ln in f if ln.startswith(">"))
    except OSError:
        return 0


def parse_hmmsearch_a3m_hits(p: Optional[str]) -> list[dict]:
    """Parse hit metadata from a hmmsearch a3m file's header lines.

    Header format: `>{pdb}_{chain}/{qs}-{qe} ... protein length:N`
    First record is the query — skipped.
    """
    if not p:
        return []
    try:
        with open(p) as f:
            lines = f.readlines()
    except OSError:
        return []
    hits: list[dict] = []
    seen_query = False
    for ln in lines:
        if not ln.startswith(">"):
            continue
        if not seen_query:
            seen_query = True
            continue
        line = ln.strip()
        m = _HMMSEARCH_HEADER.match(line)
        if m:
            pdb, chain, qs, qe = m.groups()
            ml = _HMMSEARCH_LENGTH.search(line)
            mt = _HMMSEARCH_TEMPLATE_RANGE.search(line)
            entry = {
                "pdb_chain": f"{pdb.lower()}_{chain}",
                "qs": int(qs),
                "qe": int(qe),
                "length": int(ml.group(1)) if ml else None,
            }
            if mt:
                entry["template_st"] = int(mt.group(1))
                entry["template_en"] = int(mt.group(2))
            hits.append(entry)
    return hits


def _chain_label(chain_id) -> str:
    if isinstance(chain_id, list):
        return chain_id[0] if chain_id else "?"
    return str(chain_id) if chain_id is not None else "?"


# ── Featurizer-time runtime inputs ───────────────────────────────────────

def write_runtime_inputs(out_dir, payload: dict) -> None:
    """Dump featurizer-time inputs view to `<out_dir>/inputs_runtime.json`.

    Schema (uniform across 4 models):
        {
          "msa": {
            "<chain>": {
              "unpaired": {"src": "...", "depth": <int>},
              "paired":   {"src": "...", "depth": <int>}    # null/omit if absent
            }
          },
          "templates": {
            "<chain>": [
              {"pdb_chain": "<pdb>_<chain>",
               "qs": <int>, "qe": <int>,
               "length": <int|null>,
               "selected": <bool>}
            ]
          }
        }

    Writers prefer this file over input.json provenance when emitting
    summary.json.inputs (so the field reflects what the model actually
    consumed: post-cap MSA depth + post-filter template selection).
    """
    p = Path(out_dir) / "inputs_runtime.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Preserve fields that downstream writers add (pairing) — write_runtime_inputs
    # is called per-query in some shims, so a naive overwrite would clobber the
    # post-pairing stats stashed by the model's pairing function.
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            for k, v in existing.items():
                if k not in payload and k in ("pairing", "variant"):
                    payload = {**payload, k: v}
        except Exception:
            pass
    # Extract variant from any MSA / template src path matching
    # `inputs/<sto>/<variant>/...`. Useful when comparing runs on the
    # same target across different MSA pipelines (af3_msa, tcrm_msa,
    # mmseqs2_gpu, ...) — without this the file content is identical
    # in shape and the only differentiator was the parent dir name.
    if payload.get("variant") is None:
        try:
            import re as _re
            _pat = _re.compile(r"/inputs/[^/]+/([^/]+)/")
            seen = set()
            def _scan(o):
                if isinstance(o, dict):
                    for v in o.values(): _scan(v)
                elif isinstance(o, list):
                    for v in o: _scan(v)
                elif isinstance(o, str):
                    m = _pat.search(o)
                    if m:
                        seen.add(m.group(1))
            _scan(payload)
            if len(seen) == 1:
                payload = {"variant": next(iter(seen)), **payload}
        except Exception:
            pass
    p.write_text(json.dumps(payload, indent=2))


def read_runtime_inputs(out_dir) -> Optional[dict]:
    """Read `<out_dir>/inputs_runtime.json` if it exists, else None.
    Walks up at most 3 parents so per-sample writers can find the seed-
    level file without knowing the exact depth."""
    cur = Path(out_dir)
    for _ in range(4):
        cand = cur / "inputs_runtime.json"
        if cand.is_file():
            try:
                return json.loads(cand.read_text())
            except Exception:
                return None
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def update_runtime_inputs(out_dir, key: str, value) -> bool:
    """Find existing `inputs_runtime.json` by walking up at most 3 parents
    and merge `{key: value}` into it. Idempotent — repeated calls overwrite
    the same key. Returns True if updated, False if no file was found."""
    cur = Path(out_dir)
    for _ in range(4):
        cand = cur / "inputs_runtime.json"
        if cand.is_file():
            try:
                d = json.loads(cand.read_text())
            except Exception:
                d = {}
            d[key] = value
            try:
                cand.write_text(json.dumps(d, indent=2))
                return True
            except Exception:
                return False
        if cur.parent == cur:
            break
        cur = cur.parent
    return False


# ── Per-model converters ─────────────────────────────────────────────────

def from_pt2_input_json(input_json_path: Optional[str]) -> dict[str, dict]:
    """{job_name: {msa: {chain: ...}, templates: {chain: [...]}}}"""
    if not input_json_path:
        return {}
    try:
        data = json.loads(open(input_json_path).read())
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        data = [data]
    out: dict[str, dict] = {}
    for job in data:
        name = job.get("name") or "<unnamed>"
        msa: dict = {}
        tpls: dict = {}
        for entry in (job.get("sequences") or []):
            pc = entry.get("proteinChain") or entry.get("protein")
            if not pc:
                continue
            key = _chain_label(pc.get("id"))
            unp, ppm, tpath = (pc.get("unpairedMsaPath"),
                               pc.get("pairedMsaPath"),
                               pc.get("templatesPath"))
            msa[key] = {
                "unpaired": unp,
                "paired":   ppm,
                "depth_unpaired": count_a3m_records(unp),
                "depth_paired":   count_a3m_records(ppm),
            }
            tpls[key] = parse_hmmsearch_a3m_hits(tpath)
        out[name] = {"msa": msa, "templates": tpls}
    return out


def from_of3_query_json(query_json_path: Optional[str]) -> dict[str, dict]:
    """{query_name: {msa: {chain: ...}, templates: {chain: [...]}}}"""
    if not query_json_path:
        return {}
    try:
        data = json.loads(open(query_json_path).read())
    except (OSError, json.JSONDecodeError):
        return {}
    queries = data.get("queries") or {}
    out: dict[str, dict] = {}
    for qname, q in queries.items():
        msa: dict = {}
        tpls: dict = {}
        for c in (q.get("chains") or []):
            key = _chain_label(c.get("chain_ids"))
            mains = c.get("main_msa_file_paths") or []
            paireds = c.get("paired_msa_file_paths") or []
            unp = mains[0] if mains else None
            ppm = paireds[0] if paireds else None
            tpath = c.get("template_alignment_file_path")
            msa[key] = {
                "unpaired": unp,
                "paired":   ppm,
                "depth_unpaired": count_a3m_records(unp),
                "depth_paired":   count_a3m_records(ppm),
            }
            tpls[key] = parse_hmmsearch_a3m_hits(tpath)
        out[qname] = {"msa": msa, "templates": tpls}
    return out


def from_bt2_yaml(yaml_path: Optional[str]) -> dict:
    """Single-record yaml → {msa: {chain: ...}, templates: {chain: [...]}}.

    bt2 yaml has chains under `sequences:` and templates as a flat list at
    top level with per-template `chain_id`. We group templates back per
    chain on output.
    """
    if not yaml_path:
        return {}
    try:
        text = Path(yaml_path).read_text()
    except OSError:
        return {}
    msa: dict = {}
    tpls: dict = {}
    # Per-chain msa (boltz yaml is small + regular; regex is fine).
    chain_blocks = re.split(r"^  - protein:\s*\n", text, flags=re.MULTILINE)
    for body in chain_blocks[1:]:
        body = re.split(r"^  - \w", body, maxsplit=1, flags=re.MULTILINE)[0]
        body = re.split(r"^[a-z_]+:", body, maxsplit=1, flags=re.MULTILINE)[0]
        m_id = re.search(r"id:\s*\[([^\]]+)\]", body)
        m_msa = re.search(r"msa:\s*(\S+)", body)
        if not m_id:
            continue
        ids = [s.strip() for s in m_id.group(1).split(",") if s.strip()]
        msa_path = m_msa.group(1).strip() if m_msa else None
        for cid in ids:
            msa[cid] = {
                "unpaired": msa_path,
                "paired":   None,
                "depth_unpaired": count_a3m_records(msa_path),
                "depth_paired":   0,
            }
            tpls.setdefault(cid, [])
    # Top-level templates.
    tpl_pat = re.compile(
        r"-\s*cif:\s*(\S+)\s*\n\s*chain_id:\s*\[([^\]]+)\]", re.MULTILINE
    )
    for m in tpl_pat.finditer(text):
        cif_path = m.group(1).strip()
        chain_ids = [s.strip() for s in m.group(2).split(",") if s.strip()]
        # Extract pdb id from filename: template_<C>_<NN>_<pdb>.cif → <pdb>
        stem = Path(cif_path).stem
        pdb_match = re.search(r"_([0-9a-z]{4})$", stem)
        pdb = pdb_match.group(1) if pdb_match else stem
        # Recover the source PDB chain from the cif (auth_asym_id of the
        # first atom). bt2 yaml itself only carries the local query chain,
        # but our template cifs are filtered to a single PDB chain by
        # m8_to_templates.py / hmmsearch, so reading the cif gives the
        # original chain.
        pdb_chain_letter = _read_first_auth_asym_id(cif_path) or "?"
        for cid in chain_ids:
            tpls.setdefault(cid, []).append({
                "pdb_chain": f"{pdb}_{pdb_chain_letter}",
                "cif_path": cif_path,
            })
    return {"msa": msa, "templates": tpls}


def _read_first_auth_asym_id(cif_path: str) -> Optional[str]:
    """Return the first `_atom_site.auth_asym_id` from a cif file (single-
    chain templates) — recovers the source PDB chain when callers only
    knew the query-chain id. Returns None if the file is unreadable or
    doesn't have an _atom_site loop."""
    try:
        with open(cif_path) as f:
            text = f.read()
    except OSError:
        return None
    # Quick parser: find the loop_ block whose first column is _atom_site.*,
    # locate auth_asym_id column index, return value from first data row.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols: list[str] = []
            while j < len(lines) and lines[j].lstrip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                short = [c.split(".", 1)[1] for c in cols]
                if "auth_asym_id" in short:
                    idx = short.index("auth_asym_id")
                    while j < len(lines):
                        row = lines[j].strip()
                        if not row or row.startswith("#") or row == "loop_" or row.startswith("_"):
                            break
                        toks = row.split()
                        if len(toks) > idx:
                            return toks[idx]
                        j += 1
            i = j
        else:
            i += 1
    return None


def from_af3_input_json(input_json_path: Optional[str]) -> dict:
    """AF3 input.json → {msa: {chain: ...}, templates: {chain: [...]}}.

    AF3 templates carry queryIndices/templateIndices + mmcif inline or path;
    we extract the source pdb id from the cif's `data_<ID>` header.

    Path mode and **inline mode** are both supported (the bulk pueue path
    runs inline_af3_paths.py before upload so a3m/cif are embedded as
    strings; depth is then derived by counting `>` characters).
    """
    if not input_json_path:
        return {}
    try:
        data = json.loads(open(input_json_path).read())
    except (OSError, json.JSONDecodeError):
        return {}

    def _depth_inline(s):
        if not isinstance(s, str) or not s:
            return 0
        return s.count("\n>") + (1 if s.startswith(">") else 0)

    msa: dict = {}
    tpls: dict = {}
    for entry in (data.get("sequences") or []):
        prot = entry.get("protein")
        if not prot:
            continue
        key = _chain_label(prot.get("id"))
        unp_path = prot.get("unpairedMsaPath")
        ppm_path = prot.get("pairedMsaPath")
        unp_inline = prot.get("unpairedMsa") if not unp_path else None
        ppm_inline = prot.get("pairedMsa") if not ppm_path else None
        msa[key] = {
            "unpaired": unp_path or ("<inline>" if unp_inline else None),
            "paired":   ppm_path or ("<inline>" if ppm_inline else None),
            "depth_unpaired": (count_a3m_records(unp_path) if unp_path
                               else _depth_inline(unp_inline)),
            "depth_paired":   (count_a3m_records(ppm_path) if ppm_path
                               else _depth_inline(ppm_inline)),
        }
        per_chain_tpls = []
        for t in (prot.get("templates") or []):
            pdb_id = ""
            chain = ""
            cif_text = t.get("mmcif")
            if cif_text is None and t.get("mmcifPath"):
                try:
                    cif_text = Path(t["mmcifPath"]).read_text()
                except OSError:
                    cif_text = ""
            if cif_text:
                for ln in cif_text.split("\n", 5)[:5]:
                    s = ln.strip()
                    if s.startswith("data_"):
                        pdb_id = s[5:].lower()
                        break
                # chain from _entity_poly.pdbx_strand_id
                m = re.search(r"_entity_poly\.pdbx_strand_id\s+(\S+)", cif_text)
                if m:
                    chain = m.group(1).split(",")[0]
            qi = t.get("queryIndices") or []
            ti = t.get("templateIndices") or []
            valid_q = [q for q in qi if q is not None and q >= 0]
            valid_t = [tt for tt in ti if tt is not None and tt >= 0]
            length = _count_entity_poly_seq(cif_text) if cif_text else None
            per_chain_tpls.append({
                "pdb_chain": f"{pdb_id}_{chain}" if pdb_id else "?_?",
                "qs": (min(valid_q) + 1) if valid_q else None,
                "qe": (max(valid_q) + 1) if valid_q else None,
                "length": length,
            })
        tpls[key] = per_chain_tpls
    return {"msa": msa, "templates": tpls}


# ── Lazy lookup via env var (used by per-model writers) ───────────────────

_CACHE: dict = {}


def lookup_inputs_meta(model: str, name: str) -> Optional[dict]:
    """Return the per-(query/job/record) inputs_meta entry, or None.

    Reads `BIO_TRT_INPUT_PATH` env var (set by each runner's entry shim
    before invoking the model). Caches by path so repeated calls within
    the same process don't re-parse.

    Args:
        model: "pt2" | "of3" | "bt2" | "af3"
        name:  job/query/record name (matches the entry's key in the
               input file). For af3 it's ignored — af3 input has 1 job.
    """
    import os as _os
    p = _os.environ.get("BIO_TRT_INPUT_PATH")
    if not p:
        return None

    # bt2 in bulk mode: BIO_TRT_INPUT_PATH may be a *directory* of yamls.
    # Resolve to the per-record yaml on the fly.
    if model == "bt2" and _os.path.isdir(p):
        yaml_path = _os.path.join(p, f"{name}.yaml")
        if not _os.path.isfile(yaml_path):
            return None
        key = ("bt2", yaml_path)
        if key not in _CACHE:
            _CACHE[key] = from_bt2_yaml(yaml_path)
        return _CACHE[key] or None

    key = (model, p)
    if key not in _CACHE:
        if model == "pt2":
            _CACHE[key] = from_pt2_input_json(p)
        elif model == "of3":
            _CACHE[key] = from_of3_query_json(p)
        elif model == "bt2":
            _CACHE[key] = from_bt2_yaml(p)
        elif model == "af3":
            _CACHE[key] = from_af3_input_json(p)
        else:
            _CACHE[key] = {}
    table = _CACHE[key]
    if model in ("bt2", "af3"):
        # Single-record formats — return the table directly.
        return table or None
    return table.get(name)
