#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Template structure utilities for lantern-engine inputs (runs inside the runtime venv: gemmi).

Every subcommand prints one JSON line on stdout; lantern-engine's CLI parses it.

  fetch      --pdb-id 7xyz --cache DIR                      → {"path": ".../7xyz.cif"}
  from-chain --cif F --auth-chain B --query SEQ --out F.cif  → single-chain mmCIF + queryIndices/templateIndices
  from-m8    --m8 F --query SEQ --out-dir D [--top 4] [--mmcif-dir D] [--cache D] [--max-date YYYY-MM-DD]
             → {"templates": [{mmcifPath, queryIndices, templateIndices, pdb_id, auth_chain, …}]}
  multimer   --cif F --map C=A,D=B --sequences C=SEQ,D=SEQ --out F.cif
             → multi-chain template: chain-keyed queryIndices (chain-local) / templateIndices (flat over the saved CIF)

Index conventions follow the AF3 input schema: 0-based, queryIndices into the
query sequence, templateIndices into the template chain's polymer sequence as
written to the saved mmCIF (one entity per kept chain, in the given order).
Alignment for from-chain uses a global Needleman–Wunsch with BLOSUM62 (identity
when the sequences are byte-identical, as for self-templates).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import gemmi

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K",
}

# BLOSUM62 (standard 20 aa; everything else scores -4 like a mismatch to a rare residue)
_B62_ROWS = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_lines = [ln.split() for ln in _B62_ROWS.strip().splitlines()]
_cols = _lines[0]
BLOSUM62 = {(r[0], c): int(v) for r in _lines[1:] for c, v in zip(_cols, r[1:])}


def global_align(q: str, t: str, gap_open: int = -10, gap_ext: int = -1) -> list[tuple[int, int]]:
    """Needleman–Wunsch with affine gaps; returns aligned (query_index, template_index) pairs (0-based)."""
    if q == t:
        return [(i, i) for i in range(len(q))]
    n, m = len(q), len(t)
    NEG = -10 ** 9
    M = [[NEG] * (m + 1) for _ in range(n + 1)]
    X = [[NEG] * (m + 1) for _ in range(n + 1)]   # gap in template (query residue unpaired)
    Y = [[NEG] * (m + 1) for _ in range(n + 1)]   # gap in query
    M[0][0] = 0
    for i in range(1, n + 1):
        X[i][0] = gap_open + (i - 1) * gap_ext
    for j in range(1, m + 1):
        Y[0][j] = gap_open + (j - 1) * gap_ext
    for i in range(1, n + 1):
        qi = q[i - 1]
        for j in range(1, m + 1):
            s = BLOSUM62.get((qi, t[j - 1]), -4)
            M[i][j] = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + s
            X[i][j] = max(M[i - 1][j] + gap_open, X[i - 1][j] + gap_ext, Y[i - 1][j] + gap_open)
            Y[i][j] = max(M[i][j - 1] + gap_open, Y[i][j - 1] + gap_ext, X[i][j - 1] + gap_open)
    i, j = n, m
    state = max((M[n][m], "M"), (X[n][m], "X"), (Y[n][m], "Y"))[1]
    pairs = []
    while i > 0 or j > 0:
        if state == "M":
            pairs.append((i - 1, j - 1))
            s = BLOSUM62.get((q[i - 1], t[j - 1]), -4)
            prev = M[i][j] - s
            i, j = i - 1, j - 1
            state = "M" if M[i][j] == prev else "X" if X[i][j] == prev else "Y"
        elif state == "X":
            prev = X[i][j]
            i -= 1
            if i == 0 and j == 0:
                break
            state = "M" if M[i][j] + gap_open == prev else "X" if X[i][j] + gap_ext == prev else "Y"
        else:
            prev = Y[i][j]
            j -= 1
            if i == 0 and j == 0:
                break
            state = "M" if M[i][j] + gap_open == prev else "Y" if Y[i][j] + gap_ext == prev else "X"
        if i == 0 and j == 0:
            break
    pairs.reverse()
    return pairs


# ── mmCIF handling ────────────────────────────────────────────────────────

def read_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    return st


def chain_sequence(chain: gemmi.Chain) -> tuple[str, list[gemmi.Residue]]:
    """One-letter sequence of the polymer residues (observed residues only), with the residues.

    Only the chain's polymer span counts: bound amino-acid ligands (a free ASN
    next to chain B of 12AS, for instance) are HETATM groups in their own
    subchain and would otherwise be mistaken for a polymer residue."""
    seq, residues = [], []
    span = chain.get_polymer()
    source = list(span) if len(span) else [r for r in chain if r.het_flag != "H" or r.name in ("MSE", "SEC", "PYL")]
    for res in source:
        code = THREE_TO_ONE.get(res.name)
        if code is None:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            code = info.one_letter_code.upper() if info.one_letter_code.strip() else "X"
        seq.append(code)
        residues.append(res)
    return "".join(seq), residues


def write_chains(st: gemmi.Structure, auth_chains: list[str], out: Path) -> list[str]:
    """Save a structure holding only the given chains (model 1), renamed A, B, … in that order.
    Returns the polymer sequences in saved order."""
    model = st[0]
    new = gemmi.Structure()
    new.name = st.name
    new.cell = st.cell
    new.spacegroup_hm = st.spacegroup_hm
    nm = gemmi.Model("1")
    seqs = []
    for i, auth in enumerate(auth_chains):
        src = model.find_chain(auth)
        if src is None:
            raise SystemExit(f"chain {auth!r} not in {st.name} (have {[c.name for c in model]})")
        seq, residues = chain_sequence(src)
        ch = gemmi.Chain(chr(ord("A") + i))
        for res in residues:
            r = res.clone()
            r.het_flag = "A"
            r.subchain = ch.name       # label_asym_id == auth_asym_id: parsers that read either agree
            r.entity_id = ""
            r.label_seq = None         # gemmi only renumbers residues without a label_seq_id
            ch.add_residue(r)
        nm.add_chain(ch)
        seqs.append(seq)
    new.add_model(nm)
    new.setup_entities()               # one polymer entity per saved chain
    # Readers such as Boltz's mmCIF parser align _entity_poly_seq to the
    # polymer and index into it, so the entity sequence must be exactly the
    # saved residues, numbered 1..N (label_seq_id) in that order.
    for ch in new[0]:
        poly = ch.get_polymer()
        ent = new.get_entity_of(poly)
        if ent is not None:
            ent.full_sequence = [res.name for res in poly]
    new.assign_label_seq_id(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = new.make_mmcif_document()
    doc.write_file(str(out))
    return seqs


def cif_dates(path: Path) -> list[str]:
    try:
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()
        out = []
        for tag in ("_pdbx_database_status.recvd_initial_deposition_date", "_pdbx_audit_revision_history.revision_date",
                    "_citation.year"):
            for v in block.find_values(tag):
                v = gemmi.cif.as_string(v)
                if re.match(r"\d{4}", v):
                    out.append(v[:10])
        return out
    except Exception:
        return []


def fetch_pdb(pdb_id: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{pdb_id.lower()}.cif"
    if p.is_file() and p.stat().st_size > 0:
        return p
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif"
    r = subprocess.run(["curl", "-sS", "-L", "--fail", "-o", str(p), url], capture_output=True, text=True)
    if r.returncode != 0 or not p.is_file():
        raise SystemExit(f"could not fetch {url}: {r.stderr.strip()[:200]}")
    return p


# ── subcommands ───────────────────────────────────────────────────────────

def cmd_fetch(ns) -> dict:
    return {"path": str(fetch_pdb(ns.pdb_id, Path(ns.cache)).resolve())}


def cmd_from_chain(ns) -> dict:
    st = read_structure(Path(ns.cif))
    out = Path(ns.out)
    (tseq,) = write_chains(st, [ns.auth_chain], out)
    pairs = global_align(ns.query, tseq)
    ident = sum(1 for qi, ti in pairs if ns.query[qi] == tseq[ti]) / max(1, len(pairs))
    return {"mmcifPath": str(out.resolve()), "queryIndices": [qi for qi, _ in pairs],
            "templateIndices": [ti for _, ti in pairs], "template_length": len(tseq),
            "n_aligned": len(pairs), "identity": ident}


def cmd_from_m8(ns) -> dict:
    """Top hits of an MMseqs2/ColabFold m8 file → template entries.

    m8 columns (ColabFold `pdb70.m8` / CASP17 `templates.m8`):
      query target fident alnlen mismatch gapopen qstart qend tstart tend evalue bits [cigar]
    Targets look like `6phg_A` (PDB id + auth chain). The alignment is recomputed
    on the observed template chain with the global aligner so the indices refer
    to the saved single-chain mmCIF."""
    rows = []
    for ln in Path(ns.m8).read_text().splitlines():
        f = ln.rstrip("\n").split("\t")
        if len(f) < 12:
            continue
        target = f[1]
        m = re.match(r"^([0-9A-Za-z]{4})[_.]([A-Za-z0-9]+)", target)
        if not m:
            continue
        rows.append({"pdb_id": m.group(1).lower(), "auth_chain": m.group(2), "fident": float(f[2]),
                     "evalue": float(f[10]), "bits": float(f[11])})
    rows.sort(key=lambda r: -r["bits"])
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    templates, seen = [], set()
    for r in rows:
        key = (r["pdb_id"], r["auth_chain"])
        if key in seen:
            continue
        seen.add(key)
        try:
            if ns.mmcif_dir and (Path(ns.mmcif_dir) / f"{r['pdb_id']}.cif").is_file():
                cif = Path(ns.mmcif_dir) / f"{r['pdb_id']}.cif"
            else:
                cif = fetch_pdb(r["pdb_id"], Path(ns.cache))
            if ns.max_date:
                dates = [d for d in cif_dates(cif) if re.match(r"\d{4}-\d{2}-\d{2}", d)]
                if dates and min(dates) > ns.max_date:
                    continue
            st = read_structure(cif)
            out = out_dir / f"{r['pdb_id']}_{r['auth_chain']}.cif"
            (tseq,) = write_chains(st, [r["auth_chain"]], out)
            if not tseq:
                continue
            pairs = global_align(ns.query, tseq)
            if len(pairs) < 10:
                continue
            ident = sum(1 for qi, ti in pairs if ns.query[qi] == tseq[ti]) / len(pairs)
            templates.append({"mmcifPath": str(out.resolve()), "queryIndices": [qi for qi, _ in pairs],
                              "templateIndices": [ti for _, ti in pairs], "pdb_id": r["pdb_id"],
                              "auth_chain": r["auth_chain"], "n_aligned": len(pairs), "identity": ident,
                              "bits": r["bits"], "evalue": r["evalue"]})
        except SystemExit as e:
            print(f"skip {r['pdb_id']}_{r['auth_chain']}: {e}", file=sys.stderr)
            continue
        if len(templates) >= ns.top:
            break
    return {"templates": templates}


def cmd_multimer(ns) -> dict:
    mapping = [tok.split("=", 1) for tok in ns.map.split(",")]
    seqs = dict(tok.split("=", 1) for tok in ns.sequences.split(","))
    st = read_structure(Path(ns.cif))
    out = Path(ns.out)
    tseqs = write_chains(st, [t for _, t in mapping], out)
    q_idx, t_idx, offset = {}, {}, 0
    for (q, t), tseq in zip(mapping, tseqs):
        qseq = seqs[q]
        pairs = global_align(qseq, tseq)
        ident = sum(1 for qi, ti in pairs if qseq[qi] == tseq[ti]) / max(1, len(pairs))
        if ident < 0.3:
            raise SystemExit(f"chain {q}: template chain {t} identity {ident:.0%} — wrong chain mapping?")
        q_idx[q] = [qi for qi, _ in pairs]
        t_idx[q] = [offset + ti for _, ti in pairs]
        offset += len(tseq)
    return {"mmcifPath": str(out.resolve()), "queryIndices": q_idx, "templateIndices": t_idx}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch"); s.add_argument("--pdb-id", required=True); s.add_argument("--cache", required=True)
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("from-chain"); s.add_argument("--cif", required=True); s.add_argument("--auth-chain", required=True)
    s.add_argument("--query", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_from_chain)
    s = sub.add_parser("from-m8"); s.add_argument("--m8", required=True); s.add_argument("--query", required=True)
    s.add_argument("--out-dir", required=True); s.add_argument("--top", type=int, default=4)
    s.add_argument("--mmcif-dir", default=None); s.add_argument("--cache", required=True); s.add_argument("--max-date", default=None)
    s.set_defaults(fn=cmd_from_m8)
    s = sub.add_parser("multimer"); s.add_argument("--cif", required=True); s.add_argument("--map", required=True)
    s.add_argument("--sequences", required=True); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_multimer)
    ns = ap.parse_args()
    print(json.dumps(ns.fn(ns)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
