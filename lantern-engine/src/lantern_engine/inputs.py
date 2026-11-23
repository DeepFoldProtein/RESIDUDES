"""Assemble an AF3-dialect input.json — the one file every lantern-engine model runs from.

Ports the CASP17 input toolbox (gen_inputs/combine.py) without its AlphaFold 3
dependency: stoichiometry, MSAs, per-chain templates, pinned and self
templates, multi-chain (interface) templates, ligands, covalently attached
glycans and contact constraints. Structure files are handled by
core/scripts/template_tools.py inside the runtime venv (gemmi); this module is
stdlib and only arranges the JSON.

Chain ids: source chains A, B, C … (FASTA order) expand by stoichiometry into
output chains in order (A2B1 → A, B for the first source chain, C for the
second); ligands and glycans take the next free letters. Fields consumed by the
models through core/scripts/af3_input_adapter.py: `templates` (pt2/of3/bt2),
`multimerTemplates` (pt2/of3), `bondedAtomPairs` (bt2; pt2 as covalent_bonds),
`constraint.contact` (pt2).
"""
from __future__ import annotations

import json
import re
import shutil
import string
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .msa import Chain

_LETTERS = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
GLYCAN_LINK_ATOM = {"N": "ND2", "S": "OG", "T": "OG1"}


class InputError(RuntimeError):
    pass


# ── specs (same grammar as the CASP17 gen_inputs flags) ───────────────────

def parse_ligand_spec(spec: str) -> dict:
    """`NAME[*COUNT]=smi:<SMILES> | smi-file:<path> | ccd:<CODE>[,<CODE>…]`."""
    if "=" not in spec:
        raise InputError(f"--ligand expects NAME[*COUNT]=SPEC, got {spec!r}")
    name_part, body = spec.split("=", 1)
    name, count = name_part, 1
    if "*" in name_part:
        name, cnt = name_part.rsplit("*", 1)
        if not cnt.isdigit() or int(cnt) < 1:
            raise InputError(f"--ligand count must be a positive integer (got {cnt!r})")
        count = int(cnt)
    out: dict = {"name": name.strip(), "count": count}
    body = body.strip()
    if body.startswith("smi:"):
        out["smiles"] = body[4:].strip()
    elif body.startswith("smi-file:"):
        p = Path(body[9:].strip())
        tok = next((ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")), "")
        if not tok:
            raise InputError(f"--ligand smi-file: no SMILES in {p}")
        out["smiles"] = tok
    elif body.startswith("ccd:"):
        out["ccdCodes"] = [c.strip() for c in body[4:].split(",") if c.strip()]
    else:
        raise InputError(f"--ligand SPEC must start with smi:/smi-file:/ccd: (got {body!r})")
    if not out.get("smiles") and not out.get("ccdCodes"):
        raise InputError(f"--ligand: empty specification in {spec!r}")
    return out


def parse_glycan_spec(spec: str) -> dict:
    """`<CHAIN><POS>:<CCD>[-<CCD>…][@<protAtom>[/<interAtom>]]` — e.g. A29:NAG-NAG-BMA."""
    prot_atom, inter_atom = None, "O4"
    if "@" in spec:
        spec, atoms = spec.rsplit("@", 1)
        if "/" in atoms:
            p, i = atoms.split("/", 1)
            prot_atom, inter_atom = p.strip() or None, i.strip() or "O4"
        else:
            prot_atom = atoms.strip() or None
    if ":" not in spec:
        raise InputError(f"--glycan expects <CHAIN><POS>:<CCD>[-<CCD>…], got {spec!r}")
    site, _, glyco = spec.partition(":")
    m = re.fullmatch(r"([A-Z])(\d+)", site.strip())
    if not m:
        raise InputError(f"--glycan site must be <CHAIN><POS> (e.g. A29), got {site!r}")
    ccds = [c.strip() for c in glyco.split("-") if c.strip()]
    if not ccds:
        raise InputError(f"--glycan needs at least one CCD component in {spec!r}")
    return {"chain": m.group(1), "pos": int(m.group(2)), "ccds": ccds, "prot_atom": prot_atom, "inter_atom": inter_atom}


def parse_contact_spec(spec: str) -> dict:
    """`<C1>:<P1>[.<A1>]=<C2>:<P2>[.<A2>][@max=N[,min=M]]` (Protenix contact constraint)."""
    body, max_d, min_d = spec, 6.0, 0.0
    if "@" in spec:
        body, dist = spec.split("@", 1)
        for kv in dist.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                fv = float(v)
            except ValueError:
                raise InputError(f"--contact distance must be a number: {kv!r}")
            if k.strip() == "max":
                max_d = fv
            elif k.strip() == "min":
                min_d = fv
            else:
                raise InputError(f"--contact distance key must be max or min: {k!r}")
    if body.count("=") != 1:
        raise InputError(f"--contact needs exactly one '=': {spec!r}")

    def side(s: str):
        parts = s.split(":")
        if len(parts) != 2:
            raise InputError(f"--contact side must be <chain>:<pos>[.<atom>]: {s!r}")
        pos, atom = (parts[1].split(".", 1) + [None])[:2] if "." in parts[1] else (parts[1], None)
        if not pos.isdigit():
            raise InputError(f"--contact position must be an integer: {pos!r}")
        return parts[0], int(pos), atom

    c1, p1, a1 = side(body.split("=")[0])
    c2, p2, a2 = side(body.split("=")[1])
    if max_d <= min_d:
        raise InputError(f"--contact max ({max_d}) must exceed min ({min_d})")
    return {"chain1": c1, "pos1": p1, "atom1": a1, "chain2": c2, "pos2": p2, "atom2": a2,
            "max_distance": max_d, "min_distance": min_d}


def parse_multimer_template_spec(spec: str) -> tuple[Path, list[tuple[str, str]]]:
    """`CIF:Q1=T1,Q2=T2,…` — output chain letter Q ← template auth chain T (order = concatenation order)."""
    if ":" not in spec:
        raise InputError(f"--multimer-template expects CIF:Q=T,… (got {spec!r})")
    cif, mapping = spec.rsplit(":", 1) if spec.count(":") > 1 and "=" not in spec.split(":")[-1] else spec.split(":", 1)
    pairs, seen = [], set()
    for tok in mapping.split(","):
        if "=" not in tok:
            raise InputError(f"--multimer-template token {tok!r} is not Q=T")
        q, t = tok.split("=", 1)
        q, t = q.strip(), t.strip()
        if q in seen:
            raise InputError(f"--multimer-template: duplicate query chain {q}")
        seen.add(q)
        pairs.append((q, t))
    if len(pairs) < 2:
        raise InputError("--multimer-template needs at least two chain mappings")
    return Path(cif), pairs


# ── template tools (gemmi) run inside the runtime venv ────────────────────

@dataclass
class TemplateTools:
    python: Path
    script: Path
    cache_dir: Path

    def run(self, *args: str) -> dict:
        cmd = [str(self.python), str(self.script), *args]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise InputError(f"template_tools {args[0]} failed: {res.stderr.strip()[-600:]}")
        try:
            return json.loads(res.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise InputError(f"template_tools {args[0]}: no JSON result ({res.stdout[-200:]!r})")


# ── assembly ──────────────────────────────────────────────────────────────

@dataclass
class Assembly:
    name: str
    chains: list[Chain]
    seeds: list[int] = field(default_factory=lambda: [1])
    unpaired: dict[str, Path] = field(default_factory=dict)      # source letter → a3m
    paired: dict[str, Path] = field(default_factory=dict)
    templates: dict[str, list[dict]] = field(default_factory=dict)  # source letter → templates entries
    ligands: list[dict] = field(default_factory=list)
    glycans: list[dict] = field(default_factory=list)
    contacts: list[dict] = field(default_factory=list)
    multimer_templates: list[dict] = field(default_factory=list)   # built entries
    notes: dict = field(default_factory=dict)

    # output letters per source letter, in assembly order
    def output_ids(self) -> dict[str, list[str]]:
        out, pos = {}, 0
        for ch in self.chains:
            out[ch.letter] = [_LETTERS[pos + i] for i in range(ch.copies)]
            pos += ch.copies
        return out

    def build(self, out_dir: Path) -> dict:
        ids = self.output_ids()
        used = {cid for v in ids.values() for cid in v}
        pos = sum(ch.copies for ch in self.chains)
        sequences = []
        for ch in self.chains:
            body: dict = {"id": ids[ch.letter][0] if ch.copies == 1 else ids[ch.letter], "sequence": ch.seq}
            if ch.letter in self.unpaired:
                body["unpairedMsaPath"] = str(self.unpaired[ch.letter].resolve())
            if ch.letter in self.paired:
                body["pairedMsaPath"] = str(self.paired[ch.letter].resolve())
            body["templates"] = list(self.templates.get(ch.letter, []))
            sequences.append({"protein": body})

        def next_free() -> str:
            nonlocal pos
            while _LETTERS[pos] in used:
                pos += 1
            cid = _LETTERS[pos]
            used.add(cid)
            pos += 1
            return cid

        for lig in self.ligands:
            lig_ids = [next_free() for _ in range(lig["count"])]
            body = {"id": lig_ids[0] if len(lig_ids) == 1 else lig_ids}
            if lig.get("smiles"):
                body["smiles"] = lig["smiles"]
            else:
                body["ccdCodes"] = lig["ccdCodes"]
            if lig.get("name"):
                body["description"] = lig["name"]
            sequences.append({"ligand": body})

        bonded = []
        for g in self.glycans:
            src = next((c for c in self.chains if c.letter == g["chain"]), None)
            if src is None:
                raise InputError(f"--glycan chain {g['chain']} is not a source chain")
            if not 1 <= g["pos"] <= len(src.seq):
                raise InputError(f"--glycan {g['chain']}{g['pos']}: position out of range (chain has {len(src.seq)} residues)")
            res = src.seq[g["pos"] - 1]
            link = g["prot_atom"] or GLYCAN_LINK_ATOM.get(res)
            if link is None:
                raise InputError(f"--glycan {g['chain']}{g['pos']}: residue {res} is not N/S/T; give the link atom with @<atom>")
            for out_id in ids[src.letter]:                       # every copy of the source chain
                gid = next_free()
                sequences.append({"ligand": {"id": gid, "ccdCodes": list(g["ccds"]),
                                             "description": f"glycan_{g['chain']}{g['pos']}"}})
                bonded.append([[out_id, g["pos"], link], [gid, 1, "C1"]])
                for ci in range(1, len(g["ccds"])):
                    bonded.append([[gid, ci, g["inter_atom"]], [gid, ci + 1, "C1"]])

        doc: dict = {"name": self.name, "modelSeeds": list(self.seeds), "sequences": sequences,
                     "dialect": "alphafold3", "version": 2}
        if bonded:
            doc["bondedAtomPairs"] = bonded
        if self.multimer_templates:
            doc["multimerTemplates"] = self.multimer_templates
        if self.contacts:
            doc["constraint"] = {"contact": expand_contacts(self.contacts, self.chains)}
        return doc


def expand_contacts(specs: list[dict], chains: list[Chain]) -> list[dict]:
    """Protenix `constraint.contact[]`: entity = 1-based source chain index; replicated copy i ↔ copy i."""
    idx = {c.letter: (i, c.copies) for i, c in enumerate(chains, start=1)}
    out = []
    for sp in specs:
        for c in (sp["chain1"], sp["chain2"]):
            if c not in idx:
                raise InputError(f"--contact chain {c} is not a source chain (have {sorted(idx)})")
        e1, n1 = idx[sp["chain1"]]
        e2, n2 = idx[sp["chain2"]]
        if e1 == e2:
            raise InputError("--contact: same-chain contacts are not supported (Protenix rejects them)")
        for i in range(1, min(n1, n2) + 1):
            entry = {"entity1": e1, "copy1": i, "position1": sp["pos1"], "entity2": e2, "copy2": i,
                     "position2": sp["pos2"], "max_distance": sp["max_distance"], "min_distance": sp["min_distance"]}
            if sp["atom1"]:
                entry["atom1"] = sp["atom1"]
            if sp["atom2"]:
                entry["atom2"] = sp["atom2"]
            out.append(entry)
    return out


# ── template entries (via template_tools) ─────────────────────────────────

def templates_from_m8(tools: TemplateTools, chain: Chain, m8_path: Path, out_dir: Path, *, top: int = 4,
                      mmcif_dir: Path | None = None, max_date: str | None = None, log=print) -> list[dict]:
    """MMseqs2/ColabFold template hits → AF3 `templates` entries (single-chain mmCIF + index maps)."""
    args = ["from-m8", "--m8", str(m8_path), "--query", chain.seq, "--out-dir", str(out_dir / "templates" / chain.letter),
            "--top", str(top), "--cache", str(tools.cache_dir / "mmcif")]
    if mmcif_dir:
        args += ["--mmcif-dir", str(mmcif_dir)]
    if max_date:
        args += ["--max-date", max_date]
    r = tools.run(*args)
    for t in r.get("templates", []):
        log(f"  template {chain.letter}: {t.get('pdb_id')}_{t.get('auth_chain')} "
            f"({t.get('n_aligned')} aligned, {t.get('identity', 0):.0%} identity)")
    return [{k: v for k, v in t.items() if k in ("mmcifPath", "queryIndices", "templateIndices")} for t in r.get("templates", [])]


def template_from_structure(tools: TemplateTools, chain: Chain, cif: Path, auth_chain: str, out_dir: Path,
                            *, tag: str) -> dict:
    """One template entry from a chain of a structure file (pinned PDB chain or a previous prediction)."""
    r = tools.run("from-chain", "--cif", str(cif), "--auth-chain", auth_chain, "--query", chain.seq,
                  "--out", str(out_dir / "templates" / chain.letter / f"{tag}.cif"))
    return {k: v for k, v in r.items() if k in ("mmcifPath", "queryIndices", "templateIndices")}


def multimer_template(tools: TemplateTools, assembly: Assembly, cif: Path, pairs: list[tuple[str, str]],
                      out_dir: Path, tag: str) -> dict:
    ids = assembly.output_ids()
    seq_of = {}
    for ch in assembly.chains:
        for cid in ids[ch.letter]:
            seq_of[cid] = ch.seq
    for q, _ in pairs:
        if q not in seq_of:
            raise InputError(f"--multimer-template: output chain {q} does not exist (have {sorted(seq_of)})")
    spec = ",".join(f"{q}={t}" for q, t in pairs)
    seqs = ",".join(f"{q}={seq_of[q]}" for q, _ in pairs)
    r = tools.run("multimer", "--cif", str(cif), "--map", spec, "--sequences", seqs,
                  "--out", str(out_dir / "templates" / f"{tag}.cif"))
    return {"mmcifPath": r["mmcifPath"], "queryIndices": r["queryIndices"], "templateIndices": r["templateIndices"]}


def resolve_pdb_source(src: str, tools: TemplateTools) -> Path:
    """A path to a structure file, or a 4-letter PDB id fetched into the cache."""
    p = Path(src)
    if p.is_file():
        return p
    if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", src):
        r = tools.run("fetch", "--pdb-id", src.lower(), "--cache", str(tools.cache_dir / "mmcif"))
        return Path(r["path"])
    raise InputError(f"template source {src!r} is neither a file nor a PDB id")


def write_input(doc: dict, out_dir: Path, *, pairing_mode: str | None, provenance: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "input.json"
    p.write_text(json.dumps(doc, indent=2) + "\n")
    if pairing_mode:
        (out_dir / "pairing_mode.txt").write_text(pairing_mode + "\n")
    provenance = dict(provenance, generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out_dir / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return p


def find_result_model(seed_dir: Path) -> Path:
    """Top-ranked model of a lantern-engine seed directory (for self-templates)."""
    for cand in ("sample_0/model.cif", "model.cif"):
        p = seed_dir / cand
        if p.is_file():
            return p
    if seed_dir.suffix == ".cif" and seed_dir.is_file():
        return seed_dir
    raise InputError(f"no model.cif under {seed_dir}")


def which_or(cmd: str, fallback: str) -> str:
    return cmd if shutil.which(cmd) else fallback
