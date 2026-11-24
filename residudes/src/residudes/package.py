"""`residudes package T`: the approved slate, checked and formatted for CASP (APPROVED -> PACKAGED). A person submits.

  residudes package T --server|--human [--method TEXT] [--casp-id T1234] [--mmcif] [--check-only]

For each model of the slate a person approved (the version their `residudes approve --what "slate vN"` named, else the
latest proposal):
  1. geometry checks, the ones the selection skill lists: polymer chains and residues against the sequence and the
     stoichiometry, chain breaks (C-N > 2.0 A), non-Pro cis peptides, inter-chain heavy-atom pairs under 1.5 A;
     reported, not blocking (relaxation fixes clashes; a person reads the report)
  2. a polymer-only copy: TS files carry ATOM records only, ligands go in LG files (a CASP rule learned in CASP17)
  3. the CASP file from the team's casp17-utils (`casp17 prepare`), which validates the model and writes the header
     with the author code from its own .env; residudes never reads that file and never submits
Writes <dossier>/package/ (the files and checks.md), records evidence `package`, and moves the target to PACKAGED.
Ligand (LG) files are not written here: a ligand target's poses go through the team's LG pipeline (receptor block
with a TER per chain, the ligand as MDL), which needs RDKit.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path

# the team's CASP submission tooling, which is not public: `residudes.toml` [tools] casp17_utils or
# $CASP17_UTILS points at it, and without it `residudes package` stops after the geometry checks
UTILS_CANDIDATES = ("/gpfs/deepfold/casp/casp17/casp17-utils",)


def utils_dir(cfg) -> Path | None:
    p = (cfg.extra.get("tools", {}) or {}).get("casp17_utils") or os.environ.get("CASP17_UTILS")
    for c in ([p] if p else []) + list(UTILS_CANDIDATES):
        if c and (Path(c) / "pyproject.toml").exists():
            return Path(c)
    return None


def _dihedral(a, b, c, d) -> float:
    import gemmi
    return math.degrees(gemmi.calculate_dihedral(a, b, c, d))


def check_model(path: str, sto: str, seqs: list[str]) -> dict:
    """Geometry of one model: chains and residues per chain against the expected ones, breaks, cis, clashes."""
    import gemmi

    from residudes import gate
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_waters()
    model = st[0]
    base, _, at = sto.partition("@")                     # A1@B: a per-chain run of entity B, modelled as chain A
    ents = gate.entity_chains(base)
    seq_of = lambda e: seqs[ord(e) - 65] if 0 <= ord(e) - 65 < len(seqs) else None
    expect = {c: seq_of(at or e) for e, cs in ents.items() for c in cs}
    poly = gate.polymer_chains(model)
    out = {"chains": len(poly), "expected_chains": gate.expected_chains(base) if not at else 1, "length_mismatch": [],
           "breaks": 0, "cis_nonpro": 0, "clash_pairs": 0, "atoms": sum(len(r) for ch in model for r in ch)}
    for ch in model:
        if ch.name not in poly:
            continue
        res = [r for r in ch if gate._is_poly(r.name)]
        want = expect.get(ch.name)
        if want and len(res) != len(want):
            out["length_mismatch"].append(f"{ch.name}: {len(res)} residues, sequence {len(want)}")
        for r1, r2 in zip(res, res[1:]):
            c, n = r1.find_atom("C", "*"), r2.find_atom("N", "*")
            if c and n:
                if c.pos.dist(n.pos) > 2.0:
                    out["breaks"] += 1
                    continue
                ca1, ca2 = r1.find_atom("CA", "*"), r2.find_atom("CA", "*")
                if ca1 and ca2 and r2.name != "PRO" and abs(_dihedral(ca1.pos, c.pos, n.pos, ca2.pos)) < 30:
                    out["cis_nonpro"] += 1
    out["clash_pairs"] = gate.evaluate(str(path), {"gates": [{"name": "c", "type": "clash", "cutoff": 1.5, "max": 10 ** 9}]},
                                       None)["c_value"]
    out["ok"] = not out["length_mismatch"] and (out["expected_chains"] is None or out["chains"] == out["expected_chains"])
    return out


def polymer_copy(path: str, out: Path) -> Path:
    """The model with ligands and waters removed (TS carries ATOM records only), pLDDT kept in the B-factors."""
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()                                # remove_ligands_and_waters needs entity types
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    st.setup_entities()
    out.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(out))
    return out


def approved_slate(d) -> dict:
    """The slate proposal the latest approval names ('slate v2'), else the latest proposal."""
    props = [x["value"] for x in d.evidence.get("slate_proposal", [])]
    if not props:
        raise ValueError(f"{d.target} has no slate proposal")
    what = (d.approvals[-1]["what"] if d.approvals else "") or ""
    m = re.search(r"\bv(\d+)\b", what)
    if not m:                                          # the approval names no version: the latest proposal
        return props[-1]
    hit = next((p for p in props if str(p.get("version")) == m.group(1)), None)
    if hit is None:
        raise ValueError(f"the approval names slate v{m.group(1)} ({what!r}), but the slates are "
                         f"v{', v'.join(str(p.get('version')) for p in props)}: a person approves the right one again")
    return hit


def package_cmd(cfg, d, mode: str | None, method: str, casp_id: str | None, mmcif: bool, check_only: bool, actor: str) -> str:
    import csv
    from residudes import gate
    from residudes.dossier import Dossier, now
    from residudes.slate import _model_key
    if d.stage != "APPROVED" and not check_only:
        raise ValueError(f"{d.target} is at {d.stage}: packaging needs an approved slate (a person runs `residudes approve`); "
                         "--check-only runs the checks anyway")
    if not check_only and mode not in ("server", "human"):
        raise ValueError("say which author code: --server or --human")
    ddir = Dossier.path(cfg.targets_dir, d.target).parent.resolve()
    prop = approved_slate(d)
    with open(ddir / "pool.csv") as fh:
        paths = {_model_key(r): r["path"] for r in csv.DictReader(fh)}
    seqs = gate.fasta_records(d.sequence.get("text", ""))
    tid = casp_id or re.sub(r"[^A-Za-z0-9].*$", "", d.target)
    pdir = ddir / "package"
    if not check_only and pdir.exists():               # an earlier run's files are kept beside, never mixed in
        pdir.rename(pdir.with_name(f"package.replaced-{now().replace(':', '')}"))
    pdir.mkdir(exist_ok=True)
    utils = utils_dir(cfg)
    if not check_only and (not utils or not shutil.which("uv")):
        raise FileNotFoundError("casp17-utils not found: set [tools] casp17_utils in residudes.toml or CASP17_UTILS, "
                                "or run with --check-only")
    rows, files, problems = [], [], []
    models = prop.get("models", [])
    for slot, m in enumerate(models, start=1):
        sto = m.split("__")[0]
        src = paths.get(m)
        if not src or not Path(src).exists():
            rows.append({"slot": slot, "model": m, "error": "model file not found"})
            problems.append(f"slot {slot}: model file not found")
            continue
        chk = check_model(src, sto, seqs)
        rows.append({"slot": slot, "model": m, **chk})
        if not chk["ok"]:
            problems.append(f"slot {slot}: " + ("; ".join(chk["length_mismatch"]) or f"{chk['chains']}/{chk['expected_chains']} chains"))
        if check_only:
            continue
        big = (chk["expected_chains"] or 0) > 26 or chk["chains"] > 26 or chk["atoms"] > 99_999
        as_cif = mmcif or big                          # PDB cannot hold two-letter chain ids or 100k atoms
        poly = polymer_copy(src, pdir / "polymer" / f"model_{slot}.cif").resolve()
        out = (pdir / (f"{tid}_{slot}.cif" if as_cif else f"{tid}_{slot}.pdb")).resolve()
        cmd = ["uv", "run", "--project", str(utils), "casp17", "prepare", "-i", str(poly), "-t", tid, "-n", str(slot),
               "-o", str(out), f"--{mode}", "--method", method, "--stoich", sto.split("@")[0]] + (["--mmcif"] if as_cif else [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(utils), timeout=600)
        except subprocess.TimeoutExpired:
            problems.append(f"slot {slot}: casp17 prepare timed out")
            continue
        if r.returncode != 0:
            problems.append(f"slot {slot}: casp17 prepare failed: {(r.stderr or r.stdout).strip()[-300:]}")
            continue
        files.append(str(out))
        if big and not mmcif:
            rows[-1]["note"] = "written as mmCIF (more than 26 chains or 99,999 atoms)"
    lines = [f"# {d.target}: package checks (slate v{prop.get('version')})", "",
             "| slot | model | ok | chains | length | breaks | non-Pro cis | pairs < 1.5 Å |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['slot']} | {r['model']} | **no** | {r['error']} | | | | |")
            continue
        lines.append(f"| {r['slot']} | {r['model']} | {'yes' if r['ok'] else '**no**'} | {r['chains']}/{r['expected_chains']} | "
                     f"{'; '.join(r['length_mismatch']) or 'ok'} | {r['breaks']} | {r['cis_nonpro']} | {r['clash_pairs']} |")
    lines += ["", "Clashes are fixed by relaxation; breaks, cis peptides and length mismatches need a person's look."]
    if problems:
        lines += ["", "**Needs a person before anything is submitted:**"] + [f"- {x}" for x in problems]
    (pdir / "checks.md").write_text("\n".join(lines) + "\n")
    d.evidence.setdefault("package", []).append({"time": now(), "by": actor, "value": {
        "slate": prop.get("version"), "files": files, "checks": rows, "problems": problems, "mode": mode,
        "check_only": check_only, "dir": str(pdir)}})
    text = "\n".join(lines) + ("\n\n" + "\n".join(files) if files else "")
    if check_only:
        return text
    if problems or len(files) != len(models):
        d.note(actor, f"NEEDS HUMAN: packaging of slate v{prop.get('version')} is incomplete or flagged: " + "; ".join(problems))
        return ("INCOMPLETE: " + text + f"\n\n{d.target} stays {d.stage}. Nothing is ready to submit: a person looks at "
                f"{pdir / 'checks.md'} and either fixes the slate or moves it on with `residudes stage {d.target} PACKAGED`.")
    d.move("PACKAGED", actor, f"{len(files)} files in {pdir}")
    return text + f"\n\n{d.target} is {d.stage}: a person submits these files (casp17-utils `submit`)"
