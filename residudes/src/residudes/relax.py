"""`residudes relax T`: the campaign's relaxation, on models a person has already chosen.

This is the CASP17 recipe, not a new one. `gromacs/relax/minimization.sh` runs both orders of the
two engines on every input — GROMACS energy minimisation then Rosetta (`gromacs-rosetta`), and
Rosetta then GROMACS (`rosetta-gromacs`) — under position restraints, so the model is repaired
rather than moved: backbone at 1,000,000 kJ/mol/nm² and side chains at 10,000 by default, with
disulfides read from an SSBOND file (or found by distance) and optional distance restraints.

**It ranks nothing.** The slate and the gates never read what this writes, and `residudes relax`
refuses to run before a slate exists, because choosing which models to repair is a decision that
comes after choosing which models to submit. What it records is the arithmetic of the repair:
clashes, non-proline cis peptides and chain breaks before and after, and how far the coordinates
moved, per engine order. A relaxation that fixes nothing and moves nothing is the usual outcome and
is worth seeing; one that moves a model is worth seeing before it is submitted.

    residudes relax T                      the slate's models, both orders, default restraints
    residudes relax T --models K1,K2       named models instead
    residudes relax T --sc-fc 1000         looser side chains
    residudes relax T --read               the finished runs into the dossier
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

VARIANTS = ("gromacs", "gromacs-rosetta", "rosetta", "rosetta-gromacs")
# site default: the campaign's pipeline. [tools] minimization in residudes.toml points elsewhere.
MINIMIZATION = ""                             # [tools] minimization in residudes.toml


def script(cfg) -> str:
    from residudes.msa import tool
    p = (cfg.extra.get("tools", {}) if cfg else {}).get("minimization") or MINIMIZATION
    if Path(p).exists():
        return p
    found = tool(cfg, "minimization.sh")
    if not found:
        raise FileNotFoundError(f"relaxation pipeline not found at {p}: set [tools] minimization in residudes.toml "
                                "(gromacs/relax/minimization.sh, which runs grelax and rrelax)")
    return found


def slate_models(d) -> list[str]:
    from residudes.package import approved_slate
    prop = approved_slate(d)                      # raises when there is no slate: relaxation comes after selection
    return list(prop.get("models", []))


def model_paths(cfg, d, models: list[str]) -> list[tuple[str, Path]]:
    from residudes.dossier import Dossier
    from residudes.slate import _model_key
    ddir = Dossier.path(cfg.targets_dir, d.target).parent
    pool = ddir / "pool.csv"
    if not pool.exists():
        raise FileNotFoundError(f"no pool.csv for {d.target}: `residudes pool {d.target}` first")
    with open(pool) as fh:
        paths = {_model_key(r): r["path"] for r in csv.DictReader(fh)}
    out = []
    for m in models:
        p = paths.get(m)
        if not p or not Path(p).exists():
            raise FileNotFoundError(f"{m}: model file not found in pool.csv")
        out.append((m, Path(p)))
    return out


def ca_rmsd(a: Path, b: Path) -> float | None:
    """Cα RMSD between two structures with the same residues, in place — no superposition.

    The point of a restrained relaxation is that the model does not move, so the number worth
    having is the distance it moved, not the distance after fitting it back on itself."""
    import gemmi
    try:
        sa, sb = gemmi.read_structure(str(a)), gemmi.read_structure(str(b))
    except Exception:
        return None
    for s in (sa, sb):
        s.remove_hydrogens()
        s.remove_waters()
    pa = [at.pos for ch in sa[0] for r in ch for at in r if at.name == "CA"]
    pb = [at.pos for ch in sb[0] for r in ch for at in r if at.name == "CA"]
    if not pa or len(pa) != len(pb):
        return None
    return round((sum(x.dist(y) ** 2 for x, y in zip(pa, pb)) / len(pa)) ** 0.5, 3)


def run_dir(cfg, d) -> Path:
    from residudes.dossier import Dossier
    return Dossier.path(cfg.targets_dir, d.target).parent / "relax"


def submit_cmd(cfg, d, actor: str, models: list[str] | None = None, ssbond: str = "auto",
               bb_fc: int | None = None, sc_fc: int | None = None, dstres: str | None = None,
               ligand: bool = False) -> str:
    from residudes import ops
    from residudes.dossier import now
    exe = script(cfg)
    picked = models or slate_models(d)
    if not picked:
        raise ValueError(f"{d.target}: the slate names no models to relax")
    pairs = model_paths(cfg, d, picked)
    out = run_dir(cfg, d)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, (key, src) in enumerate(pairs, start=1):
        wd = out / f"{i}_{key}"
        wd.mkdir(exist_ok=True)
        staged = wd / f"model{src.suffix}"
        shutil.copy2(src, staged)
        args = [exe, str(staged), "--ssbond", ssbond]
        if bb_fc is not None:
            args += ["--bb-fc", str(bb_fc)]
        if sc_fc is not None:
            args += ["--sc-fc", str(sc_fc)]
        if dstres:
            args += ["--dstres", str(Path(dstres).resolve())]
        if ligand:
            args += ["--ligand", "on"]
        job = ops._sbatch(" ".join(args), f"residudes-relax-{d.target}-{i}", wd / "relax.%j.log",
                          gpu=False, hours=8, cpus=8)
        jobs.append(job)
        d.runs.append({"time": now(), "by": actor, "job": job, "scheduler": "slurm",
                       "what": f"relax {key}", "state": "submitted", "out": str(wd)})
    d.note(actor, f"relaxation submitted for {len(pairs)} model(s): jobs {', '.join(jobs)}")
    return (f"relaxation of {len(pairs)} model(s), both engine orders each "
            f"(gromacs-rosetta and rosetta-gromacs), restraints bb={bb_fc or 'default'} sc={sc_fc or 'default'}\n"
            f"jobs {', '.join(jobs)} -> {out}\n"
            f"when they are done: `residudes relax {d.target} --read`")


def read_one(wd: Path, sto: str, seqs: list[str]) -> dict:
    """The input and every finished variant of one model, measured the same way."""
    from residudes.package import check_model
    src = next((p for p in sorted(wd.glob("model.*")) if p.suffix in (".cif", ".pdb")), None)
    if src is None:
        return {}
    got = {"model": str(src), "before": check_model(str(src), sto, seqs), "after": {}}
    for v in VARIANTS:
        relaxed = wd / f"{src.stem}_relaxed" / v / "relaxed.pdb"
        if not relaxed.exists():
            continue
        row = {"path": str(relaxed), **check_model(str(relaxed), sto, seqs), "ca_rmsd": ca_rmsd(src, relaxed)}
        side = relaxed.with_name("relaxed.json")
        if side.exists():
            try:
                row["pipeline"] = json.loads(side.read_text())
            except ValueError:
                pass
        got["after"][v] = row
    return got


def read_cmd(cfg, d, actor: str) -> str:
    from residudes import gate
    from residudes.dossier import now
    out = run_dir(cfg, d)
    if not out.is_dir():
        raise FileNotFoundError(f"nothing relaxed yet for {d.target}: `residudes relax {d.target}` first")
    seqs = gate.fasta_records(d.sequence.get("text", ""))
    value: dict = {}
    lines = [f"{'model':34s} {'engines':16s} {'clash':>12s} {'cis':>8s} {'breaks':>8s} {'Cα moved':>9s}"]
    for wd in sorted(p for p in out.iterdir() if p.is_dir()):
        key = wd.name.partition("_")[2]
        sto = key.split("__")[0] if "__" in key else (d.stoichiometry.get("given") or "A1")
        got = read_one(wd, sto, seqs)
        if not got:
            continue
        value[key] = got
        b = got["before"]
        if not got["after"]:
            lines.append(f"{key[:34]:34s} {'(not finished)':16s}")
            continue
        for v, a in got["after"].items():
            moved = f"{a['ca_rmsd']:.3f} A" if a.get("ca_rmsd") is not None else "-"
            lines.append(f"{key[:34]:34s} {v:16s} {b['clash_pairs']:5d} -> {a['clash_pairs']:<4d} "
                         f"{b['cis_nonpro']:3d} -> {a['cis_nonpro']:<2d} {b['breaks']:3d} -> {a['breaks']:<3d} "
                         f"{moved:>9s}")
    if not value:
        raise FileNotFoundError(f"no relaxation output under {out}")
    d.evidence.setdefault("relax", []).append({"time": now(), "by": actor, "value": value})
    lines.append("")
    lines.append("Repair, not ranking: nothing here reaches the slate or the gates. A person decides whether to "
                 "submit a relaxed file, and `residudes package` still takes the slate's own paths.")
    return "\n".join(lines)
