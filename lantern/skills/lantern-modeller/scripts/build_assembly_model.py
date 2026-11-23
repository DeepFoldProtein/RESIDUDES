#!/usr/bin/env python3
"""Thread the target's sequences onto a whole assembly with MODELLER, and write it as an AF3 multimer template.

Two uses, one script:
  homolog assembly   a homolog has the oligomer (T2461o: 3VCD's 24-mer, 63 % identical): every chain gets the
                     target sequence; --copies checks the count
  released complex   the complex itself (or a near-identical one) is public but has gaps (H2422: 9F8E, 19 of 1,172
                     residues missing; H2423: 7ACZ, 77 of 1,182): a FASTA with one record per distinct chain, each
                     template chain gets the record it matches best (or --chain-map), MODELLER fills the gaps

    # 1. the assembly as explicit chains (casp17/gen_inputs, or any expanded assembly CIF/PDB)
    python3 gen_inputs/expand_template_assembly.py --in-cif 3vcd.cif --out-cif 3vcd_asm1.cif
    # 2. this script (sbatch: a 24-mer of 191 residues is ~4,600 residues per model)
    python3 build_assembly_model.py --template 3vcd_asm1.cif --target target.fasta --copies 24 \
        --out-dir modeller/ --models 5
    python3 build_assembly_model.py --template 9f8e.cif --target h2422.fasta --out-dir modeller/   # hetero A2B2
    # -> modeller/aln.ali, modeller/target.B9999000N.pdb, modeller/best.cif (+ scores.tsv, chain_map.tsv)

What it does:
  1. reads the template assembly (N chains, in file order) and the target sequence;
  2. writes the MODELLER alignment: template = its N chains, target = N copies of the target sequence, chain breaks
     '/', aligned per chain with MODELLER's align2d (structure-aware gaps);
  3. builds --models models with AutoModel, scores them with DOPE, keeps the best;
  4. writes best.cif with chains A, B, ... and _entity_poly_seq filled from the target sequence. AF3 drops a multimer
     template without _entity_poly_seq silently, and gemmi does not write it unless the entity's sequence is set.

The result carries the target's own sequences, so `combine.py --multimer-template best.cif` maps it 1:1 to the query
chains: no index-space translation, no missing residues. On CASP17 T2461o (A24, 191 aa; 3VCD 63 % identical) this
template gave AF3 median DockQ 0.88 (40 models), against 0.62 with the 3VCD assembly used directly and 0.46 without
a template.

MODELLER needs a licence key (free for academic use): a person sets KEY_MODELLER and installs it
(`KEY_MODELLER=... uv pip install modeller` into a venv, or conda's salilab channel). Steps 1, 2 (PIR) and 4 do not
need MODELLER and are what `--dry-run` exercises.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def read_fasta_all(path: Path) -> list[tuple[str, str]]:
    """[(name, sequence)] for every record (a bare sequence is one record named 'target')."""
    text = Path(path).read_text()
    if ">" not in text:
        return [("target", re.sub(r"[^A-Za-z]", "", text).upper())]
    out = []
    for name, body in re.findall(r">([^\n]*)\n([^>]*)", text):
        seq = re.sub(r"[^A-Za-z]", "", body).upper()
        if seq:
            out.append((name.split()[0] if name.split() else f"seq{len(out) + 1}", seq))
    return out


def read_fasta(path: Path) -> str:
    return read_fasta_all(path)[0][1]


def identity(a: str, b: str) -> float:
    """Identical positions over the shorter sequence, after a global alignment (gemmi)."""
    import gemmi
    al = gemmi.align_string_sequences(list(a), list(b), [])
    return al.match_count / max(1, min(len(a), len(b)))


def assign(chains: list[tuple[str, str]], records: list[tuple[str, str]], chain_map: dict[str, str] | None = None
           ) -> list[tuple[str, str, float]]:
    """For each template chain: (record name, record sequence, identity). --chain-map wins; otherwise the record the
    chain matches best. A single record goes on every chain (the homolog-assembly case)."""
    by_name = dict(records)
    out = []
    for ch, seq in chains:
        if chain_map and ch in chain_map:
            name = chain_map[ch]
            if name not in by_name:
                raise ValueError(f"--chain-map {ch}={name}: no FASTA record {name!r} (records: {', '.join(by_name)})")
        elif len(records) == 1:
            name = records[0][0]
        else:
            name = max(records, key=lambda r: identity(seq, r[1]))[0]
        out.append((name, by_name[name], identity(seq, by_name[name]) if len(records) > 1 or chain_map else float("nan")))
    return out


CHAIN_IDS = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + [str(i) for i in range(10)]


def load_template(path: Path):
    """The assembly with ligands and waters removed and chains renamed to single characters (A.., a.., 0..9): an
    expanded assembly names chains like 'A-2', which the PDB format MODELLER reads cannot hold."""
    import gemmi
    st = gemmi.read_structure(str(path))
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    if len(st[0]) > len(CHAIN_IDS):
        raise ValueError(f"{len(st[0])} chains: more than a PDB file can name ({len(CHAIN_IDS)})")
    for ch, name in zip(st[0], CHAIN_IDS):
        ch.name = name
    st.setup_entities()
    return st


def template_chains(st) -> list[tuple[str, str]]:
    """(chain name, one-letter sequence of the modelled residues) for every polymer chain, in file order."""
    import gemmi
    out = []
    for ch in st[0]:
        poly = ch.get_polymer()
        if len(poly) == 0:
            continue
        seq = gemmi.one_letter_code([r.name for r in poly])
        out.append((ch.name, seq.upper()))
    return out


def pir(tmpl_code: str, tmpl_file: str, chains: list[tuple[str, str]], target: str | list[str], copies: int | None = None) -> str:
    """An unaligned MODELLER PIR file: align2d fills in the gaps. Chain breaks are '/'.
    target: one sequence for every chain (homo-oligomer; copies must equal the chain count), or one per chain."""
    if isinstance(target, str):
        if copies is not None and copies != len(chains):
            raise ValueError(f"the template has {len(chains)} chains but --copies is {copies}: expand the right assembly")
        target = [target] * len(chains)
    if len(target) != len(chains):
        raise ValueError(f"{len(target)} target sequences for {len(chains)} template chains")
    first, last = chains[0][0], chains[-1][0]
    tseq = "/".join(s for _, s in chains)
    qseq = "/".join(target)
    return (f">P1;{tmpl_code}\nstructureX:{tmpl_file}:FIRST:{first}:LAST:{last}::::\n{tseq}*\n\n"
            f">P1;target\nsequence:target:::::::0.00: 0.00\n{qseq}*\n")


def to_af3_cif(pdb: Path, target: str | list[str], out: Path) -> Path:
    """MODELLER's PDB -> mmCIF with chains A, B, ... and each chain's target sequence as its entity's full sequence."""
    import gemmi
    st = gemmi.read_structure(str(pdb))
    st.remove_ligands_and_waters()
    seqs = [target] * len(st[0]) if isinstance(target, str) else list(target)
    if len(seqs) != len(st[0]):
        raise ValueError(f"{len(seqs)} sequences for {len(st[0])} chains in {pdb}")
    names = [chr(65 + i) if i < 26 else chr(65 + (i // 26) - 1) + chr(65 + i % 26) for i in range(len(st[0]))]
    for ch, name in zip(st[0], names):                       # mmCIF allows two-letter chains (AF3: A..Z, AA, AB, ...)
        ch.name = name
    st.setup_entities()
    order = [ch.name for ch in st[0]]
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer:
            ch = next(c for c in st[0] if c.get_polymer().subchain_id() in ent.subchains)
            ent.full_sequence = gemmi.expand_one_letter_sequence(seqs[order.index(ch.name)], gemmi.ResidueKind.AA)
            ent.polymer_type = gemmi.PolymerType.PeptideL
    st.assign_label_seq_id()
    doc = st.make_mmcif_document()
    doc.write_file(str(out))
    if "_entity_poly_seq." not in out.read_text():
        raise RuntimeError(f"{out}: no _entity_poly_seq; AF3 would drop this template")
    return out


def build(template: Path, records: list[tuple[str, str]] | str, copies: int | None, out_dir: Path, models: int, dry_run: bool,
          chain_map: dict[str, str] | None = None) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(records, str):
        records = [("target", records)]
    st = load_template(template)
    chains = template_chains(st)
    if len(records) == 1 and copies is not None and copies != len(chains):
        raise ValueError(f"the template has {len(chains)} chains but --copies is {copies}: expand the right assembly")
    plan = assign(chains, records, chain_map)
    target = [seq for _, seq, _ in plan]
    with open(out_dir / "chain_map.tsv", "w") as fh:
        fh.write("template_chain\ttemplate_residues\trecord\trecord_length\tidentity\n")
        for (ch, tseq), (name, seq, ident) in zip(chains, plan):
            fh.write(f"{ch}\t{len(tseq)}\t{name}\t{len(seq)}\t{'' if ident != ident else f'{ident:.3f}'}\n")
    tmpl_pdb = out_dir / "tmpl.pdb"
    st.write_pdb(str(tmpl_pdb))
    (out_dir / "unaligned.ali").write_text(pir("tmpl", tmpl_pdb.name, chains, target))
    gaps = sum(len(seq) for seq in target) - sum(len(t) for _, t in chains)
    print(f"template: {len(chains)} chains, {sum(len(t) for _, t in chains)} residues modelled; target "
          + ", ".join(f"{n} x{sum(1 for p in plan if p[0] == n)}" for n, _ in records if any(p[0] == n for p in plan))
          + f" ({max(gaps, 0)} residues to build) -> {out_dir / 'unaligned.ali'}, {out_dir / 'chain_map.tsv'}")
    if dry_run:
        return None
    from modeller import Alignment, Environ, Model, log           # noqa: F401  (needs the licensed package)
    from modeller.automodel import AutoModel, assess
    log.minimal()
    env = Environ()
    env.io.atom_files_directory = [str(out_dir)]
    env.io.hetatm = False
    aln = Alignment(env)
    mdl = Model(env, file=str(tmpl_pdb))
    aln.append_model(mdl, align_codes="tmpl", atom_files=str(tmpl_pdb))
    aln.append(file=str(out_dir / "unaligned.ali"), align_codes="target")
    aln.align2d(max_gap_length=50)
    aln.write(file=str(out_dir / "aln.ali"), alignment_format="PIR")
    import os
    os.chdir(out_dir)
    a = AutoModel(env, alnfile="aln.ali", knowns="tmpl", sequence="target", assess_methods=(assess.DOPE,))
    a.starting_model, a.ending_model = 1, models
    a.make()
    ok = [m for m in a.outputs if m.get("failure") is None]
    ok.sort(key=lambda m: m["DOPE score"])
    with open("scores.tsv", "w") as fh:
        fh.write("model\tDOPE\tmolpdf\n" + "".join(f"{m['name']}\t{m['DOPE score']:.1f}\t{m['molpdf']:.1f}\n" for m in ok))
    best = out_dir / ok[0]["name"]
    cif = to_af3_cif(best, target, out_dir / "best.cif")
    print(f"best model {best.name} (DOPE {ok[0]['DOPE score']:.1f}) -> {cif}")
    return cif


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template", required=True, type=Path, help="the homolog assembly with every chain explicit")
    ap.add_argument("--target", required=True, type=Path,
                    help="FASTA: one record (every chain gets it) or one per distinct chain (each template chain gets its best match)")
    ap.add_argument("--copies", type=int, help="homo-oligomer check: the template must have this many chains")
    ap.add_argument("--chain-map", help="template chain = FASTA record, e.g. A=heavy,B=light (overrides the best match)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="write the alignment input only (no MODELLER needed)")
    a = ap.parse_args(argv)
    cmap = dict(x.split("=", 1) for x in a.chain_map.split(",")) if a.chain_map else None
    build(a.template, read_fasta_all(a.target), a.copies, a.out_dir, a.models, a.dry_run, cmap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
