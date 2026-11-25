"""`residudes-engine msa`, `residudes-engine inputs` and `residudes-engine prepare` — from a FASTA to the
AF3-dialect input.json every model runs from."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import msa as msa_mod
from .config import SiteError, load_site, venv_python
from .inputs import (Assembly, InputError, TemplateTools, find_result_model, multimer_template,
                     parse_contact_spec, parse_glycan_spec, parse_ligand_spec, parse_multimer_template_spec,
                     resolve_pdb_source, template_from_structure, templates_from_m8, write_input)
from .msa import Chain, MsaError, chains_from_fasta, read_msa_dir, write_msa_dir


def _log(msg: str) -> None:
    print(msg, flush=True)


def _default_out(fasta: Path, name: str | None) -> Path:
    return Path("engine_out") / f"{name or fasta.stem}_inputs"


def _msa_settings(site, ns) -> dict:
    """Site `[msa]` table with environment and flag overrides."""
    m = dict(site.raw.get("msa", {}) if site else {})
    env = os.environ
    return {
        "plmmsa_api": ns.api or env.get("RESIDUDES_ENGINE_PLMMSA_API") or m.get("plmmsa_api") or msa_mod.PLMMSA_API,
        "plmmsa_token": ns.token or env.get("PLMMSA_TOKEN") or m.get("plmmsa_token"),
        "colabfold_api": ns.api or env.get("RESIDUDES_ENGINE_COLABFOLD_API") or m.get("colabfold_api") or msa_mod.COLABFOLD_API,
        "mmseqs_bin": ns.mmseqs or env.get("RESIDUDES_ENGINE_MMSEQS") or m.get("mmseqs_bin") or "mmseqs",
        "mmseqs_db": ns.db or env.get("RESIDUDES_ENGINE_MMSEQS_DB") or m.get("mmseqs_db"),
        # AlphaFold 3's own pipeline: the genetic databases, and optionally a container to run it in
        "af3_db_dir": getattr(ns, "af3_db", None) or env.get("RESIDUDES_ENGINE_AF3_DB_DIR") or m.get("af3_db_dir"),
        "af3_container": env.get("RESIDUDES_ENGINE_AF3_CONTAINER") or m.get("af3_container"),
        "af3_script": m.get("af3_script"),
        "af3_binaries": m.get("af3_binaries") or {},
        "af3_max_template_date": getattr(ns, "max_template_date", None) or m.get("af3_max_template_date"),
    }


def _af3(chains: list[Chain], out_dir: Path, ns, site, s: dict, log):
    """AlphaFold 3's data pipeline, from the site's AF3 entry point or a container."""
    from .config import SiteError, venv_python
    if not s["af3_db_dir"]:
        raise MsaError("the af3 source needs the AlphaFold 3 genetic databases: --af3-db DIR, "
                       "[msa] af3_db_dir in the site file, or $RESIDUDES_ENGINE_AF3_DB_DIR")
    container = s["af3_container"]
    script, python = s["af3_script"], None
    if not container:
        if site is None:
            raise MsaError("the af3 source needs a site (for the AF3 checkout and its venv), or a container")
        script = script or (site.core / "models" / "alphafold3" / "run_alphafold.py")
        if not Path(script).is_file():
            raise MsaError(f"AlphaFold 3 entry point missing: {script}")
        try:
            python = venv_python(site.jax_venv_for("af3"))
        except SiteError as e:
            raise MsaError(f"{e} — the data pipeline runs in the AF3 environment") from None
    return msa_mod.af3_search(
        chains, out_dir=out_dir / "msa", script=script, python=python, db_dir=s["af3_db_dir"],
        binaries=s["af3_binaries"], container=container, max_template_date=s["af3_max_template_date"],
        n_cpu=ns.threads, timeout=max(ns.timeout, 86400.0), log=log)


def run_msa(chains: list[Chain], out_dir: Path, ns, site) -> Path:
    s = _msa_settings(site, ns)
    sources = [x.strip() for x in ns.source.split(",") if x.strip()]
    for src in sources:
        if src not in msa_mod.SOURCES:
            raise MsaError(f"unknown MSA source {src!r} (choose from {', '.join(msa_mod.SOURCES)})")
    results = []
    no_templates = getattr(ns, "no_templates", False)     # only the inputs/prepare parsers carry it
    for src in sources:
        _log(f"▶ MSA search: {src}")
        if src == "plmmsa":
            results.append(msa_mod.plmmsa_search(
                chains, api=s["plmmsa_api"], token=s["plmmsa_token"], aligner=ns.aligner, mode=ns.mode,
                score_model=ns.score_model, poll=ns.poll, timeout=ns.timeout, log=_log))
        elif src == "colabfold":
            results.append(msa_mod.colabfold_search(
                chains, api=s["colabfold_api"], use_env=not ns.no_env, use_filter=not ns.no_filter,
                use_templates=not no_templates, pair=not ns.no_pair, pairing=ns.pairing,
                poll=ns.poll, timeout=ns.timeout, workdir=out_dir / "msa" / "_colabfold", log=_log))
        elif src == "af3":
            results.append(_af3(chains, out_dir, ns, site, s, _log))
        else:
            if not s["mmseqs_db"]:
                raise MsaError("local mmseqs2 needs --db (or [msa] mmseqs_db in the site file / RESIDUDES_ENGINE_MMSEQS_DB)")
            results.append(msa_mod.mmseqs2_search(
                chains, out_dir=out_dir / "msa", db_dir=Path(s["mmseqs_db"]), mmseqs=s["mmseqs_bin"], gpu=ns.gpu,
                use_env=not ns.no_env, use_templates=not no_templates, threads=ns.threads, log=_log))
    # several sources → one merged unpaired A3M per chain; paired/templates from the first source that has them
    merged = msa_mod.MsaResult("+".join(sources), meta={r.source: r.meta for r in results})
    for ch in chains:
        texts = [r.unpaired[ch.letter] for r in results if ch.letter in r.unpaired]
        if texts:
            merged.unpaired[ch.letter] = msa_mod.merge_a3ms(texts, ch.seq, ch.name) if len(texts) > 1 else texts[0]
        for r in results:
            if ch.letter in r.paired and ch.letter not in merged.paired:
                merged.paired[ch.letter] = r.paired[ch.letter]
            if ch.letter in r.templates_m8 and ch.letter not in merged.templates_m8:
                merged.templates_m8[ch.letter] = r.templates_m8[ch.letter]
    d = write_msa_dir(merged, chains, out_dir)
    for ch in chains:
        depth = msa_mod.a3m_depth(merged.unpaired.get(ch.letter, "")) - 1
        _log(f"  {ch.letter} {ch.name}: {depth} sequences" + (" + paired" if ch.letter in merged.paired else "")
             + (" + template hits" if ch.letter in merged.templates_m8 else ""))
    _log(f"✓ MSAs in {d}")
    return d


def _tools(site) -> TemplateTools:
    if site is None:
        raise InputError("template handling needs a site (runtime venv with gemmi); run `residudes-engine setup site`")
    return TemplateTools(python=venv_python(site.venv), script=site.core / "scripts" / "template_tools.py",
                         cache_dir=site.cache_dir)


def run_inputs(chains: list[Chain], out_dir: Path, msa_dir: Path | None, ns, site) -> Path:
    name = ns.name or ns.fasta.stem
    asm = Assembly(name=name, chains=chains, seeds=[int(s) for s in ns.seeds.split(",")] if ns.seeds else [1])
    provenance: dict = {"fasta": str(ns.fasta.resolve()), "stoichiometry": ns.sto, "msa": None, "templates": {}}
    pairing_mode = None
    m8: dict[str, Path] = {}
    if msa_dir and (msa_dir / "msa.json").is_file():
        man, unp, par, m8 = read_msa_dir(msa_dir)
        asm.unpaired, asm.paired = unp, par
        provenance["msa"] = {"dir": str(msa_dir.resolve()), "source": man.get("source")}
        if par:
            pairing_mode = "precomputed"       # server-paired rows are row-aligned across chains
    elif not ns.no_msa:
        _log("note: no MSA directory — single-sequence input (pass --no-msa to silence)")

    tools = None
    want_templates = not ns.no_templates and (m8 or ns.pin_template or ns.self_template or ns.multimer_template
                                              or ns.templates_from)
    if want_templates:
        tools = _tools(site)
    tpl_dir = out_dir
    cleared = {L.strip() for s in (ns.no_per_chain_template or []) for L in s.split(",")}
    # 1. per-chain templates from search hits (m8)
    if m8 and not ns.no_templates and not ns.no_search_templates:
        for ch in chains:
            if ch.letter in m8 and ch.letter not in cleared:
                asm.templates[ch.letter] = templates_from_m8(
                    tools, ch, m8[ch.letter], tpl_dir, top=ns.top_templates,
                    mmcif_dir=Path(ns.mmcif_dir) if ns.mmcif_dir else None, max_date=ns.max_template_date, log=_log)
                provenance["templates"][ch.letter] = {"type": "search", "n": len(asm.templates[ch.letter])}
    # 2. explicit template lists (AF3 templates.json: [{mmcifPath|mmcif, queryIndices, templateIndices}])
    for spec in ns.templates_from or []:
        letter, _, path = spec.partition("=")
        entries = json.loads(Path(path).read_text())
        asm.templates[letter] = entries if isinstance(entries, list) else entries.get("templates", [])
        provenance["templates"][letter] = {"type": "file", "path": str(Path(path).resolve())}
    # 3. pinned PDB chains: X=<pdb id|cif>:<auth chain>[@rank]
    for spec in ns.pin_template or []:
        letter, _, rest = spec.partition("=")
        src, _, auth = rest.partition(":")
        rank = 0
        if "@" in auth:
            auth, r = auth.split("@", 1)
            rank = int(r)
        ch = _chain(chains, letter)
        cif = resolve_pdb_source(src, tools)
        entry = template_from_structure(tools, ch, cif, auth, tpl_dir, tag=f"pin_{Path(src).stem}_{auth}")
        lst = asm.templates.setdefault(letter, [])
        lst.insert(min(rank, len(lst)), entry)
        del lst[4:]                                  # AF3 keeps at most four templates per chain
        provenance["templates"].setdefault(letter, {})["pinned"] = f"{src}:{auth}"
        _log(f"  pinned template {letter} ← {src}:{auth} (rank {rank})")
    # 4. self-templates from a previous prediction: X=<seed dir|model.cif>[:<chain in that model>]
    for spec in ns.self_template or []:
        letter, _, rest = spec.partition("=")
        src, _, auth = rest.partition(":")
        ch = _chain(chains, letter)
        model = find_result_model(Path(src))
        entry = template_from_structure(tools, ch, model, auth or asm.output_ids()[letter][0], tpl_dir,
                                        tag=f"self_{letter}")
        asm.templates[letter] = [entry] + asm.templates.get(letter, [])[:3]
        provenance["templates"].setdefault(letter, {})["self"] = str(model.resolve())
        _log(f"  self-template {letter} ← {model}")
    # 5. multi-chain (interface) templates: CIF:Q1=T1,Q2=T2
    for i, spec in enumerate(ns.multimer_template or []):
        cif, pairs = parse_multimer_template_spec(spec)
        cif = resolve_pdb_source(str(cif), tools)
        asm.multimer_templates.append(multimer_template(tools, asm, cif, pairs, tpl_dir, tag=f"multimer_{i}_{cif.stem}"))
        provenance["templates"][f"multimer_{i}"] = {"cif": str(cif.resolve()), "map": pairs}
        _log(f"  multimer template {cif.name}: " + ", ".join(f"{q}←{t}" for q, t in pairs))
    for letters in ns.no_per_chain_template or []:
        for L in letters.split(","):
            asm.templates[L.strip()] = []
    # ligands, glycans, contacts
    asm.ligands = [parse_ligand_spec(s) for s in (ns.ligand or [])]
    asm.glycans = [parse_glycan_spec(s) for s in (ns.glycan or [])]
    asm.contacts = [parse_contact_spec(s) for s in (ns.contact or [])]

    doc = asm.build(out_dir)
    provenance.update(ligands=[l["name"] for l in asm.ligands], glycans=ns.glycan or [], contacts=ns.contact or [])
    p = write_input(doc, out_dir, pairing_mode=pairing_mode, provenance=provenance)
    n_tpl = sum(len(v) for v in asm.templates.values())
    _log(f"✓ {p}  ({len(doc['sequences'])} entities, {n_tpl} templates"
         + (f", {len(asm.multimer_templates)} multimer template(s)" if asm.multimer_templates else "")
         + (f", {len(doc.get('bondedAtomPairs', []))} bonds" if doc.get("bondedAtomPairs") else "")
         + (f", {len(doc['constraint']['contact'])} contacts" if doc.get("constraint") else "") + ")")
    _log(f"  next: bin/residudes-engine predict --model pt2 --input {p}" + (" --pairing-mode precomputed" if pairing_mode else ""))
    return p


def _chain(chains: list[Chain], letter: str) -> Chain:
    for c in chains:
        if c.letter == letter:
            return c
    raise InputError(f"chain {letter!r} is not a source chain (have {[c.letter for c in chains]})")


def _site_or_none(ns):
    try:
        return load_site(ns.site)
    except SiteError:
        return None


def cmd_msa(ns: argparse.Namespace) -> int:
    chains = chains_from_fasta(ns.fasta, ns.sto)
    out_dir = ns.out or _default_out(ns.fasta, ns.name)
    try:
        run_msa(chains, out_dir, ns, _site_or_none(ns))
    except MsaError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    return 0


def cmd_inputs(ns: argparse.Namespace) -> int:
    chains = chains_from_fasta(ns.fasta, ns.sto)
    out_dir = ns.out or _default_out(ns.fasta, ns.name)
    msa_dir = ns.msa_dir or (out_dir / "msa")
    try:
        run_inputs(chains, out_dir, msa_dir if msa_dir.is_dir() else None, ns, _site_or_none(ns))
    except (InputError, MsaError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    return 0


def cmd_prepare(ns: argparse.Namespace) -> int:
    chains = chains_from_fasta(ns.fasta, ns.sto)
    out_dir = ns.out or _default_out(ns.fasta, ns.name)
    site = _site_or_none(ns)
    try:
        msa_dir = None if ns.no_msa else run_msa(chains, out_dir, ns, site)
        run_inputs(chains, out_dir, msa_dir, ns, site)
    except (InputError, MsaError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    return 0


# ── argparse ──────────────────────────────────────────────────────────────

def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--fasta", type=Path, required=True, help="query FASTA: one record per distinct chain (A, B, … in order)")
    sp.add_argument("--sto", default=None, help="stoichiometry, e.g. A2B1 (default: one copy of each record)")
    sp.add_argument("--name", default=None, help="input name (default: FASTA stem)")
    sp.add_argument("--out", type=Path, default=None, help="output directory (default engine_out/<name>_inputs)")
    sp.add_argument("--site", default=None)


def _msa_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("MSA search")
    g.add_argument("--source", default="plmmsa", help="plmmsa | colabfold | mmseqs2 | af3, or a comma list to merge (default plmmsa)")
    g.add_argument("--api", default=None, help="API base URL for plmmsa/colabfold (site [msa], RESIDUDES_ENGINE_PLMMSA_API / RESIDUDES_ENGINE_COLABFOLD_API)")
    g.add_argument("--token", default=None, help="bearer token for the plmmsa API ($PLMMSA_TOKEN)")
    g.add_argument("--aligner", default="otalign", choices=msa_mod.PLMMSA_ALIGNERS, help="plmmsa aligner (default otalign)")
    g.add_argument("--mode", default="local", help="plmmsa alignment mode: local | global | glocal | q2t | t2q")
    g.add_argument("--score-model", default=None, help="plmmsa score model (default per aligner)")
    g.add_argument("--no-env", action="store_true", help="colabfold/mmseqs2: skip the environmental database")
    g.add_argument("--no-filter", action="store_true", help="colabfold: no diversity filter")
    g.add_argument("--no-pair", action="store_true", help="colabfold: no paired MSA for complexes")
    g.add_argument("--pairing", default="greedy", choices=("greedy", "complete"), help="colabfold pairing strategy")
    g.add_argument("--db", default=None, help="mmseqs2: database directory ([msa] mmseqs_db)")
    g.add_argument("--mmseqs", default=None, help="mmseqs2: binary or wrapper command ([msa] mmseqs_bin)")
    g.add_argument("--af3-db", default=None, help="af3: the AlphaFold 3 genetic databases ([msa] af3_db_dir)")
    g.add_argument("--gpu", action="store_true", help="mmseqs2: GPU prefilter (MMseqs2-GPU build)")
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--poll", type=float, default=10.0, help="API poll interval, s")
    g.add_argument("--timeout", type=float, default=3600.0, help="per-job wall cap, s")


def _input_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("input assembly")
    g.add_argument("--msa-dir", type=Path, default=None, help="MSA directory from `residudes-engine msa` (default <out>/msa)")
    g.add_argument("--no-msa", action="store_true", help="single-sequence input")
    g.add_argument("--seeds", default=None, help="modelSeeds, e.g. 1,2,3 (default 1)")
    t = sp.add_argument_group("templates")
    t.add_argument("--no-templates", action="store_true", help="no templates at all")
    t.add_argument("--no-search-templates", action="store_true", help="ignore template hits from the MSA search")
    t.add_argument("--top-templates", type=int, default=4, help="templates per chain from search hits (default 4)")
    t.add_argument("--max-template-date", default=None, help="drop search templates deposited after YYYY-MM-DD")
    t.add_argument("--mmcif-dir", default=None, help="local mmCIF mirror (<pdbid>.cif); otherwise fetched from RCSB into the cache")
    t.add_argument("--templates-from", action="append", metavar="X=templates.json", help="use an AF3 templates list for chain X")
    t.add_argument("--pin-template", action="append", metavar="X=<pdb|cif>:<chain>[@rank]",
                   help="force a PDB chain into chain X's templates (e.g. D=7mem:E)")
    t.add_argument("--self-template", action="append", metavar="X=<seed_dir|model.cif>[:<chain>]",
                   help="use a previous prediction of chain X as its template (stepwise assembly)")
    t.add_argument("--multimer-template", action="append", metavar="CIF:Q1=T1,Q2=T2",
                   help="interface template across output chains Q1,Q2… from template chains T1,T2… (PDB id or file)")
    t.add_argument("--no-per-chain-template", action="append", metavar="A,B",
                   help="clear per-chain templates of these chains (FASTA letters), e.g. when a multimer template covers them")
    c = sp.add_argument_group("entities and constraints")
    c.add_argument("--ligand", action="append", metavar="NAME[*N]=smi:<SMILES>|ccd:<CODE>[,…]|smi-file:<path>")
    c.add_argument("--glycan", action="append", metavar="A29:NAG-NAG-BMA[@ND2/O4]",
                   help="covalent glycan on chain A residue 29 (Asn ND2 / Ser OG / Thr OG1 by default)")
    c.add_argument("--contact", action="append", metavar="A:104=B:166[@max=6,min=0]",
                   help="Protenix contact constraint between source chains (replicated per protomer)")


def add_parsers(sub) -> None:
    sp = sub.add_parser("msa", help="MSA search: DeepFold-PLM API, ColabFold/MMseqs2 API, or local MMseqs2")
    _common(sp); _msa_opts(sp)
    sp.set_defaults(fn=cmd_msa)
    sp = sub.add_parser("inputs", help="assemble the AF3-dialect input.json (MSAs, templates, ligands, glycans, constraints)")
    _common(sp); _input_opts(sp)
    sp.set_defaults(fn=cmd_inputs)
    sp = sub.add_parser("prepare", help="msa + inputs in one go")
    _common(sp); _msa_opts(sp); _input_opts(sp)
    sp.set_defaults(fn=cmd_prepare)
