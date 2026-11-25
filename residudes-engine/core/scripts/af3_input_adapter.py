#!/usr/bin/env python3
"""AF3 input.json → pt2 / of3 / bt2 native input converter.

Lets the four runners share a single canonical input file (AF3 dialect)
instead of maintaining four parallel `gen_inputs/<model>/` outputs.

Templates conversion: AF3 carries `(queryIndices, templateIndices, mmcif/mmcifPath)`
per template hit. pt2/of3 expect a hmmsearch-style a3m alignment file
(one record per hit). The adapter regenerates a3m on the fly from the cif
+ indices so no extra preprocessing artefact is needed.

bt2 takes cif paths directly (yaml `templates: [{cif, chain_id}]`),
which AF3's `mmcifPath` already provides.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional


# ── Template alignment regeneration (AF3 indices → hmmsearch a3m) ─────────

_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _extract_chain_seq_from_cif(cif_text: str) -> tuple[str, str, str]:
    """Return (pdb_id, chain_id, polymer_one_letter_seq) for a single-chain
    AF3 template cif.

    Tries `_entity_poly.pdbx_seq_one_letter_code_can` first; falls back to
    walking the `_entity_poly_seq` loop and 3→1-letter conversion (gemmi
    single-chain cifs from `af3st.filter().to_mmcif()` carry only the loop).
    """
    lines = cif_text.splitlines()

    # PDB id from data_<ID> line
    pdb_id = ""
    for ln in lines[:5]:
        s = ln.strip()
        if s.startswith("data_"):
            pdb_id = s[5:].lower()
            break

    chain = ""
    seq_one_letter = ""
    capture_seq = False
    seq_lines: list[str] = []

    # First pass: header values + optional one-letter-code-can.
    for ln in lines:
        s = ln.strip()
        if s.startswith("_entity_poly.pdbx_strand_id"):
            parts = s.split()
            if len(parts) > 1:
                chain = parts[1].split(",")[0]
        elif s.startswith("_entity_poly.pdbx_seq_one_letter_code_can"):
            parts = s.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                seq_one_letter = parts[1].strip().strip("'\";")
            else:
                capture_seq = True
        elif capture_seq:
            if s.startswith(";"):
                if seq_lines:
                    capture_seq = False
                else:
                    body = s.lstrip(";").rstrip()
                    if body:
                        seq_lines.append(body)
            else:
                seq_lines.append(s)
    if not seq_one_letter and seq_lines:
        seq_one_letter = "".join(seq_lines).replace(" ", "")

    # Fallback: parse _entity_poly_seq loop (3-letter codes).
    if not seq_one_letter:
        i = 0
        n = len(lines)
        while i < n:
            s = lines[i].strip()
            if s == "loop_":
                j = i + 1
                local_cols: list[str] = []
                while j < n and lines[j].lstrip().startswith("_"):
                    local_cols.append(lines[j].strip())
                    j += 1
                is_eps = any(c.startswith("_entity_poly_seq.") for c in local_cols)
                if is_eps:
                    cols = [c.split(".", 1)[1] for c in local_cols]
                    rows: list[list[str]] = []
                    while j < n:
                        s2 = lines[j].strip()
                        if (not s2) or s2.startswith("#") or s2.startswith("loop_") \
                                or s2.startswith("data_") or s2.startswith("_"):
                            break
                        rows.append(s2.split())
                        j += 1
                    if rows:
                        try:
                            ci = cols.index("mon_id")
                            seq_one_letter = "".join(
                                _AA_3TO1.get(r[ci], "X") for r in rows
                            )
                        except (ValueError, IndexError):
                            pass
                    i = j
                    continue
                i = j
            else:
                i += 1

    return pdb_id, chain, seq_one_letter


def _extract_cif_polymer_chain_order(cif_text: str) -> list[str]:
    """Walk a cif's _atom_site loop and return polymer auth_asym_id values
    in first-appearance order (one per polymer chain). Used to map an AF3
    multimerTemplates entry's query chain ids to the cif's actual chain
    labels — combine.py emits the multimer cif with chains re-ordered to
    match the queryIndices key sequence but keeps the source PDB's auth
    chain labels (e.g. 8YIV → A/B/C, not the query's C/D/E)."""
    lines = cif_text.splitlines()
    # Locate the _atom_site loop. _atom_site columns start with "_atom_site."
    # and the data rows follow. We need the auth_asym_id column index.
    cols: list[str] = []
    in_loop = False
    auth_idx = -1
    seen: list[str] = []
    seen_set: set[str] = set()
    for ln in lines:
        s = ln.strip()
        if not in_loop:
            if s == "loop_":
                cols = []
                in_loop = True
            continue
        if s.startswith("_atom_site."):
            cols.append(s[len("_atom_site."):])
            continue
        if not cols:
            in_loop = False
            continue
        # First non-column line of an atom_site loop.
        if auth_idx < 0:
            try:
                auth_idx = cols.index("auth_asym_id")
            except ValueError:
                # No auth_asym_id; fall back to label_asym_id.
                try:
                    auth_idx = cols.index("label_asym_id")
                except ValueError:
                    return []
        if not s or s.startswith(("#", "loop_", "data_", "_")):
            break
        toks = s.split()
        if len(toks) <= auth_idx:
            continue
        # Only count ATOM rows (skip HETATM ligands).
        if toks[0] != "ATOM":
            continue
        cid = toks[auth_idx]
        if cid not in seen_set:
            seen_set.add(cid)
            seen.append(cid)
    return seen


def _af3_template_to_a3m_record(query_seq: str,
                                template: dict,
                                cif_text: Optional[str] = None) -> Optional[str]:
    """Turn one AF3 template entry into a hmmsearch-style a3m record.

    Returns the record (header + alignment line) or None if regen fails.
    """
    qi = list(template.get("queryIndices") or [])
    ti = list(template.get("templateIndices") or [])
    if not qi or not ti or len(qi) != len(ti):
        return None

    if cif_text is None:
        # mmcif inline or path: caller resolves
        if "mmcif" in template:
            cif_text = template["mmcif"]
        elif "mmcifPath" in template:
            cif_text = Path(template["mmcifPath"]).read_text()
        else:
            return None

    pdb_id, chain, tpl_seq = _extract_chain_seq_from_cif(cif_text)
    if not tpl_seq or not pdb_id:
        return None

    n_q = len(query_seq)
    aln = ["-"] * n_q
    valid_qis = []
    valid_tis = []
    for q, t in zip(qi, ti):
        if 0 <= q < n_q and 0 <= t < len(tpl_seq):
            aln[q] = tpl_seq[t]
            valid_qis.append(q)
            valid_tis.append(t)
    if not valid_qis:
        return None
    # `qs/qe` in the hmmsearch header are *query* start/end (1-based);
    # `template_range:N-M` is template start/end so downstream parsers
    # can recover both coordinate systems without reading the cif.
    qs = min(valid_qis) + 1
    qe = max(valid_qis) + 1
    ts = min(valid_tis) + 1 if valid_tis else 0
    te = max(valid_tis) + 1 if valid_tis else 0
    aln_str = "".join(aln)
    n_tpl = len(tpl_seq)
    header = (
        f">{pdb_id.lower()}_{chain or 'A'}/{qs}-{qe} "
        f"E=0 mol:protein protein length:{n_tpl} "
        f"template_range:{ts}-{te}"
    )
    return f"{header}\n{aln_str}\n"


def af3_templates_to_hmmsearch_a3m(query_seq: str, templates: list[dict]) -> str:
    """Build a complete hmmsearch a3m string from an AF3 templates list.

    First record is the query (so HmmsearchA3MParser.skip_first works); each
    subsequent record is one template hit's alignment vs the query.
    """
    out = [f">query\n{query_seq}\n"]
    for t in templates:
        rec = _af3_template_to_a3m_record(query_seq, t)
        if rec:
            out.append(rec)
    return "".join(out)


# ── Per-model converters ─────────────────────────────────────────────────

def _af3_chain_entity_map(af3: dict) -> dict:
    """Map each AF3 output chain id → (entity_idx_1based, copy_idx_1based).

    Entity index = the 1-based order in which protein/ligand blocks are
    emitted by the converters (they all walk `af3["sequences"]` in order and
    append one block per AF3 entry), so this map is valid for pt2's
    `covalent_bonds` entity numbering and for resolving chain ids generally.
    A homomer block (`id:[A,B]`) contributes copy 1→A, copy 2→B under one
    entity index; a scalar id is copy 1.
    """
    m: dict = {}
    eidx = 0
    for blk in af3.get("sequences", []):
        body = (blk.get("protein") or blk.get("ligand")
                or blk.get("rna") or blk.get("dna"))
        if body is None:
            continue
        eidx += 1
        ids = body["id"] if isinstance(body["id"], list) else [body["id"]]
        for copy, cid in enumerate(ids, start=1):
            m[cid] = (eidx, copy)
    return m


def _af3_bonds(af3: dict):
    """Yield validated ((c1,r1,a1),(c2,r2,a2)) tuples from bondedAtomPairs.

    AF3 atom pointer = [chain_id, residue_id (1-based), atom_name]. Skips
    malformed entries (so a bad bond never aborts a whole conversion).
    """
    for bond in (af3.get("bondedAtomPairs") or []):
        try:
            (c1, r1, a1), (c2, r2, a2) = bond
        except (ValueError, TypeError):
            continue
        yield (str(c1), int(r1), str(a1)), (str(c2), int(r2), str(a2))


def _af3_modifications(prot: dict) -> list[tuple[int, str]]:
    """AF3 protein modifications as (1-based position, CCD code), the one reading all three converters share.

    Every native format takes them, 1-indexed: Protenix as `{"ptmPosition", "ptmType": "CCD_<code>"}`, OpenFold3 as
    `non_canonical_residues: {position: code}`, Boltz-2 as `modifications: [{position, ccd}]`. None of the three
    converters passed them on, so a phosphoserine in the canonical input reached AF3 and nobody else - H1388's
    Protenix, Boltz-2 and OpenFold3 "phosphorylated" runs wrote SER at the site, and only the agent's own results
    review noticed."""
    out = []
    for m in prot.get("modifications") or []:
        code, pos = m.get("ptmType"), m.get("ptmPosition")
        if not code or pos is None:
            continue
        code = code[4:] if code.startswith("CCD_") else code
        out.append((int(pos), code))
    return out


def af3_to_pt2(af3: dict, sidecar_dir: Optional[Path] = None) -> list:
    """Convert AF3 input.json to Protenix native (list-of-jobs) format.

    Templates: regenerated as a single hmmsearch a3m and written to
    `sidecar_dir / "templates_<chain>_pt2.a3m"`. pt2 reads it via
    `templatesPath`. If sidecar_dir is None, writes alongside the AF3
    input file's nominal location (caller responsibility).

    Ligand entities (AF3 `{"ligand":{"id","smiles"|"ccdCodes"}}`) become
    Protenix `{"ligand":{"ligand":"<SMI>" | "CCD_<CODE>", "count":N}}`.
    Multi-CCD components (`ccdCodes:["A","B"]`) are joined with `_`: Protenix's
    json_parser splits `CCD_<A>_<B>` on `_` (a glycan is CCD_NAG_NAG_BMA).
    """
    job = {
        "name": af3.get("name"),
        "modelSeeds": af3.get("modelSeeds", [1]),
        "sequences": [],
    }
    for sidx, seq_block in enumerate(af3.get("sequences", [])):
        prot = seq_block.get("protein")
        lig = seq_block.get("ligand")
        if prot is not None:
            cid = prot["id"] if isinstance(prot["id"], list) else [prot["id"]]
            sequence = prot["sequence"]

            proteinChain = {
                "sequence": sequence,
                "count": len(cid),
                "id": cid,
            }
            if prot.get("unpairedMsaPath"):
                proteinChain["unpairedMsaPath"] = prot["unpairedMsaPath"]
            if prot.get("pairedMsaPath"):
                proteinChain["pairedMsaPath"] = prot["pairedMsaPath"]
            if mods := _af3_modifications(prot):
                proteinChain["modifications"] = [{"ptmPosition": p, "ptmType": f"CCD_{c}"} for p, c in mods]

            # Templates → hmmsearch a3m sidecar.
            tpls = prot.get("templates") or []
            if tpls and sidecar_dir is not None:
                a3m_str = af3_templates_to_hmmsearch_a3m(sequence, tpls)
                chain_tag = cid[0] if cid else f"c{sidx}"
                a3m_path = sidecar_dir / f"templates_{chain_tag}_pt2.a3m"
                a3m_path.parent.mkdir(parents=True, exist_ok=True)
                a3m_path.write_text(a3m_str)
                proteinChain["templatesPath"] = str(a3m_path.resolve())

            job["sequences"].append({"proteinChain": proteinChain})
        elif lig is not None:
            cid = lig["id"] if isinstance(lig["id"], list) else [lig["id"]]
            ligand_body: dict = {"count": len(cid)}
            if lig.get("smiles"):
                ligand_body["ligand"] = lig["smiles"]
            elif lig.get("ccdCodes"):
                codes = lig["ccdCodes"]
                if isinstance(codes, str):
                    codes = [codes]
                ligand_body["ligand"] = "CCD_" + "_".join(codes)
            else:
                raise ValueError(
                    f"af3_to_pt2: ligand at sequences[{sidx}] has neither "
                    f"`smiles` nor `ccdCodes`"
                )
            job["sequences"].append({"ligand": ligand_body})
        elif seq_block.get("rna") is not None or seq_block.get("dna") is not None:
            is_rna = seq_block.get("rna") is not None
            na = seq_block.get("rna") if is_rna else seq_block.get("dna")
            cid = na["id"] if isinstance(na["id"], list) else [na["id"]]
            na_chain = {
                "sequence": na["sequence"],
                "count": len(cid),
                "id": cid,
            }
            # Nucleic-acid MSA is optional. A single-sequence inline
            # `unpairedMsa` (query only) is equivalent to no MSA, so we skip
            # it; a real precomputed path is carried through untouched.
            if na.get("unpairedMsaPath"):
                na_chain["unpairedMsaPath"] = na["unpairedMsaPath"]
            if na.get("pairedMsaPath"):
                na_chain["pairedMsaPath"] = na["pairedMsaPath"]
            key = "rnaSequence" if is_rna else "dnaSequence"
            job["sequences"].append({key: na_chain})
        else:
            continue

    # AF3 multimerTemplates → top-level passthrough on the protenix job.
    # Protenix native JSON has no native concept of multi-chain templates;
    # we pass the AF3 structure verbatim and let
    # InferenceTemplateFeaturizer.make_template_feature read it from the
    # job dict (Phase 2 pipeline change). queryIndices is the dict-of-
    # chain-to-residue-indices form (gen_inputs/combine.py output);
    # mmcifPath is absolute on the writer side. is_cross_chain is
    # implicit per entry — every entry in this list represents a single
    # template that spans len(queryIndices) > 1 chains.
    mt_pt2 = []
    for mt in (af3.get("multimerTemplates") or []):
        cif = mt.get("mmcifPath")
        qi = mt.get("queryIndices")
        ti = mt.get("templateIndices")
        if cif and isinstance(qi, dict) and qi:
            mt_pt2.append({
                "mmcifPath": cif,
                "queryIndices": qi,
                "templateIndices": ti,
            })
    if mt_pt2:
        job["multimerTemplates"] = mt_pt2

    # AF3 bondedAtomPairs → Protenix `covalent_bonds`. Protenix references
    # bonded atoms by (entity, copy, position, atom-name): entity = 1-based
    # order in `sequences`, position = residue index for polymers OR the CCD
    # serial for multi-CCD ligands (both 1-based — same convention AF3 uses),
    # atom = atom name. Glycans (multi-CCD ligand + protein↔sugar bond) are an
    # explicitly documented use case (infer_json_format.md §4-5).
    bonds_pt2 = []
    if af3.get("bondedAtomPairs"):
        cmap = _af3_chain_entity_map(af3)
        for (c1, r1, a1), (c2, r2, a2) in _af3_bonds(af3):
            if c1 not in cmap or c2 not in cmap:
                continue
            e1, k1 = cmap[c1]
            e2, k2 = cmap[c2]
            bonds_pt2.append({
                "entity1": str(e1), "copy1": k1, "position1": str(r1), "atom1": a1,
                "entity2": str(e2), "copy2": k2, "position2": str(r2), "atom2": a2,
            })
    if bonds_pt2:
        job["covalent_bonds"] = bonds_pt2

    # Pass-through `constraint` field (Protenix-native contact/pocket
    # constraints). gen_inputs/combine.py emits this when --contact is used;
    # AF3 / OF3 / Boltz ignore the same field, so the AF3 input.json drives
    # all four runners without divergence.
    if "constraint" in af3 and af3["constraint"]:
        job["constraint"] = af3["constraint"]
    return [job]


def af3_to_of3(af3: dict, sidecar_dir: Optional[Path] = None,
               pairing_mode: str = None) -> dict:
    """Convert to OF3 native (queries dict) format.

    OF3's MSA parser filters by basename against `max_seq_counts` keys
    (colabfold_main, uniref90_hits, ...) — files with arbitrary names like
    `unpaired_A.a3m` are silently dropped.  When a sidecar_dir is given,
    we copy each chain's MSA files there with conforming names so OF3
    actually reads them.

    pairing_mode (REQUIRED, no default):
        "precomputed": stage paired_X.a3m as `colabfold_paired.a3m` in
            `paired_msa_file_paths` → OF3 takes the
            `create_paired_from_precomputed` branch (row-index aligned
            stack, no species inference). Use this when the upstream
            paired MSA is already row-aligned across chains (ColabFold
            output, mmseqs2_gpu pipeline).
        "online": stage paired_X.a3m as `uniprot_hits.a3m` in
            `main_msa_file_paths` → OF3 takes the `create_paired`
            branch (find_pairing_indices with species-intersect
            drain). Use this when the paired MSA carries UniProt
            TrEMBL headers with species mnemonics (af3 5-DB pipeline,
            tcrm_msa).

    No silent fallback: caller must specify which path to take.
    """
    if pairing_mode not in ("precomputed", "online"):
        raise ValueError(
            f"af3_to_of3: pairing_mode must be 'precomputed' or 'online', "
            f"got {pairing_mode!r}. Set explicitly via OF3_PAIRING_MODE env "
            f"or `--pairing-mode` flag in the OF3 submit script."
        )
    import shutil
    seeds = af3.get("modelSeeds") or [42]
    name = af3.get("name") or "query"
    chains = []
    for sidx, seq_block in enumerate(af3.get("sequences", [])):
        prot = seq_block.get("protein")
        lig = seq_block.get("ligand")
        if lig is not None:
            cid = lig["id"] if isinstance(lig["id"], list) else [lig["id"]]
            lig_chain: dict = {"molecule_type": "ligand", "chain_ids": cid}
            if lig.get("smiles"):
                lig_chain["smiles"] = lig["smiles"]
            elif lig.get("ccdCodes"):
                codes = lig["ccdCodes"]
                if isinstance(codes, list) and len(codes) == 1:
                    codes = codes[0]
                lig_chain["ccd_codes"] = codes
            else:
                raise ValueError(
                    f"af3_to_of3: ligand at sequences[{sidx}] has neither "
                    f"`smiles` nor `ccdCodes`"
                )
            chains.append(lig_chain)
            continue
        na = seq_block.get("rna") or seq_block.get("dna")
        if na is not None:
            cid = na["id"] if isinstance(na["id"], list) else [na["id"]]
            mol = "rna" if seq_block.get("rna") is not None else "dna"
            na_chain = {
                "molecule_type": mol,
                "chain_ids": cid,
                "sequence": na["sequence"],
            }
            chain_tag = cid[0] if cid else f"c{sidx}"
            # Single-seq (inline unpairedMsa) → no MSA files; OF3 runs the
            # nucleic chain single-sequence. A real precomputed unpaired path
            # is staged as colabfold_main.a3m (same as the protein branch).
            unp = na.get("unpairedMsaPath")
            if unp:
                if sidecar_dir is not None:
                    msa_dir = sidecar_dir / name / f"chain_{chain_tag}_msa"
                    msa_dir.mkdir(parents=True, exist_ok=True)
                    dst = msa_dir / "colabfold_main.a3m"
                    shutil.copy2(unp, dst)
                    na_chain["main_msa_file_paths"] = [str(dst.resolve())]
                else:
                    na_chain["main_msa_file_paths"] = [unp]
            chains.append(na_chain)
            continue
        if prot is None:
            continue
        cid = prot["id"] if isinstance(prot["id"], list) else [prot["id"]]
        sequence = prot["sequence"]
        chain = {
            "molecule_type": "protein",
            "chain_ids": cid,
            "sequence": sequence,
        }
        if mods := _af3_modifications(prot):
            chain["non_canonical_residues"] = {p: c for p, c in mods}
        chain_tag = cid[0] if cid else f"c{sidx}"

        unp_path = prot.get("unpairedMsaPath")
        ppm_path = prot.get("pairedMsaPath")
        # Branch on pairing_mode (caller-specified, no default):
        #   online      → stage paired as `uniprot_hits.a3m` in main pool
        #                 (OF3 runs find_pairing_indices on TrEMBL headers).
        #   precomputed → stage paired as `colabfold_paired.a3m` in paired
        #                 pool (OF3 stacks rows by index, no species).
        if sidecar_dir is not None and (unp_path or ppm_path):
            # IMPORTANT: nest per-chain dirs under `name/` so multi-target
            # batches don't collide. inference.py reuses the same sidecar_dir
            # for every target in a batch; without the `name` prefix the
            # second call's shutil.copy2() overwrote the first call's MSA at
            # `chain_<X>_msa/colabfold_main.a3m`, leaving OF3 reading whichever
            # target won the race. That mismatch produced a silent IndexError
            # in map_msas_to_tokens when later targets had longer chains than
            # the surviving MSA.
            msa_dir = sidecar_dir / name / f"chain_{chain_tag}_msa"
            msa_dir.mkdir(parents=True, exist_ok=True)
            main_paths: list[str] = []
            if unp_path:
                dst = msa_dir / "colabfold_main.a3m"
                shutil.copy2(unp_path, dst)
                main_paths.append(str(dst.resolve()))
            if ppm_path:
                if pairing_mode == "online":
                    # OF3's parser filters by basename ∈ max_seq_counts. The
                    # default config has `uniprot_hits` (cap=50000); `uniprot`
                    # alone would be skipped. `uniprot_hits` is also listed
                    # in `msas_to_pair`. Sanitize headers: OF3's
                    # `process_msa_pairing_metadata` splits the FULL header
                    # on r"[|_/:-]" and expects exactly 6 columns
                    # (tr|UniProt|UniProt_SPECIES/start-end). Our
                    # paired_X.a3m carries verbose AF3-style headers with
                    # extra `_` in the description (OS=, OX=, GN=, PE=, SV=)
                    # that produce 12+ split tokens → ValueError. Keep only
                    # the first whitespace-separated token. Also strip
                    # `-<digit>+` isoform suffix from the accession so the
                    # split lands on exactly 6 tokens.
                    dst = msa_dir / "uniprot_hits.a3m"
                    _ISO = re.compile(r"^(>?(?:sp|tr)\|[A-Za-z0-9]+)-\d+(\|.*)$")
                    with open(ppm_path) as _src_f, open(dst, "w") as _dst_f:
                        for _line in _src_f:
                            if _line.startswith(">"):
                                _h = _line.split(maxsplit=1)[0]
                                _m = _ISO.match(_h)
                                if _m:
                                    _h = _m.group(1) + _m.group(2)
                                _dst_f.write(_h + "\n")
                            else:
                                _dst_f.write(_line)
                    main_paths.append(str(dst.resolve()))
                elif pairing_mode == "precomputed":
                    # Use `colabfold_paired` basename — OF3 default config
                    # has it in `paired_msa_order` and `max_seq_counts`.
                    dst = msa_dir / "colabfold_paired.a3m"
                    shutil.copy2(ppm_path, dst)
                    chain["paired_msa_file_paths"] = [str(dst.resolve())]
                # (no else: ValueError already raised above)
            if main_paths:
                chain["main_msa_file_paths"] = main_paths
        else:
            # No sidecar staging — pass paths through. Same branching:
            #   online → paired joins main pool (relies on basename match).
            #   precomputed → paired stays in paired_msa_file_paths.
            main_paths = []
            if unp_path:
                main_paths.append(unp_path)
            if pairing_mode == "online" and ppm_path:
                main_paths.append(ppm_path)
            elif pairing_mode == "precomputed" and ppm_path:
                chain["paired_msa_file_paths"] = [ppm_path]
            if main_paths:
                chain["main_msa_file_paths"] = main_paths

        # Templates → hmmsearch a3m sidecar. Same per-target nesting as the
        # MSA staging above — without `name/` the second target's templates
        # would overwrite the first in a batch.
        tpls = prot.get("templates") or []
        if tpls and sidecar_dir is not None:
            a3m_str = af3_templates_to_hmmsearch_a3m(sequence, tpls)
            a3m_path = sidecar_dir / name / f"templates_{chain_tag}_of3.a3m"
            a3m_path.parent.mkdir(parents=True, exist_ok=True)
            a3m_path.write_text(a3m_str)
            chain["template_alignment_file_path"] = str(a3m_path.resolve())

        chains.append(chain)
    out = {"seeds": seeds, "queries": {name: {"chains": chains}}}

    # Pass AF3 multimerTemplates through to OF3 so its template
    # featurization can build cross-chain Templates rows. Stored under
    # the per-query dict alongside chains. OF3 uses snake_case Pydantic
    # fields (Query.multimer_templates), so emit snake_case keys.
    mt_of3 = []
    for mt in (af3.get("multimerTemplates") or []):
        cif = mt.get("mmcifPath")
        qi = mt.get("queryIndices")
        ti = mt.get("templateIndices")
        if cif and isinstance(qi, dict) and qi:
            mt_of3.append({
                "mmcif_path": cif,
                "query_indices": qi,
                "template_indices": ti,
            })
    if mt_of3:
        out["queries"][name]["multimer_templates"] = mt_of3

    # AF3 bondedAtomPairs → OF3 covalent bonds: NOT emitted. OF3's query
    # schema has a `covalent_bonds` field but its Atom triplet is
    # (chain_id, residue_id, atom_id:int) — an integer atom *index*, not the
    # atom *name* AF3/pt2/bt2 use, so faithfully mapping a glycan link would
    # require resolving each CCD's atom ordering. OF3 docs also list
    # "covalently bound ligands" and "polymeric ligands such as glycans" as
    # not-yet-supported (inference.md §1.4). So glycans pass through as
    # NON-covalent ligands here; warn so the divergence from af3/pt2/bt2 is
    # visible rather than silent.
    if af3.get("bondedAtomPairs"):
        sys.stderr.write(
            f"[af3_to_of3] WARNING: {len(af3['bondedAtomPairs'])} bondedAtomPair(s) "
            f"dropped — OF3 has no atom-name covalent-bond input and flags "
            f"covalent/glycan ligands as unsupported; glycans modelled as "
            f"non-covalent ligands for OF3.\n"
        )
    return out


_OX_RE = re.compile(r"OX=(\d+)")


def _a3m_records(a3m_text: str):
    """Yield (header, aligned_seq) pairs from an a3m string. Drops blank lines."""
    cur_header = None
    for line in a3m_text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            cur_header = line
        elif cur_header is not None:
            yield cur_header, line
            cur_header = None


def af3_to_bt2_yaml(af3: dict, sidecar_dir: Optional[Path] = None) -> str:
    """Convert AF3 input.json → Boltz YAML.

    Boltz's a3m parser (boltz/data/parse/a3m.py) calls parse_a3m with
    `taxonomy=None`, so even with valid UniProt OX= headers the per-row
    taxonomy_id ends up as -1 and `construct_paired_msa` can't pair across
    chains. To pre-bake the pairing signal, we emit a `.csv` with two
    columns (sequence,key) — boltz's parse_csv reads `key` as taxonomy_id
    directly. We populate `key` with the OX id parsed from paired
    (uniprot) headers; unpaired-pool rows get an empty `key` so they
    never enter the cross-chain pairing pool.
    """
    lines: list[str] = ["version: 1", "sequences:"]
    # Each entry is either:
    #   (cif_path, chain_id_str)  — single-chain per-chain template
    #   {"cif", "chain_id":[...], "template_id":[...]}  — multi-chain template
    #     emitted from AF3 top-level `multimerTemplates`. Boltz schema reads
    #     this as matched-mode: chain_id[i] is mapped to template_id[i] in
    #     the cif, and process_template_features unifies all records sharing
    #     a template name into one row that spans every chain it covers.
    #     Combined with our boltz_inference.py monkey-patch on
    #     TemplateModule.forward (BOLTZ_MULTIMER_TEMPLATE=1), the inter-chain
    #     pair features that naturally land on those chain regions survive
    #     the embedder asym_mask gate.
    template_entries: list = []
    for sidx, seq_block in enumerate(af3.get("sequences", [])):
        prot = seq_block.get("protein")
        lig = seq_block.get("ligand")
        if lig is not None:
            cid = lig["id"] if isinstance(lig["id"], list) else [lig["id"]]
            cid_str = ", ".join(cid)
            lines.append("  - ligand:")
            lines.append(f"      id: [{cid_str}]")
            if lig.get("smiles"):
                # Quote SMILES — `:` and special chars trip YAML otherwise.
                smi = lig["smiles"].replace('"', '\\"')
                lines.append(f'      smiles: "{smi}"')
            elif lig.get("ccdCodes"):
                codes = lig["ccdCodes"]
                if isinstance(codes, list):
                    if len(codes) == 1:
                        lines.append(f"      ccd: {codes[0]}")
                    else:
                        lines.append(
                            f"      ccd: [{', '.join(codes)}]"
                        )
                else:
                    lines.append(f"      ccd: {codes}")
            else:
                raise ValueError(
                    f"af3_to_bt2_yaml: ligand at sequences[{sidx}] has neither "
                    f"`smiles` nor `ccdCodes`"
                )
            continue
        if prot is None:
            continue
        cid = prot["id"] if isinstance(prot["id"], list) else [prot["id"]]
        cid_str = ", ".join(cid)
        seq = prot["sequence"]
        head_id = cid[0] if cid else f"c{sidx}"

        msa_ref = "empty"
        if sidecar_dir is not None:
            unp_text = ""
            ppm_text = ""
            if prot.get("unpairedMsaPath"):
                p = Path(prot["unpairedMsaPath"])
                if p.exists():
                    unp_text = p.read_text()
            elif prot.get("unpairedMsa"):
                unp_text = prot["unpairedMsa"]
            if prot.get("pairedMsaPath"):
                p = Path(prot["pairedMsaPath"])
                if p.exists():
                    ppm_text = p.read_text()
            elif prot.get("pairedMsa"):
                ppm_text = prot["pairedMsa"]

            if unp_text or ppm_text:
                csv_path = Path(sidecar_dir) / f"msa_{head_id}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                # No cross-pool dedup in our adapter — match af3/pt2/of3
                # which keep the paired (uniprot) and unpaired
                # (uniref90+bfd+mgnify) tracks side by side. boltz's
                # parse_csv has its own `visited` sequence-set that
                # dedupes by sequence and keeps the FIRST occurrence;
                # by emitting paired rows first we make sure the
                # tax-tagged row wins for sequences present in both pools.
                # The reported counts in inputs_runtime.json then equal
                # the source-a3m record counts (matching pt2/of3).
                csv_lines = ["sequence,key", f"{seq},"]
                # Paired pool first, with OX tax_id from header. Skip the
                # leading `>query` record of each pool (same sequence as
                # the row we just emitted; boltz would dedup it anyway).
                for i, (hdr, row_seq) in enumerate(_a3m_records(ppm_text)):
                    if i == 0:
                        continue
                    m = _OX_RE.search(hdr)
                    tax = m.group(1) if m else ""
                    csv_lines.append(f"{row_seq},{tax}")
                # Unpaired pool second, no tax. Same skip-the-query rule.
                for i, (hdr, row_seq) in enumerate(_a3m_records(unp_text)):
                    if i == 0:
                        continue
                    csv_lines.append(f"{row_seq},")
                csv_path.write_text("\n".join(csv_lines) + "\n")
                msa_ref = str(csv_path.resolve())
        else:
            # No sidecar dir → fall back to unpaired-only a3m (no pairing).
            msa_ref = prot.get("unpairedMsaPath") or "empty"

        lines += [
            "  - protein:",
            f"      id: [{cid_str}]",
            f'      sequence: "{seq}"',
            f"      msa: {msa_ref}",
        ]
        if mods := _af3_modifications(prot):
            lines.append("      modifications:")
            for p, c in mods:
                lines += [f"        - position: {p}", f"          ccd: {c}"]
        for t in (prot.get("templates") or []):
            cif_path = t.get("mmcifPath")
            if cif_path:
                template_entries.append((cif_path, head_id))

    # AF3 multimerTemplates → one yaml entry per cif spanning every chain it
    # covers. queryIndices is a dict keyed by query chain id (e.g.
    # {"C": [...], "D": [...], "E": [...]}). The cif itself preserves the
    # source PDB's auth chain ids (e.g. 8YIV's A/B/C), NOT the query ids,
    # because combine.py re-orders chains to match queryIndices key order
    # but doesn't relabel them. Boltz schema wants template_id to be the
    # cif's actual auth chain ids in matched-mode, so we walk the cif and
    # pick the first auth chain id per (label_seq_id reset, atom appearance)
    # in the order they appear → 1:1 with queryIndices key order.
    for mt in (af3.get("multimerTemplates") or []):
        cif = mt.get("mmcifPath")
        qi = mt.get("queryIndices")
        ti = mt.get("templateIndices") or {}
        if not (cif and isinstance(qi, dict) and qi):
            continue
        chain_list = list(qi.keys())
        # Walk the CIF and build (chain_name, start_global_idx, end_global_idx)
        # so we can map each query chain's templateIndices range → CIF chain.
        chain_ranges: list[tuple[str, int, int]] = []
        try:
            cif_text = Path(cif).read_text()
            template_chain_order = _extract_cif_polymer_chain_order(cif_text)
            # Parse _atom_site column headers to find auth_seq_id/auth_asym_id
            # column indices (varies — combine.py may strip pdbx_formal_charge).
            atom_site_cols: list[str] = []
            for ln in cif_text.splitlines():
                if ln.startswith("_atom_site."):
                    atom_site_cols.append(ln.split(".", 1)[1].strip())
                elif atom_site_cols and ln.startswith("ATOM "):
                    break
            try:
                asym_col = atom_site_cols.index("auth_asym_id")
                seq_col = atom_site_cols.index("auth_seq_id")
            except ValueError:
                asym_col = seq_col = -1
            chain_to_residues: dict[str, set] = {}
            if asym_col >= 0 and seq_col >= 0:
                for ln in cif_text.splitlines():
                    if not ln.startswith("ATOM "):
                        continue
                    tok = ln.split()
                    if len(tok) > max(asym_col, seq_col):
                        chain_to_residues.setdefault(
                            tok[asym_col], set()
                        ).add(tok[seq_col])
            cum = 0
            for ch in template_chain_order:
                n = len(chain_to_residues.get(ch, set()))
                chain_ranges.append((ch, cum, cum + n - 1))
                cum += n
        except Exception:  # pylint: disable=broad-except
            chain_ranges = []

        # For each query chain, find which CIF chain its templateIndices
        # range falls into. templateIndices[query_chain] is a list of
        # global integer indices in the CIF's flat polymer space.
        template_id: list[str] = []
        if chain_ranges and ti:
            for qc in chain_list:
                tidx = ti.get(qc) or []
                if not tidx:
                    template_id = []
                    break
                first_idx = int(tidx[0])
                found = None
                for ch, lo, hi in chain_ranges:
                    if lo <= first_idx <= hi:
                        found = ch
                        break
                if found is None:
                    template_id = []
                    break
                template_id.append(found)

        if not template_id:
            # Fallback: positional mapping (legacy behavior).
            template_chain_list = (
                [c for c, _, _ in chain_ranges]
                if chain_ranges else []
            )
            if len(template_chain_list) >= len(chain_list):
                template_id = template_chain_list[: len(chain_list)]
            else:
                template_id = chain_list
        template_entries.append({
            "cif": cif,
            "chain_id": chain_list,
            "template_id": template_id,
        })

    if template_entries:
        lines.append("templates:")
        for entry in template_entries:
            if isinstance(entry, tuple):
                cif_path, chain = entry
                lines += [f"  - cif: {cif_path}", f"    chain_id: [{chain}]"]
            else:
                lines.append(f"  - cif: {entry['cif']}")
                lines.append(
                    f"    chain_id: [{', '.join(entry['chain_id'])}]"
                )
                lines.append(
                    f"    template_id: [{', '.join(entry['template_id'])}]"
                )

    # AF3 → Boltz YAML `constraints:` block (bonds + contacts, one shared
    # list). Emitted once if either is present.
    constraint_lines: list[str] = []

    # AF3 bondedAtomPairs → Boltz `bond` constraints. Boltz references bonded
    # atoms by [chain_id, residue_index, atom_name] — the SAME triplet AF3
    # uses — so the mapping is verbatim (chain ids and atom names are
    # preserved by this converter). Covers protein↔glycan and inter-sugar
    # bonds (multi-CCD ligand residue index = CCD serial).
    for (c1, r1, a1), (c2, r2, a2) in _af3_bonds(af3):
        constraint_lines += [
            "  - bond:",
            f"      atom1: [{c1}, {r1}, {a1}]",
            f"      atom2: [{c2}, {r2}, {a2}]",
        ]

    # AF3 constraint.contact[] → Boltz `contact` constraints.
    # Maps (entity1/copy1) → chain_id by walking sequences in order:
    # entity index k (1-based) = sequences[k-1].protein.id[copy-1].
    # `force: true` is set globally — Boltz's contact potential then runs
    # as a hard constraint during diffusion (vs default soft / weak).
    contacts = ((af3.get("constraint") or {}).get("contact") or [])
    if contacts:
        # Build entity_idx (1-based) → list of chain ids
        entity_chains: list[list[str]] = []
        for sblk in af3.get("sequences", []):
            prot = sblk.get("protein")
            if prot is None:
                entity_chains.append([])
                continue
            cid = prot["id"] if isinstance(prot["id"], list) else [prot["id"]]
            entity_chains.append(cid)
        for c in contacts:
            e1, k1 = int(c["entity1"]), int(c["copy1"])
            e2, k2 = int(c["entity2"]), int(c["copy2"])
            p1, p2 = int(c["position1"]), int(c["position2"])
            chain1 = entity_chains[e1 - 1][k1 - 1]
            chain2 = entity_chains[e2 - 1][k2 - 1]
            max_d = float(c.get("max_distance", 6.0))
            constraint_lines += [
                "  - contact:",
                f"      token1: [{chain1}, {p1}]",
                f"      token2: [{chain2}, {p2}]",
                f"      max_distance: {max_d}",
                f"      force: true",
            ]

    if constraint_lines:
        lines.append("constraints:")
        lines += constraint_lines
    return "\n".join(lines) + "\n"


# ── Smoke-test entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("af3_json", type=Path, help="AF3 input.json")
    ap.add_argument("out_dir", type=Path, help="output dir for converted files")
    ap.add_argument("--target", choices=["pt2", "of3", "bt2", "all"], default="all")
    args = ap.parse_args()

    af3 = json.loads(args.af3_json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.target in ("pt2", "all"):
        pt2 = af3_to_pt2(af3, sidecar_dir=args.out_dir)
        (args.out_dir / "pt2_input.json").write_text(json.dumps(pt2, indent=2))
        print(f"  pt2 → {args.out_dir / 'pt2_input.json'}")

    if args.target in ("of3", "all"):
        of3 = af3_to_of3(af3, sidecar_dir=args.out_dir)
        (args.out_dir / "of3_input.json").write_text(json.dumps(of3, indent=2))
        print(f"  of3 → {args.out_dir / 'of3_input.json'}")

    if args.target in ("bt2", "all"):
        yaml_str = af3_to_bt2_yaml(af3)
        (args.out_dir / "bt2_input.yaml").write_text(yaml_str)
        print(f"  bt2 → {args.out_dir / 'bt2_input.yaml'}")
