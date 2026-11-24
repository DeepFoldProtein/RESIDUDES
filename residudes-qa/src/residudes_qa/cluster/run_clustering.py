#!/usr/bin/env python3
"""
Run MassiveClustering on a CASP17 target's predictions, reading CIFs directly.

Discovers all `model.cif` files for one (target, stoichiometry, MSA-variant)
combo, optionally drops antibody/TCR CDR residues per ANARCI (IMGT scheme),
computes one distance matrix on GPU (TM / iTM / dRMSD), then hands the matrix
to `massivecluster comprehensive` which performs adaptive K selection,
clustering, t-SNE/UMAP/heatmap plotting, consensus extraction, medoid PDB
extraction, provenance analysis (per-model occupancy of clusters), and
cluster-level confidence statistics from summary.json (mean/min/max
plddt/ptm/iptm per cluster).

Original CIFs in <target>/results/.../sample_*/ are read-only — every
produced artefact lands under <target>/cluster_results/.

Path conventions (mirror inferences/af3/submit_inference.sh):
    sto with one distinct chain letter (A1, A2, B3 …) and a target/<X>/
    subdir exists ->
        <target>/<X>/results/<sto>/<variant>/<model>/seed_*/sample_*/model.cif
    multi-letter sto (A1B1, A1B1C1D1E1 …) or monomer target ->
        <target>/results/<sto>/<variant>/<model>/seed_*/sample_*/model.cif

Output (mirrors layout of inferences/visualization):
    <target>/cluster_results/<sto>/<variant>/<method>[_cdr]/
        matrices/
            <method>_matrix.npy
            structure_names.npy        (matrix row index -> filename)
            cdr_residues.json          (when --cdr)
        ranking.csv                    summary.json -> per-prediction confidence
        results/                       massivecluster comprehensive output:
            <tag>_*_clustering.csv     (per-structure -> cluster + soft membership)
            <tag>_*_cluster_summary.csv
            <tag>_*_native_scores.csv  per-cluster confidence stats (mean/min/max)
            <tag>_*_provenance.csv     per-cluster model distribution
            <tag>_consensus/           per-cluster consensus CIF + RMSF
            <tag>_representatives/     per-cluster medoid PDB
            *.png                      t-SNE + UMAP + silhouette + heatmap

Usage:
    run_clustering.py <target_dir> --sto A1B1 --method tm
    run_clustering.py <target_dir> --sto C1D1 --method itm --cdr
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gemmi
import numpy as np


# ─── Constants ──────────────────────────────────────────────────────────────

SAMPLE_RE = re.compile(r"sample_(\d+)$")
SEED_RE = re.compile(r"seed_(\d+)$")

# IMGT CDR positions (inclusive, both heavy/light Ig and TCR α/β)
IMGT_CDR_RANGES = [(27, 38), (56, 65), (105, 117)]


def _log(msg: str) -> None:
    print(msg, flush=True)


# ─── Path resolution ────────────────────────────────────────────────────────


def resolve_base_dir(target_dir: Path, sto: str) -> Path:
    """
    Mirror inferences/af3/submit_inference.sh's BASE rule.

    Single distinct chain letter in `sto` (A1, A2, …) AND a target/<X>/ subdir
    exists -> base = target/<X>. Otherwise base = target.
    """
    letters = sorted(set(re.findall(r"[A-Z]", sto)))
    if len(letters) == 1:
        candidate = target_dir / letters[0]
        if (candidate / "results" / sto).is_dir():
            return candidate
    if (target_dir / "results" / sto).is_dir():
        return target_dir
    raise SystemExit(f"could not find results/{sto}/ under {target_dir}")


# An "item" tuple flowing through the pipeline:
#   (variant, model, seed, sample, structure_path, name_override)
# - The first 5 fields are how every prediction is identified end-to-end.
# - `name_override` (str | None) is the canonical filename to put into the
#   distance matrix / ranking CSV. None means "derive via name_for(...)"
#   (the in-house convention). A non-None value means "use this filename
#   verbatim" — used by the flat-dir mode so original MassiveFold-style
#   filenames (Model_<idx>_<engine>_..._pred_<i>.{pdb,cif}) reach the
#   provenance regex unchanged.
ItemTuple = tuple[str, str, int, int, Path, "str | None"]  # last field: name_override


def discover_predictions(
    base_dir: Path, sto: str, variants: list[str]
) -> list[ItemTuple]:
    """Walk <base>/results/<sto>/<variant>/<model>/seed_*/sample_*/model.cif
    across one or more variants. Returns (variant, model, seed, sample, cif, None)
    — the trailing None signals "use name_for(...) for the canonical filename".

    Variant strings are preserved as the directory name (e.g.
    'AB_CDE_template_AB1_CDE1') so downstream provenance can attribute
    each prediction to its specific MSA / template build.
    """
    items: list[ItemTuple] = []
    sto_root = base_dir / "results" / sto
    for variant in variants:
        variant_root = sto_root / variant
        if not variant_root.is_dir():
            raise SystemExit(f"no results dir: {variant_root}")
        for model_dir in sorted(variant_root.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for seed_dir in sorted(model_dir.iterdir()):
                sm = SEED_RE.match(seed_dir.name)
                if not sm:
                    continue
                seed = int(sm.group(1))
                for sample_dir in sorted(seed_dir.iterdir()):
                    ssm = SAMPLE_RE.match(sample_dir.name)
                    if not ssm:
                        continue
                    sample = int(ssm.group(1))
                    cif = sample_dir / "model.cif"
                    if cif.is_file():
                        items.append((variant, model, seed, sample, cif, None))
    items.sort(key=lambda it: (it[0], it[1], it[2], it[3]))
    return items


# Filenames inside a flat MassiveFold-style bundle look like
#   Model_<idx>_<engine>_<paramset>_..._pred_<i>.{pdb,cif}
# We capture <idx> as the "seed" so each model has a unique numeric handle
# that survives into ranking lookup; sample is fixed at 0.
_FLAT_NAME_RE = re.compile(r"^Model_(\d+)_")


def discover_flat_predictions(flat_dir: Path, tag: str) -> list[ItemTuple]:
    """Enumerate `Model_<idx>_..._pred_<i>.{pdb,cif}` files in a flat directory.

    Returns 6-tuples (variant=tag, model=path.stem, seed=<idx>, sample=0,
    path=<absolute>, name_override=path.name). The name_override preserves
    the original MassiveFold filename so the provenance regex
    (^Model_<digits>_...) attributes each prediction to its parameter set
    (e.g. mf_afm_basic, mf_af3_woPaired) without any renaming.

    Files not matching the `Model_<digits>_` prefix (e.g. features.pkl,
    ranking.csv, ranking_*.json) are silently skipped.
    """
    if not flat_dir.is_dir():
        raise SystemExit(f"--flat-dir not a directory: {flat_dir}")
    items: list[ItemTuple] = []
    for p in sorted(flat_dir.iterdir()):
        if p.suffix not in (".pdb", ".cif"):
            continue
        m = _FLAT_NAME_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        items.append((tag, p.stem, idx, 0, p, p.name))
    items.sort(key=lambda it: (it[0], it[2]))  # by tag then numeric idx
    return items


def list_available_variants(base_dir: Path, sto: str) -> list[str]:
    """All sub-directories under <base>/results/<sto>/. Used by --variants all."""
    sto_root = base_dir / "results" / sto
    if not sto_root.is_dir():
        return []
    return sorted([d.name for d in sto_root.iterdir() if d.is_dir()])


# ─── ANARCI / CDR detection ────────────────────────────────────────────────


def get_chain_sequences(cif_path: Path) -> dict[str, str]:
    """Extract per-chain one-letter sequences from a representative CIF."""
    st = gemmi.read_structure(str(cif_path))
    out: dict[str, str] = {}
    if len(st) == 0:
        return out
    for chain in st[0]:
        seq = []
        for r in chain:
            tab = gemmi.find_tabulated_residue(r.name)
            seq.append(tab.one_letter_code.upper() if tab else "X")
        out[chain.name] = "".join(seq)
    return out


def chain_shape(cif_path: Path) -> tuple[tuple[str, int], ...]:
    """Compact shape signature: ((chain_id, length), ...). Used to detect
    inputs_runtime mix-ups where some models in the same sto folder ran
    against a different stoichiometry's input."""
    seqs = get_chain_sequences(cif_path)
    return tuple(sorted((c, len(s)) for c, s in seqs.items()))


def verify_sequence_consistency(
    items: list[tuple[str, str, int, int, Path]],
    allow_mismatch: bool = False,
) -> tuple[tuple[str, int], ...]:
    """
    Check that every (model.cif) under this (target, sto, variant(s)) has
    the same per-chain shape — i.e. they're all really the same complex.
    If not, summarise which (variant, model, shape) groups exist and abort.
    """
    from collections import defaultdict
    by_shape: dict[tuple[tuple[str, int], ...], list[tuple[str, str, int, int, Path]]] = defaultdict(list)
    for it in items:
        try:
            shape = chain_shape(it[4])  # cif is now items[4]
        except Exception as exc:
            _log(f"  ! could not read {it[4]}: {exc}")
            continue
        by_shape[shape].append(it)

    if len(by_shape) == 1:
        only_shape = next(iter(by_shape))
        _log(f"  shape ok: every structure has chains "
             f"{', '.join(f'{c}={l}' for c, l in only_shape)}")
        return only_shape

    # Multiple shapes → diagnose, broken down by variant + model
    _log("  ! sequence shape MISMATCH detected — input mix-up upstream")
    _log(f"  found {len(by_shape)} distinct shapes:")
    for shape, group in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        per_vm: dict[tuple[str, str], int] = {}
        for v, m, _, _, _, _ in group:
            per_vm[(v, m)] = per_vm.get((v, m), 0) + 1
        shape_str = ", ".join(f"{c}={l}" for c, l in shape)
        vm_str = ", ".join(f"{v}/{m}:{n}" for (v, m), n in sorted(per_vm.items()))
        _log(f"    {shape_str}  -- {len(group)} structures ({vm_str})")

    if allow_mismatch:
        # pick the most-populous shape as canonical
        canonical = max(by_shape.items(), key=lambda kv: len(kv[1]))[0]
        _log(f"  --allow-mismatch given. Continuing with the dominant shape only "
             f"({len(by_shape[canonical])} structures).")
        items[:] = by_shape[canonical]
        return canonical

    raise SystemExit(
        "aborting. The clustering only makes sense when every structure shares "
        "the same chain composition. Re-run inference for the offending models, "
        "or pass --allow-mismatch to keep only the dominant shape."
    )


_CDR_WORKER_SCRIPT = r"""
import json, sys
from anarci import anarci

# IMGT CDR ranges for both Ig and TCR variable domains
CDR_RANGES = {'cdr1': (27, 38), 'cdr2': (56, 65), 'cdr3': (105, 117)}
inputs = json.loads(sys.stdin.read())
results, _, _ = anarci(inputs, scheme="imgt", assign_germline=False, output=False)
out = {}
for (name, _seq), result in zip(inputs, results):
    if result is None:
        out[name] = {'cdr1': [], 'cdr2': [], 'cdr3': []}
        continue
    per_cdr = {'cdr1': [], 'cdr2': [], 'cdr3': []}
    for numbering, start, _end in result:
        seq_idx = start
        for (pos, _ins), aa in numbering:
            if aa == "-":
                continue
            for cdr_name, (lo, hi) in CDR_RANGES.items():
                if lo <= pos <= hi:
                    per_cdr[cdr_name].append(seq_idx)
                    break
            seq_idx += 1
    out[name] = {k: sorted(set(v)) for k, v in per_cdr.items()}
print(json.dumps(out))
"""


def detect_cdr_residues(
    chain_sequences: dict[str, str],
) -> dict[str, dict[str, list[int]]]:
    """
    Run ANARCI per chain. Returns
        {chain_id: {'cdr1': [...], 'cdr2': [...], 'cdr3': [...]}}
    where the lists hold 0-based residue indices in the chain's IMGT
    CDR loops. Chains with no Ig/TCR domain detected get all-empty
    lists (= equivalent to BASE for that chain).

    Use the helpers `merge_all_cdr` / `extract_cdr_subset` (downstream)
    if you want a single merged set or a per-loop subset.

    ANARCI lives in a sibling venv `analysis/.venv-cdr/` (to avoid pinning
    the main venv to a specific biopython for HMMER parsing). We shell out
    to it, passing/receiving JSON over stdio.
    """
    here = Path(__file__).resolve()
    cdr_python = here.parent.parent / ".venv-cdr" / "bin" / "python"
    if not cdr_python.is_file():
        raise SystemExit(
            "ANARCI venv not set up.\n"
            "  Run: bash analysis/setup.sh --with-cdr"
        )

    inputs = [(name, seq) for name, seq in chain_sequences.items()]
    env = os.environ.copy()
    # Make the symlinked HMMER binaries (analysis/.venv-cdr/bin/hmmscan)
    # findable from the cdr venv subprocess.
    env["PATH"] = f"{cdr_python.parent}:" + env.get("PATH", "")
    proc = subprocess.run(
        [str(cdr_python), "-c", _CDR_WORKER_SCRIPT],
        input=json.dumps(inputs),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"ANARCI subprocess failed (rc={proc.returncode}).\n"
            f"  stderr: {proc.stderr.strip()[:500]}"
        )
    return json.loads(proc.stdout)


# ─── Anchor / probe spec parsing ───────────────────────────────────────────

VALID_MODES = {"all", "framework", "cdr", "cdr1", "cdr2", "cdr3"}


def _parse_residue_ranges(s: str) -> frozenset[int]:
    """'115-151,160-180,200' → frozenset of 1-based residue numbers.
    Caller converts to 0-based when building masks."""
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"empty residue range {s!r}")
    return frozenset(out)


def parse_chain_spec(spec: str) -> list[tuple[set[str], str]]:
    """
    Mini-language for anchor / probe selection.

    Grammar:
        spec     ::= group ('+' group)*
        group    ::= chain_chars [':' mode]
        chain_chars ::= [A-Z]+
        mode     ::= 'all' | 'framework' | 'cdr' | 'cdr1' | 'cdr2' | 'cdr3'
                   | 'res=<ranges>'        (e.g. 'res=115-151,160-180' — 1-based)

    Examples:
        'AB:cdr+E'                → [({A,B}, 'cdr'), ({E}, 'all')]
        'AB:cdr3+E'               → [({A,B}, 'cdr3'), ({E}, 'all')]
        'A:cdr1+B:cdr3+E'         → [({A}, 'cdr1'), ({B}, 'cdr3'), ({E}, 'all')]
        'CDE'                     → [({C,D,E}, 'all')]
        'A:res=115-151+B:cdr'     → [({A}, 'res=115-151'), ({B}, 'cdr')]
        'A:res=115-151,160-180'   → [({A}, 'res=115-151,160-180')]   (multi-range)
    """
    groups: list[tuple[set[str], str]] = []
    for raw in spec.split("+"):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"empty group in spec {spec!r}")
        if ":" in raw:
            chains_part, mode = raw.split(":", 1)
            mode = mode.strip()
        else:
            chains_part = raw
            mode = "all"
        chain_chars = re.findall(r"[A-Z]", chains_part)
        if not chain_chars:
            raise ValueError(
                f"group {raw!r} has no uppercase chain letters"
            )
        # res=... mode: validate ranges parse cleanly here so errors surface early
        if mode.startswith("res="):
            try:
                _parse_residue_ranges(mode[len("res="):])
            except Exception as exc:
                raise ValueError(
                    f"group {raw!r}: bad res= ranges ({exc})"
                )
        elif mode not in VALID_MODES:
            raise ValueError(
                f"group {raw!r} has unknown mode {mode!r}; "
                f"valid: {sorted(VALID_MODES)} or 'res=<ranges>'"
            )
        groups.append((set(chain_chars), mode))
    if not groups:
        raise ValueError(f"empty spec {spec!r}")
    return groups


def spec_to_tag(groups: list[tuple[set[str], str]]) -> str:
    """
    Path / job-name friendly encoding of a parsed spec.
    Uses '_' to join groups (dir-name safe, no '+').
    For 'res=<ranges>' modes, sanitises commas → '.' for filename safety.

    [({A,B}, 'cdr'), ({E}, 'all')]                       → 'AB-cdr_E'
    [({A}, 'cdr1'), ({B}, 'cdr3'), ({E}, 'all')]         → 'A-cdr1_B-cdr3_E'
    [({C,D}, 'all')]                                     → 'CD'
    [({A}, 'res=115-151'), ({B}, 'cdr')]                 → 'A-res115-151_B-cdr'
    [({A}, 'res=115-151,160-180')]                       → 'A-res115-151.160-180'
    """
    parts: list[str] = []
    for chains, mode in groups:
        chain_str = "".join(sorted(chains))
        if mode == "all":
            parts.append(chain_str)
        elif mode.startswith("res="):
            # 'res=115-151,160-180' → 'res115-151.160-180'  (sanitize for path)
            sanitised = mode[len("res="):].replace(",", ".")
            parts.append(f"{chain_str}-res{sanitised}")
        else:
            parts.append(f"{chain_str}-{mode}")
    return "_".join(parts)


# ─── Provenance-friendly structure naming ──────────────────────────────────


def name_for(model: str, seed: int, sample: int, variant: str | None = None) -> str:
    """Filename understood by massivecluster's provenance regex
    (^af3_|^bt2_|^of3_|^pt2_) — uses .pdb suffix to match downstream
    expectations even though we stay in CIF land for matrix computation.

    With variant != None, the variant is appended as a SUFFIX with '__'
    separator so the model prefix at the start still matches the
    provenance regex unambiguously. None of the H1311 variant directory
    names contain '__' (double underscore) so this is a safe separator.

    Layout:
        single-variant:  af3_seed1_sample0.pdb                   (legacy)
        multi-variant:   af3_seed1_sample0__af3_msa.pdb          (new)
                         bt2_seed21_sample2__AB_CDE_template_AB1_CDE1.pdb
    """
    base = f"{model}_seed{seed}_sample{sample}"
    if variant is None:
        return f"{base}.pdb"
    return f"{base}__{variant}.pdb"


def key_for(item: ItemTuple, multi_variant: bool) -> str:
    """Canonical structure name for an item, used everywhere a filename
    serves as the lookup key (matrix row label, ranking.csv `file` column,
    provenance regex input).

    If the item carries an explicit `name_override` (last tuple field —
    populated by `discover_flat_predictions` to keep MassiveFold-style
    filenames intact), that name is returned verbatim. Otherwise the
    in-house convention `<model>_seed<S>_sample<I>[__<variant>].pdb` from
    `name_for(...)` is used.
    """
    v, m, s, i, _path, name_override = item
    if name_override is not None:
        return name_override
    return name_for(m, s, i, variant=(v if multi_variant else None))


def parse_name(name: str) -> tuple[str | None, str, int, int]:
    """Inverse of name_for. Returns (variant_or_None, model, seed, sample)."""
    stem = name[:-4] if name.endswith(".pdb") else name
    variant = None
    if "__" in stem:
        head, variant = stem.split("__", 1)
    else:
        head = stem
    m = re.match(r"^([a-zA-Z0-9]+)_seed(-?\d+)_sample(\d+)$", head)
    if not m:
        raise ValueError(f"can't parse structure name: {name!r}")
    return variant, m.group(1), int(m.group(2)), int(m.group(3))


# ─── GPU matrix computation ─────────────────────────────────────────────────


def compute_matrix(
    method: str,
    cif_paths: list[Path],
    out_path: Path,
    gpu: int,
    exclude_indices: dict[str, set[int]] | None,
    anchor_residues: dict[str, set[int]] | None = None,
    probe_residues: dict[str, set[int]] | None = None,
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """Run the GPU kernel directly on CIF files. With CDR mode we pass the
    per-chain residue exclude mask through to the loader so dropped residues
    never reach the GPU buffers. With anchored modes we pass anchor/probe
    KEEP sets to a separate kernel that decouples alignment from measurement."""
    from residudes_qa.cluster.core.gpu import (  # type: ignore
        compute_tm_matrix_gpu,
        compute_drmsd_matrix_gpu,
        compute_itm_matrix_gpu,
        compute_anchored_matrix_gpu,
    )
    from residudes_qa.cluster.core.gpu.com_distance import compute_com_matrix_gpu
    from residudes_qa.cluster.core.gpu.tcr_angle import compute_tcr_angle_matrix

    t0 = time.time()
    _log(f"  computing {method} matrix on GPU {gpu} (N={len(cif_paths)}) ...")
    if method == "tm":
        mat, names = compute_tm_matrix_gpu(
            cif_paths, batch_size=2048, gpu_id=gpu,
            output_path=out_path, show_progress=True,
            exclude_indices=exclude_indices,
        )
    elif method == "drmsd":
        mat, names = compute_drmsd_matrix_gpu(
            cif_paths, batch_size=50, gpu_id=gpu,
            output_path=out_path, show_progress=True,
            exclude_indices=exclude_indices,
        )
    elif method == "itm":
        mat, names, _ = compute_itm_matrix_gpu(
            cif_paths, batch_size=1024, gpu_id=gpu, interface_cutoff=8.0,
            output_path=out_path, show_progress=True,
            exclude_indices=exclude_indices,
        )
    elif method in ("anchored-rmsd", "anchored-tm"):
        if anchor_residues is None or probe_residues is None:
            raise SystemExit(
                f"compute_matrix({method!r}) needs anchor_residues and probe_residues"
            )
        kernel_metric = "rmsd" if method == "anchored-rmsd" else "tm"
        mat, names = compute_anchored_matrix_gpu(
            cif_paths,
            anchor_residues=anchor_residues,
            probe_residues=probe_residues,
            metric=kernel_metric,
            batch_size=1024, gpu_id=gpu,
            output_path=out_path, show_progress=True,
        )
    elif method == "com":
        # Anchored centre-of-mass distance.
        # Reuses the same anchor_residues mask as anchored-* modes.
        # Probe chains are derived from probe_residues keys — we take the
        # CHAIN names, ignoring residue subset (COM is whole-chain by design).
        if anchor_residues is None or probe_residues is None:
            raise SystemExit(
                "compute_matrix('com') needs anchor_residues and probe_residues "
                "(probe residues' chain keys define the probe chains)"
            )
        probe_chains = sorted(probe_residues.keys())
        mat, names, _features = compute_com_matrix_gpu(
            cif_paths,
            anchor_residues=anchor_residues,
            probe_chains=probe_chains,
            batch_size=1024, gpu_id=gpu,
            output_path=out_path, show_progress=True,
        )
    elif method == "tcr-angle":
        # Pierce-based TCR docking angle distance.
        # - tcr_angle_csv : pre-computed Pierce CSV (re-used if exists)
        # - tcr_angle_recompute : force re-invoking Pierce even if CSV exists
        # - tcr_angle_chain_layout / tcr_angle_target / tcr_angle_workers :
        #     pass-through to the Pierce worker
        # - tcr_angle_group_filter : CSV group_dir filter (e.g. 'af3_msa')
        # cif_paths supplied as before; structure_names derived from kernel
        # itself by stripping .cif and re-deriving pdb-name. But here we
        # already have the names list (built upstream from items[]); pass
        # via kwargs.
        names_list = kwargs.get("structure_names")
        if not names_list:
            # fall back to deriving from cif_paths
            names_list = []
            import re as _re
            for p in cif_paths:
                pp = Path(p)
                if pp.name == "model.cif" and len(pp.parts) >= 4:
                    gen = pp.parts[-4]
                    seed = pp.parts[-3].replace("seed_", "")
                    sample = pp.parts[-2].replace("sample_", "")
                    names_list.append(f"{gen}_seed{seed}_sample{sample}.pdb")
                else:
                    names_list.append(pp.name)
        mat, kept_names, _features_deg = compute_tcr_angle_matrix(
            structure_names=names_list,
            cif_paths=cif_paths,
            angle_csv=kwargs.get("tcr_angle_csv"),
            group_filter=kwargs.get("tcr_angle_group_filter"),
            chain_layout=kwargs.get("tcr_angle_chain_layout", "ours_5chain"),
            target_name=kwargs.get("tcr_angle_target"),
            recompute=bool(kwargs.get("tcr_angle_recompute", False)),
            workers=kwargs.get("tcr_angle_workers", 16),
            output_path=out_path, show_progress=True,
        )
        # If some names were dropped (no CSV match), surface to caller via
        # the names list.
        names = kept_names
    elif method in ("jaccard-epitope", "jaccard-paratope", "jaccard-contact"):
        # Interface binding-mode distance: 1 - Jaccard over residue sets
        # (epitope = Ag side, paratope = Ab side) or contact pairs
        # (-contact = Ag+Ab simultaneously). Computation lives in core
        # (iface_metrics, vectorised); here we extract the per-model contacts
        # table and dispatch. Result is ALREADY a distance in [0, 1].
        import pandas as _pd
        from residudes_qa.cluster.analysis.pair_contacts import (
            extract_pair_contacts_batch,
        )
        from residudes_qa.cluster.core.iface_metrics import (
            iface_jaccard_residue_matrix,
            iface_jaccard_contact_matrix,
        )
        ag_chains = kwargs.get("ag_chains")
        ab_chains = kwargs.get("ab_chains")
        if not ag_chains or not ab_chains:
            raise SystemExit(
                f"compute_matrix({method!r}) needs ag_chains and ab_chains "
                "(pass --ag-chains / --ab-chains)"
            )
        names = list(kwargs.get("structure_names") or [])
        if len(names) != len(cif_paths):
            raise SystemExit(
                f"{method}: structure_names ({len(names)}) != "
                f"cif_paths ({len(cif_paths)})"
            )
        cif_records = _pd.DataFrame(
            {"model_id": names, "src_cif": [str(p) for p in cif_paths]}
        )
        contacts = extract_pair_contacts_batch(
            cif_records,
            paratope_chains=list(ab_chains),
            epitope_chains=list(ag_chains),
            cutoff=float(kwargs.get("iface_cutoff", 5.0)),
            workers=16,
        )
        # bridge column names: pair_contacts emits epitope_*/paratope_*;
        # iface_metrics expects abc_*/ab_*.
        contacts = contacts.rename(columns={
            "epitope_chain": "abc_chain", "epitope_resnum": "abc_resnum",
            "paratope_chain": "ab_chain", "paratope_resnum": "ab_resnum",
        })
        if method == "jaccard-epitope":
            mat = iface_jaccard_residue_matrix(contacts, names, side="epitope")
        elif method == "jaccard-paratope":
            mat = iface_jaccard_residue_matrix(contacts, names, side="paratope")
        else:  # jaccard-contact
            mat = iface_jaccard_contact_matrix(contacts, names)
        np.save(out_path, mat)
    else:
        raise SystemExit(f"unknown method: {method}")
    _log(f"    done in {time.time() - t0:.1f}s, shape={mat.shape}")
    return mat, names


# ─── Ranking CSV (per-cluster confidence) ───────────────────────────────────


_RANKING_COLUMNS = [
    "file", "variant", "plddt", "ptm", "iptm",
    "ranking_score", "gpde", "fraction_disordered",
    "model", "seed", "sample_idx",
]


def build_ranking_csv(
    items: list[ItemTuple],
    keys_in_matrix_order: list[str],
    out_path: Path,
    multi_variant: bool,
    *,
    flat_ranking_path: Path | None = None,
    flat_tag: str | None = None,
) -> None:
    """Write `<output_dir>/ranking.csv` consumed by
    `massivecluster comprehensive --ranking` for per-cluster mean/min/max
    confidence aggregation.

    Two input modes:

    - **Default (in-house pipeline)**: walks per-prediction `summary.json`
      siblings of each model.cif. Each summary contributes one row.

    - **Flat-dir mode** (``flat_ranking_path`` set): loads a single
      MassiveFold-style ranking CSV (``ranking_<TARGET>.csv`` with a
      ``mapped_name`` column) once and joins on the on-disk filename. No
      per-prediction summary.json walk. Used by ``--flat-dir``.

    Both paths emit the same column schema so downstream consumption is
    uniform.
    """
    if flat_ranking_path is not None:
        return _build_ranking_from_flat_csv(
            items, keys_in_matrix_order, out_path,
            flat_ranking_path=flat_ranking_path,
            flat_tag=flat_tag or "",
        )
    by_key = {key_for(it, multi_variant): it[4] for it in items}
    by_key_variant = {key_for(it, multi_variant): it[0] for it in items}
    rows: list[dict] = []
    missing = 0
    for key in keys_in_matrix_order:
        cif = by_key.get(key)
        if cif is None:
            missing += 1
            continue
        summary = cif.parent / "summary.json"
        if not summary.is_file():
            missing += 1
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception:
            missing += 1
            continue
        rows.append({
            "file": key,
            "variant": by_key_variant.get(key, ""),
            "plddt": data.get("plddt"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "gpde": data.get("gpde"),
            "fraction_disordered": data.get("fraction_disordered"),
            "model": data.get("model_name"),
            "seed": data.get("seed"),
            "sample_idx": data.get("sample_idx"),
        })
    if not rows:
        _log("  ! no summary.json files found - ranking CSV not written")
        return
    cols = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _log(f"  ranking.csv: {len(rows)} rows -> {out_path}"
         + (f"  (missing summary.json for {missing})" if missing else ""))


def _build_ranking_from_flat_csv(
    items: list[ItemTuple],
    keys_in_matrix_order: list[str],
    out_path: Path,
    *,
    flat_ranking_path: Path,
    flat_tag: str,
) -> None:
    """Adapter: MassiveFold's `ranking_<TARGET>.csv` → massivecluster ranking.csv.

    Source schema (`ranking_<TARGET>.csv`):
        model, run, prediction, iptm+ptm, mapped_name, iptm, ptm,
        af3_ranking_score
    where `mapped_name` is the on-disk filename
    (e.g. `Model_1_afm_basic_model_4_multimer_v3_pred_27.pdb`).

    Mapping to the in-house schema:
        file                 = mapped_name (= structure key)
        variant              = flat_tag (one source pool)
        plddt                = NaN  (not exposed by MassiveFold)
        ptm                  = ptm
        iptm                 = iptm
        ranking_score        = `iptm+ptm` column (uniform AF2/AF3 ranker per
                               README; `0.8*ipTM+0.2*pTM`)
        gpde                 = NaN  (AF3-only, not exposed)
        fraction_disordered  = NaN  (AF3-only, not exposed)
        model                = run column (parameter set, e.g. "afm_basic")
        seed                 = model column (the `Model_<idx>` integer)
        sample_idx           = 0
    """
    if not flat_ranking_path.is_file():
        raise SystemExit(f"--flat-ranking not found: {flat_ranking_path}")

    # Load CSV and index by mapped_name. Use csv stdlib (not pandas) since
    # we're in a hot CLI path and the dependency isn't always loaded yet.
    by_mapped: dict[str, dict[str, str]] = {}
    with flat_ranking_path.open() as f:
        reader = csv.DictReader(f)
        if "mapped_name" not in (reader.fieldnames or []):
            raise SystemExit(
                f"--flat-ranking missing required column 'mapped_name': "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            by_mapped[row["mapped_name"]] = row

    def _num(s: str | None) -> float | None:
        if s is None or s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    rows: list[dict] = []
    missing = 0
    for key in keys_in_matrix_order:
        src = by_mapped.get(key)
        if src is None:
            missing += 1
            continue
        rows.append({
            "file": key,
            "variant": flat_tag,
            "plddt": None,
            "ptm": _num(src.get("ptm")),
            "iptm": _num(src.get("iptm")),
            "ranking_score": _num(src.get("iptm+ptm")),
            "gpde": None,
            "fraction_disordered": None,
            "model": src.get("run", ""),
            "seed": _num(src.get("model")),  # the Model_<idx> integer
            "sample_idx": 0,
        })

    if not rows:
        _log("  ! no rows matched between flat-ranking CSV and discovered "
             "structures - ranking CSV not written")
        return
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RANKING_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    _log(f"  ranking.csv: {len(rows)} rows -> {out_path}"
         f"  (from {flat_ranking_path.name}"
         + (f", missing rows for {missing} keys" if missing else "")
         + ")")


# ─── Comprehensive driver ──────────────────────────────────────────────────


def run_comprehensive(
    matrix_path: Path,
    method: str,
    names_path: Path,
    pdbs_dir: Path | None,
    ranking_csv: Path,
    out_dir: Path,
    target_tag: str,
    cluster_method: str,
) -> int:
    # `massivecluster comprehensive --matrix metric:path` uses the metric
    # label to decide similarity_to_distance: 'tm'/'itm' do 1-x, 'drmsd'
    # normalises by max. For our anchored modes:
    #   anchored-tm   -> similarity in [0, 1]  -> label as 'tm' (1-x)
    #   anchored-rmsd -> distance in Å         -> label as 'drmsd' (max-norm)
    label_map = {
        "tm": "tm", "itm": "itm", "drmsd": "drmsd",
        "anchored-tm": "tm", "anchored-rmsd": "drmsd",
    }
    metric_label = label_map.get(method, method)
    cmd = [
        sys.executable, "-m", "residudes_qa.cluster.cli", "comprehensive",
        "--matrix", f"{metric_label}:{matrix_path}",
        "--names", str(names_path),
        "--output", str(out_dir),
        "--target", target_tag,
        "--method", cluster_method,
    ]
    if pdbs_dir is not None:
        cmd += ["--pdbs", str(pdbs_dir)]
    if ranking_csv.is_file():
        cmd += ["--ranking", str(ranking_csv)]
    _log("  $ " + " ".join(cmd))
    return subprocess.call(cmd)


# ─── Confidence distribution post-processing ──────────────────────────────


def plot_confidence_distributions(
    ranking_csv: Path,
    clustering_csv: Path,
    out_path: Path,
    target_tag: str,
) -> None:
    """
    6-panel violin plot — one panel per confidence field
    (ranking_score, iptm, ptm, plddt, gpde, fraction_disordered).

    Each panel: cluster ID on x-axis (sized labels), distribution as
    violin + box, medoid marked with a black dot. Lets you eyeball
    per-cluster confidence spread vs. the cluster_summary.csv's
    mean/min/max table at a glance.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if not ranking_csv.is_file():
        return
    if not clustering_csv.is_file():
        return

    rk = pd.read_csv(ranking_csv)
    cl = pd.read_csv(clustering_csv)
    if "structure" not in cl.columns and "structure_name" in cl.columns:
        cl = cl.rename(columns={"structure_name": "structure"})
    if "cluster" not in cl.columns:
        return
    # Match on filename stem
    rk["_key"] = rk["file"].astype(str)
    cl["_key"] = cl["structure"].astype(str)
    merged = rk.merge(cl[["_key", "cluster"]], on="_key", how="inner")
    if merged.empty:
        return

    fields = [
        ("ranking_score", "AF3 ranking_score (0.8·iPTM + 0.2·pTM)"),
        ("iptm", "iPTM"),
        ("ptm", "pTM"),
        ("plddt", "pLDDT"),
        ("gpde", "gPDE"),
        ("fraction_disordered", "fraction disordered"),
    ]
    # Drop fields where >50% of values are NaN — they tend to have empty
    # cluster slices that break violinplot's reduction ops.
    available = []
    for c, l in fields:
        if c not in merged.columns:
            continue
        non_na_frac = merged[c].notna().mean()
        if non_na_frac >= 0.5:
            available.append((c, l))
    if not available:
        return

    n = len(available)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows),
                             constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    cluster_ids = sorted(merged["cluster"].unique())
    cluster_sizes = merged.groupby("cluster").size().to_dict()
    xticklabels = [f"C{c}\n(n={cluster_sizes.get(c, 0)})" for c in cluster_ids]

    for ax, (field, label) in zip(axes, available):
        # Build data; substitute single-zero placeholder for empty slices so
        # violinplot doesn't crash on min/max reductions over empty arrays.
        data = []
        n_empty = 0
        for c in cluster_ids:
            arr = merged.loc[merged["cluster"] == c, field].dropna().values
            if len(arr) == 0:
                n_empty += 1
                arr = np.array([np.nan])  # placeholder; violinplot skips
            data.append(arr)
        # filter clusters with empty data for violin (matplotlib can't handle)
        valid_idx = [i for i, d in enumerate(data) if not np.isnan(d).all()]
        if not valid_idx:
            ax.set_visible(False)
            continue
        valid_data = [data[i] for i in valid_idx]
        valid_pos = [i + 1 for i in valid_idx]
        parts = ax.violinplot(valid_data, positions=valid_pos,
                              showmeans=False, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_alpha(0.45)
        ax.boxplot(valid_data, positions=valid_pos,
                   widths=0.18, showfliers=False, patch_artist=False,
                   boxprops=dict(color="black", linewidth=1),
                   medianprops=dict(color="black", linewidth=1.2),
                   whiskerprops=dict(color="black"),
                   capprops=dict(color="black"))
        ax.set_xticks(range(1, len(cluster_ids) + 1))
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_title(label + (f"  (skipped {n_empty} empty clusters)" if n_empty else ""),
                     fontsize=11)
        ax.set_ylabel(field, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle(f"{target_tag} — per-cluster confidence distributions",
                 fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                pad_inches=0.4, facecolor="white")
    plt.close(fig)


# ─── Variant × cluster provenance (multi-variant only) ───────────────────


def plot_variant_cluster_breakdown(
    clustering_csv: Path,
    out_csv: Path,
    out_png: Path,
    target_tag: str,
) -> None:
    """
    Multi-variant runs only. Reads clustering.csv, parses each structure
    name (variant suffix), and emits:

      - <out_csv>: cross-tab of variant rows × cluster cols (counts).
        Plus a 'fraction_of_cluster' table (each cell / column total)
        for enrichment reading.
      - <out_png>: 2-panel heatmap (counts + fraction_of_cluster).
        Lets you spot which variants dominate which clusters at a glance.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if not clustering_csv.is_file():
        return
    df = pd.read_csv(clustering_csv)
    if "structure" not in df.columns and "structure_name" in df.columns:
        df = df.rename(columns={"structure_name": "structure"})
    if "cluster" not in df.columns:
        return

    # Parse variant from each name. If none have a variant suffix, skip.
    variants = []
    for name in df["structure"].astype(str):
        try:
            v, _m, _s, _i = parse_name(name)
        except ValueError:
            v = None
        variants.append(v if v is not None else "(none)")
    df["variant"] = variants

    if df["variant"].nunique() < 2:
        return  # not multi-variant

    counts = pd.crosstab(df["variant"], df["cluster"])
    counts = counts.reindex(sorted(counts.index), axis=0)
    counts = counts.reindex(sorted(counts.columns), axis=1)
    frac = counts.div(counts.sum(axis=0), axis=1).fillna(0.0)

    # Combined CSV (counts + fractions interleaved? simpler: two CSVs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_csv)
    frac_csv = out_csv.with_name(out_csv.stem.replace("_counts", "") + "_frac.csv")
    frac.to_csv(frac_csv)

    n_var = len(counts.index)
    n_clu = len(counts.columns)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(6, 0.45 * n_clu + 3), max(4, 0.42 * n_var + 2)),
        constrained_layout=True,
    )

    def _heat(ax, mat, title, fmt="{:.0f}"):
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(n_clu))
        ax.set_xticklabels([f"C{c}\n(n={int(counts[c].sum())})"
                            for c in mat.columns], fontsize=7)
        ax.set_yticks(range(n_var))
        ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_xlabel("cluster", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        # cell labels
        for i in range(n_var):
            for j in range(n_clu):
                v = mat.iat[i, j]
                if v > 0:
                    ax.text(j, i, fmt.format(v),
                            ha="center", va="center",
                            color="white" if v > mat.values.max() * 0.5 else "black",
                            fontsize=6)

    _heat(axes[0], counts, "structures per (variant, cluster) — counts")
    _heat(axes[1], frac, "fraction within each cluster (column-normalised)",
          fmt="{:.2f}")

    fig.suptitle(f"{target_tag} — variant × cluster", fontsize=11, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                pad_inches=0.4, facecolor="white")
    plt.close(fig)


# ─── (msa, template, model) provenance ─────────────────────────────────────


def _split_variant_msa_template(variant: str) -> tuple[str, str]:
    """Split a variant directory name into (msa_or_prefix, template_tag).

    Heuristic: '<X>_template_<Y>' → (X, Y); else (variant, 'default').
    Examples (H1311):
      af3_msa                       → ('af3_msa', 'default')
      mmseqs2_gpu                   → ('mmseqs2_gpu', 'default')
      tcrm_msa                      → ('tcrm_msa', 'default')
      AB_CDE_template_AB1_CDE1      → ('AB_CDE', 'AB1_CDE1')
      monomer_template_test         → ('monomer', 'test')
    """
    if "_template_" in variant:
        a, b = variant.split("_template_", 1)
        return a, b
    return variant, "default"


def plot_msa_template_model_breakdown(
    clustering_csv: Path,
    out_csv: Path,
    out_png: Path,
    target_tag: str,
) -> None:
    """Per-cluster (msa_or_prefix, template, model) breakdown.

    Decomposes each structure's filename into (variant, model), splits
    variant into (msa_prefix, template) on '_template_', and emits a
    heatmap of (msa+template+model) rows × cluster cols (counts +
    column-normalised fractions).

    Skips silently if clustering_csv is missing, has no variant suffix
    in any name, or yields fewer than 2 distinct keys.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if not clustering_csv.is_file():
        return
    df = pd.read_csv(clustering_csv)
    if "structure" not in df.columns and "structure_name" in df.columns:
        df = df.rename(columns={"structure_name": "structure"})
    if "cluster" not in df.columns:
        return

    rows = []
    for name in df["structure"].astype(str):
        try:
            variant, model, _seed, _idx = parse_name(name)
        except ValueError:
            variant, model = None, None
        if variant is None or model is None:
            rows.append((None, None, None))
            continue
        msa, tmpl = _split_variant_msa_template(variant)
        rows.append((msa, tmpl, model))
    df["msa"], df["template"], df["model"] = zip(*rows)

    if df["msa"].isna().all():
        return  # no variant suffix anywhere
    df = df.dropna(subset=["msa", "template", "model"])
    df["combo"] = df["msa"] + "+" + df["template"] + "+" + df["model"]
    if df["combo"].nunique() < 2:
        return

    counts = pd.crosstab(df["combo"], df["cluster"])
    counts = counts.reindex(sorted(counts.index), axis=0)
    counts = counts.reindex(sorted(counts.columns), axis=1)
    frac = counts.div(counts.sum(axis=0), axis=1).fillna(0.0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_csv)
    frac_csv = out_csv.with_name(out_csv.stem.replace("_counts", "") + "_frac.csv")
    frac.to_csv(frac_csv)

    n_row = len(counts.index)
    n_clu = len(counts.columns)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(8, 0.5 * n_clu + 3), max(6, 0.28 * n_row + 2)),
        constrained_layout=True,
    )

    def _heat(ax, mat, title, fmt="{:.0f}"):
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(n_clu))
        ax.set_xticklabels([f"C{c}\n(n={int(counts[c].sum())})"
                            for c in mat.columns], fontsize=7)
        ax.set_yticks(range(n_row))
        ax.set_yticklabels(mat.index, fontsize=6)
        ax.set_xlabel("cluster", fontsize=9)
        ax.set_ylabel("msa+template+model", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        # cell labels (only non-zero, otherwise too crowded)
        vmax = mat.values.max() if mat.values.size else 1
        for i in range(n_row):
            for j in range(n_clu):
                v = mat.iat[i, j]
                if v > 0:
                    ax.text(j, i, fmt.format(v),
                            ha="center", va="center",
                            color="white" if v > vmax * 0.5 else "black",
                            fontsize=5)

    _heat(axes[0], counts, "structures per (msa+template+model, cluster) — counts")
    _heat(axes[1], frac, "fraction within each cluster (column-normalised)",
          fmt="{:.2f}")

    fig.suptitle(f"{target_tag} — msa × template × model × cluster",
                 fontsize=11, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                pad_inches=0.4, facecolor="white")
    plt.close(fig)


# ─── Network analysis (graph topology, community detection) ───────────────


def _run_network_analysis(
    matrix_path: Path,
    names_path: Path,
    clustering_csv: Path,
    method: str,
    out_dir: Path,
    target_tag: str,
) -> None:
    """Run residudes_qa.cluster.core.network on a clustering result.

    Converts tm-like similarity to distance (1-x) before passing.
    Skips betweenness (slow for N>5000). Saves JSON summary + PNG report.
    Skips silently if dependencies (networkx) are missing.

    Edge selection per metric:
      - tm-like   → threshold = 0.2 (i.e. TM > 0.8)
      - rmsd-like → adaptive KNN (auto, k≈28 for N=14925)
    """
    import json
    import numpy as np
    import pandas as pd

    try:
        from residudes_qa.cluster.core.network import (
            analyze_cluster_network,
            compute_cluster_transition_matrix,
        )
        from residudes_qa.cluster.visualization.network_viz import (
            create_network_report,
        )
    except Exception as exc:
        # networkx/igraph/leidenalg not installed — non-fatal but log so
        # the user can spot why network/ output is missing.
        _log(f"[network] skipped: dependency missing ({exc!r})")
        return

    if not matrix_path.is_file() or not clustering_csv.is_file():
        return

    M = np.load(matrix_path).astype(np.float32)
    # Convert similarity to distance for tm-like metrics. anchored-rmsd /
    # drmsd are already distances.
    is_tm = method in ("tm", "itm", "anchored-tm")
    if is_tm:
        M = 1.0 - M
        np.fill_diagonal(M, 0.0)

    df = pd.read_csv(clustering_csv)
    if "structure" not in df.columns and "structure_name" in df.columns:
        df = df.rename(columns={"structure_name": "structure"})
    label_col = next(
        (c for c in ("cluster", "tm_cluster", "itm_cluster", "labels")
         if c in df.columns),
        None,
    )
    if label_col is None:
        return
    labels = df[label_col].values

    names = None
    if names_path.is_file():
        names = np.load(names_path, allow_pickle=True).tolist()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Edge selection: adaptive KNN (auto-k = 2*log2(N)) for all metrics.
    # KNN depends on per-row distance ranking, not absolute scale, so the
    # same logic works for TM-distance (1-TM in [0,1]) and physical-distance
    # metrics (dRMSD in Å). The previous TM-mode default (threshold=0.2,
    # i.e. TM > 0.8) produced near-complete graphs on tight ensembles
    # (density up to 1.0 on Matrix 2 A1B1 af3_msa pool) and collapsed
    # Leiden modularity to ~0. Sweep on real data (k_selection_sweep.py)
    # confirmed 2*log2(N) is the sweet spot for connectivity vs modularity.
    result = analyze_cluster_network(
        distance_matrix=M,
        labels=labels,
        structure_names=names,
        compute_betweenness=False,   # too slow for N>5000
        n_hubs=10,
        community_method="auto",       # Leiden if available, else Louvain
    )

    # Inter-cluster transition matrix (mean distance from cluster i to j).
    transition = compute_cluster_transition_matrix(M, labels)

    # Also compute the other community method on the same graph. We write
    # each method's result to a SEPARATE summary file so each is easy to
    # read on its own. Graph-level stats (n_nodes, n_edges, density, hubs)
    # are duplicated into each file so each is self-contained.
    from residudes_qa.cluster.core.network import (
        build_structure_graph, _detect_communities,
    )
    _alt_method = "louvain" if result.community_method == "leiden" else "leiden"
    _G_compare = build_structure_graph(M, structure_names=names)
    try:
        _alt_comms, _alt_mod, _alt_used = _detect_communities(
            _G_compare, method=_alt_method
        )
    except Exception as exc:
        _log(f"[network] {_alt_method} secondary detection failed: {exc!r}")
        _alt_comms = None
        _alt_mod = None
        _alt_used = _alt_method

    def _sizes_top10(comms):
        if comms is None:
            return None
        s = np.bincount(comms[comms >= 0]).tolist()
        return sorted(s, reverse=True)[:10]

    _graph_stats = {
        "n_nodes": int(result.n_nodes),
        "n_edges": int(result.n_edges),
        "density": float(result.density),
        "graph_components": int(result.metadata.get("graph_components", -1)),
        "hub_indices": [int(i) for i in result.hub_indices],
        "hub_names": (
            [names[i] for i in result.hub_indices] if names else None
        ),
    }
    # Primary (Leiden by default)
    summary = {
        **_graph_stats,
        "community_method": result.community_method,
        "n_communities": (
            int(len(np.unique(result.communities)))
            if result.communities is not None else None
        ),
        "modularity": (
            float(result.modularity)
            if result.modularity is not None else None
        ),
        "community_sizes_top10": _sizes_top10(result.communities),
    }
    # Filenames live inside <run>/results/network/, which already encodes
    # target/sto/method — no need for a target_tag prefix on each file.
    # Primary file (Leiden by default) keeps the legacy name `summary.json`
    # so downstream tooling stays compatible.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(transition).to_csv(out_dir / "transition.csv", index=False)

    # Side-by-side comparison file for the OTHER community algorithm.
    # Self-contained: includes the same graph stats so it reads on its own.
    alt_summary = {
        **_graph_stats,
        "community_method": _alt_used,
        "n_communities": (
            int(len(np.unique(_alt_comms))) if _alt_comms is not None else None
        ),
        "modularity": float(_alt_mod) if _alt_mod is not None else None,
        "community_sizes_top10": _sizes_top10(_alt_comms),
    }
    (out_dir / f"summary_{_alt_used}.json").write_text(
        json.dumps(alt_summary, indent=2)
    )

    # Plot the primary (Leiden) communities. Plotting both methods would
    # double runtime — the JSON is enough for quantitative comparison.
    try:
        fig = create_network_report(
            distance_matrix=M,
            labels=labels,
            network_result=result,
            transition_matrix=transition,
            target_name=target_tag,
        )
        fig.savefig(out_dir / "report.png",
                    dpi=120, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as exc:
        # Plot failure non-fatal — summary JSON still written. But log so
        # the user can spot it instead of silently missing a figure.
        _log(f"[network] report.png failed: {exc!r}")

    # Pretty community graph: communities-as-nodes view, grouped by dominant
    # predictor (parsed from structure name). For metrics with too many
    # micro-communities (e.g. dRMSD ~thousands of comms), only the top 20
    # are shown for readability.
    try:
        from residudes_qa.cluster.visualization.network_viz import (
            create_pretty_community_graph,
        )
        from residudes_qa.cluster.core.network import build_structure_graph
        # Rebuild kNN graph for inter-community edges (analyze_cluster_network
        # doesn't return the raw graph object).
        G_pretty = build_structure_graph(M, structure_names=names)
        n_comm = int(len(np.unique(result.communities)))
        # Auto top_k: cap at 20 if there are many micro-communities
        top_k = 20 if n_comm > 25 else None
        fig_pretty = create_pretty_community_graph(
            graph=G_pretty,
            communities=result.communities,
            structure_names=names if names else
                [f'node_{i}' for i in range(int(result.n_nodes))],
            top_k=top_k,
            title=(f'{target_tag}\n{n_comm} {result.community_method} '
                   f'communities, modularity {result.modularity:.3f}'),
        )
        fig_pretty.savefig(out_dir / "community_graph_pretty.png",
                           dpi=150, bbox_inches="tight", facecolor="white")
        import matplotlib.pyplot as plt
        plt.close(fig_pretty)
    except Exception as exc:
        _log(f"[network] community_graph_pretty.png failed: {exc!r}")

    # Medoid face renders — Cα-trace panels per cluster + per community.
    # Pulls PDBs from the run's _pdb_view/ directory (same one consensus
    # builder uses). Auto-skips if _pdb_view is missing.
    try:
        from residudes_qa.cluster.visualization.medoid_faces import (
            render_medoid_faces, parse_chain_spec,
        )
        # _pdb_view is at <run>/_pdb_view (sibling of results/)
        pdb_view = out_dir.parent.parent / "_pdb_view"
        if not pdb_view.is_dir():
            _log(f"[medoid_faces] skipping: _pdb_view not found at {pdb_view}")
        else:
            # Detect anchor + probe chains from the run dir name. Works for
            # both simple specs ('anchor-CDE_probe-AB') and mode-suffixed
            # ('anchor-AB-framework_probe-AB-cdr') thanks to parse_chain_spec.
            run_name = out_dir.parent.name
            anchor_chains: tuple = ()
            probe_chains: tuple | None = None
            if 'anchor-' in run_name:
                # 'anchored-rmsd_anchor-AB-framework_probe-AB-cdr'
                #                 ↑           split by '_probe-'
                tail = run_name.split('anchor-', 1)[1]
                if '_probe-' in tail:
                    anchor_spec, probe_spec = tail.split('_probe-', 1)
                else:
                    anchor_spec, probe_spec = tail, ''
                anchor_chains = parse_chain_spec(anchor_spec)
                if probe_spec:
                    probe_chains = parse_chain_spec(probe_spec)
            # (a) K-cluster medoids
            try:
                fig_clu = render_medoid_faces(
                    distance_matrix=M, labels=labels,
                    structure_names=names if names else
                        [f'node_{i}' for i in range(int(result.n_nodes))],
                    pdb_view_dir=pdb_view,
                    anchor_chains=anchor_chains,
                    probe_chains=probe_chains,
                    top_k=24,
                    title=f'{target_tag} — cluster medoids ({len(np.unique(labels))} clusters)',
                )
                fig_clu.savefig(out_dir / "cluster_medoid_faces.png",
                                dpi=140, bbox_inches="tight", facecolor="white")
                import matplotlib.pyplot as plt
                plt.close(fig_clu)
            except Exception as exc:
                _log(f"[medoid_faces] cluster_medoid_faces.png failed: {exc!r}")
            # (b) Leiden community medoids
            try:
                fig_com = render_medoid_faces(
                    distance_matrix=M, labels=result.communities,
                    structure_names=names if names else
                        [f'node_{i}' for i in range(int(result.n_nodes))],
                    pdb_view_dir=pdb_view,
                    anchor_chains=anchor_chains,
                    probe_chains=probe_chains,
                    top_k=24,
                    title=(f'{target_tag} — Leiden community medoids '
                           f'({len(np.unique(result.communities))} comms)'),
                )
                fig_com.savefig(out_dir / "community_medoid_faces.png",
                                dpi=140, bbox_inches="tight", facecolor="white")
                plt.close(fig_com)
            except Exception as exc:
                _log(f"[medoid_faces] community_medoid_faces.png failed: {exc!r}")
    except Exception as exc:
        _log(f"[medoid_faces] outer failure: {exc!r}")


# ─── Per-model summary CSV ────────────────────────────────────────────────


def _build_model_summary_csv(
    run_dir: Path,
    clustering_csv: Path,
    ranking_csv: Path,
    anchor_probe_json: Path,
    pdb_view: Path | None,
    out_csv: Path,
) -> None:
    """One row per structure:
        model | variant | cluster_id | cluster_size_ranking |
        ptm | iptm | cdr_plddt_mean | confidence_score

    `cluster_size_ranking` = 1 for the largest cluster, 2 for next, etc.
    `cdr_plddt_mean` is the CA B-factor mean over chains A & B at the
    residue indices stored in anchor_probe.json (skips E peptide).
    Skips silently if any input is missing.
    """
    import json
    import pandas as pd
    import numpy as np
    import gemmi

    if not clustering_csv.is_file() or not ranking_csv.is_file():
        return
    if not anchor_probe_json.is_file():
        return
    rk = pd.read_csv(ranking_csv)
    cl = pd.read_csv(clustering_csv)
    if "structure" not in cl.columns and "structure_name" in cl.columns:
        cl = cl.rename(columns={"structure_name": "structure"})
    if "cluster" not in cl.columns:
        return

    probe = json.loads(anchor_probe_json.read_text()).get(
        "probe_residues_per_chain", {}
    )
    if not probe:
        rk["cdr_plddt_mean"] = float("nan")
    elif pdb_view is None or not pdb_view.is_dir():
        rk["cdr_plddt_mean"] = float("nan")
    else:
        plddt = []
        for fname in rk["file"]:
            p = pdb_view / fname
            if not p.is_file():
                plddt.append(float("nan"))
                continue
            try:
                st = gemmi.read_structure(str(p))
            except Exception:
                plddt.append(float("nan"))
                continue
            vals = []
            for chain in st[0]:
                if chain.name not in ("A", "B"):
                    continue
                if chain.name not in probe:
                    continue
                idxs = set(probe[chain.name])
                for i, res in enumerate(chain):
                    if i not in idxs:
                        continue
                    for atom in res:
                        if atom.name == "CA":
                            vals.append(atom.b_iso)
                            break
            plddt.append(float(np.mean(vals)) if vals else float("nan"))
        rk["cdr_plddt_mean"] = plddt

    merged = rk.merge(
        cl[["structure", "cluster"]],
        left_on="file", right_on="structure", how="left",
    )

    # cluster_size_ranking: 1 = largest cluster, 2 = next, ... (ties broken
    # by lower cluster_id first). Members of a given cluster all share the
    # same rank.
    cluster_sizes = merged["cluster"].value_counts(dropna=False)
    rank_map = {
        cid: r for r, cid in enumerate(
            cluster_sizes.sort_values(ascending=False).index, start=1
        )
    }
    merged["cluster_size_ranking"] = merged["cluster"].map(rank_map)

    out_df = pd.DataFrame({
        "model":                merged["file"],
        "cluster_id":           merged["cluster"].astype("Int64"),
        "cluster_size_ranking": merged["cluster_size_ranking"].astype("Int64"),
        "ptm":                  merged.get("ptm"),
        "iptm":                 merged.get("iptm"),
        "cdr_plddt_mean":       merged["cdr_plddt_mean"],
        "confidence_score":     merged.get("ranking_score"),
    })
    if "variant" in merged.columns:
        out_df.insert(1, "variant", merged["variant"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    # Parquet sibling so the XGBDimerScorer bridge (`join-xgb`) can consume this
    # model-level summary directly. Non-fatal if pyarrow is unavailable.
    try:
        out_df.to_parquet(out_csv.with_suffix(".parquet"), index=False)
    except Exception:
        pass


# ─── PDB exports for medoids/consensus ─────────────────────────────────────


def export_medoid_inputs(
    items: list[tuple[str, str, int, int, Path]],
    keys_in_matrix_order: list[str],
    out_dir: Path,
    exclude_indices: dict[str, set[int]] | None,
    multi_variant: bool,
) -> Path:
    """
    `massivecluster comprehensive --pdbs` needs a flat directory of PDB files
    matching `structure_names.npy` (it does consensus alignment + medoid copy
    from there). We create that view ON DEMAND from the original CIFs:
        if --cdr: write filtered minimal PDBs (CDR residues removed)
        else:     symlink the CIF as <name>.pdb (gemmi reads either format,
                  but massivecluster's downstream consensus code expects PDB
                  text -> we still need to convert).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {key_for(it, multi_variant): it[4] for it in items}

    from tqdm import tqdm
    todo = []
    for key in keys_in_matrix_order:
        cif = by_key.get(key)
        if cif is None:
            continue
        pdb_out = out_dir / key
        if pdb_out.is_file() and pdb_out.stat().st_size > 0:
            continue
        todo.append((key, cif, pdb_out))

    if not todo:
        return out_dir

    for _key, cif, pdb_out in tqdm(todo, desc="Exporting CIF->PDB for consensus"):
        try:
            st = gemmi.read_structure(str(cif))
        except (IndexError, RuntimeError, ValueError, PermissionError, FileNotFoundError, OSError):
            continue  # skip empty/corrupted/unreadable cifs
        st.setup_entities()
        st.remove_ligands_and_waters()
        if exclude_indices:
            for model in st:
                for chain in model:
                    excl = exclude_indices.get(chain.name) or set()
                    if not excl:
                        continue
                    for i in range(len(chain) - 1, -1, -1):
                        if i in excl:
                            del chain[i]
        st.write_minimal_pdb(str(pdb_out))
    return out_dir


# ─── Main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("target_dir", type=Path,
                    help="CASP target directory, e.g. targets/H1311")
    ap.add_argument("--sto", required=True,
                    help="Stoichiometry, e.g. A1B1, C1D1, A1B1C1D1E1")
    ap.add_argument("--method", required=True,
                    choices=["tm", "itm", "drmsd", "anchored-rmsd",
                             "anchored-tm", "com", "tcr-angle",
                             "jaccard-epitope", "jaccard-paratope",
                             "jaccard-contact"],
                    help="Distance metric. tm/itm/drmsd: standard "
                         "(alignment region = measurement region). "
                         "jaccard-epitope/-paratope/-contact: binding-mode "
                         "distance = 1 - Jaccard over interface residue sets "
                         "(epitope=Ag side, paratope=Ab side) or contact pairs "
                         "(-contact = Ag+Ab simultaneously). Needs --ag-chains "
                         "+ --ab-chains. Already a distance (no 1-x). "
                         "anchored-rmsd / anchored-tm: Kabsch on --anchor-chains, "
                         "RMSD or TM-score on --probe-chains in that frame. "
                         "com: anchored centre-of-mass — Kabsch on --anchor-chains, "
                         "then L2 distance on per-probe-chain Cα COMs. "
                         "tcr-angle: wrap-safe L2 on (sin,cos) of TCR docking "
                         "crossing+incident angles. Loads Pierce CSV "
                         "(--tcr-angle-csv); auto-invokes Pierce via "
                         ".venv-cdr/bin/python if CSV missing.")
    ap.add_argument("--tcr-angle-csv", default=None,
                    help="Pierce angle CSV (per --method tcr-angle). If the "
                         "file exists it is consumed; otherwise Pierce is "
                         "invoked to produce it here. Format matches "
                         "target_notes/<target>/analysis_outputs/<target>_angles_pierce.csv.")
    ap.add_argument("--tcr-angle-recompute", action="store_true",
                    help="Force re-running Pierce even if --tcr-angle-csv exists.")
    ap.add_argument("--tcr-angle-group-filter", default=None,
                    help="Restrict CSV rows to this group_dir (e.g. 'af3_msa').")
    ap.add_argument("--tcr-angle-chain-layout", default="ours_5chain",
                    choices=["ours_5chain", "ours_4chain"],
                    help="Chain layout for Pierce (see tcr3d_pierce.compute_angles).")
    ap.add_argument("--tcr-angle-target", default=None,
                    help="Optional HLA-offset key (e.g. '9RU5').")
    ap.add_argument("--tcr-angle-workers", type=int, default=16,
                    help="Parallel workers for Pierce auto-compute.")
    ap.add_argument("--cdr", action="store_true",
                    help="Exclude IMGT CDR1/2/3 residues per chain (ANARCI). "
                         "Chains where ANARCI finds no antibody/TCR domain "
                         "are kept intact. Only valid for tm/itm/drmsd.")
    ap.add_argument("--anchor-chains", default=None,
                    help="Chain set used for Kabsch alignment in anchored modes "
                         "(e.g. 'CD' for chains C and D combined; 'C,D' also OK). "
                         "Required for --method anchored-rmsd / anchored-tm.")
    ap.add_argument("--probe-chains", default=None,
                    help="Chain set whose RMSD or TM-score is measured after "
                         "anchor alignment (e.g. 'E'). Required for "
                         "--method anchored-*. May reuse --anchor-chains "
                         "if --anchor-mode/--probe-mode select disjoint "
                         "residue subsets (e.g. framework vs cdr).")
    ap.add_argument("--anchor-mode", default="all",
                    choices=["all", "framework", "cdr", "cdr1", "cdr2", "cdr3"],
                    help="Residue subset within --anchor-chains. "
                         "'all' (default): entire chain. "
                         "'framework': chain minus IMGT CDR1/2/3 (ANARCI). "
                         "'cdr': all IMGT CDR. 'cdr1'/'cdr2'/'cdr3': "
                         "individual loops. Non-'all' requires "
                         "bash analysis/setup.sh --with-cdr.")
    ap.add_argument("--probe-mode", default="all",
                    choices=["all", "framework", "cdr", "cdr1", "cdr2", "cdr3"],
                    help="Residue subset within --probe-chains; same options "
                         "as --anchor-mode.")
    ap.add_argument("--anchor-spec", default=None,
                    help="Mixed-mode spec, replaces --anchor-chains/--anchor-mode. "
                         "Format: 'chains[:mode](+chains[:mode])*'. Examples: "
                         "'CD' (= --anchor-chains CD), 'AB:cdr+E' (AB CDR + E all), "
                         "'A:cdr1+B:cdr3+E'. Mutually exclusive with --anchor-chains.")
    ap.add_argument("--probe-spec", default=None,
                    help="Same syntax as --anchor-spec, applied to probe.")
    ap.add_argument("--ag-chains", default=None,
                    help="Antigen (epitope) chains for --method jaccard-* "
                         "(e.g. 'AB' or 'A,B'). The epitope = Ag residues "
                         "contacted by any --ab-chains residue.")
    ap.add_argument("--ab-chains", default=None,
                    help="Antibody (paratope) chains for --method jaccard-* "
                         "(e.g. 'CD' or 'C,D').")
    ap.add_argument("--iface-cutoff", type=float, default=5.0,
                    help="Heavy-atom contact cutoff (Å) for jaccard-* "
                         "interface extraction (default 5.0).")
    ap.add_argument("--variant", default="af3_msa",
                    help="MSA variant subdirectory (default: af3_msa). "
                         "Use --variants to pool multiple variants.")
    ap.add_argument("--variants", default=None,
                    help="Pool predictions from multiple variants (comma-list "
                         "or 'all'). Each prediction's filename gets a "
                         "'<variant>@@' prefix so provenance can attribute "
                         "it. Mutually exclusive with --variant. "
                         "Examples: 'af3_msa,mmseqs2_gpu', 'all'.")
    ap.add_argument("--flat-dir", type=Path, default=None,
                    help="Cluster a flat directory of structures "
                         "(e.g. MassiveFold's <T>_all_pdbs/), bypassing the "
                         "in-house <sto>/<variant>/<runner>/seed_*/sample_*/ "
                         "tree. Files must be named "
                         "'Model_<idx>_<engine>_..._pred_<i>.{pdb,cif}'. "
                         "Mutually exclusive with --variant / --variants.")
    ap.add_argument("--flat-ranking", type=Path, default=None,
                    help="Path to a ranking CSV with a `mapped_name` column "
                         "(e.g. MassiveFold's `ranking_<TARGET>.csv`). "
                         "Auto-detected as <flat-dir>/ranking_<TARGET>.csv "
                         "if exactly one such file is found in --flat-dir.")
    ap.add_argument("--flat-tag", default=None,
                    help="Tag used as the `<variant>` segment in the output "
                         "path (e.g. `massivefold` -> "
                         "`cluster_results/<sto>/massivefold/<method>/`). "
                         "Default: basename of --flat-dir.")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU device id for matrix computation (default: 0)")
    ap.add_argument("--cluster-method", default="fuzzy",
                    choices=["fuzzy", "kmeans", "hierarchical", "leiden"],
                    help="Clustering method passed to massivecluster (default: fuzzy)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse cached matrix + medoid PDB exports on re-run")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="If structures under this (sto, variant) have different "
                         "chain compositions (i.e. data mix-up), keep only the "
                         "dominant shape and continue. Default: abort with a "
                         "diagnostic table.")
    ap.add_argument("--no-medoid-pdbs", action="store_true",
                    help="Skip CIF->PDB export needed for consensus/medoid output. "
                         "Faster, but skips per-cluster consensus structures + medoid PDBs.")
    ap.add_argument("--keep-pdb-view", action="store_true",
                    help="Keep the intermediate _pdb_view/ folder (full PDB pool used "
                         "for medoid extraction). Default: delete after representatives/ "
                         "are written, since _pdb_view can be many GB.")
    ap.add_argument("--target-name", default=None,
                    help="Override target name (default: target_dir.name)")
    ap.add_argument("--output-base", default=None, type=Path,
                    help="Override output root (default: <target>/cluster_results/). "
                         "Used to redirect to e.g. target_notes/<TARGET>/clustering/ "
                         "for curated analysis archives. Final layout: "
                         "<output-base>/<sto>/<variant>/<method>/.")
    ap.add_argument("--list", action="store_true",
                    help="Just list discovered structures and exit")
    ap.add_argument("--allow-self-overlap", action="store_true",
                    help="Allow anchor and probe to use identical residue sets "
                         "(e.g. anchor=AB probe=AB to compute chain-restricted "
                         "TM/RMSD for self-comparison).")
    args = ap.parse_args(argv)

    # Anchored modes: validate flags up front (before any IO).
    # 'com' is anchored too — same anchor/probe semantics as anchored-rmsd.
    is_anchored = args.method in ("anchored-rmsd", "anchored-tm", "com")
    if args.method in ("jaccard-epitope", "jaccard-paratope", "jaccard-contact"):
        if not args.ag_chains or not args.ab_chains:
            ap.error(f"--method {args.method} requires --ag-chains and --ab-chains")
        if args.cdr:
            ap.error("--cdr is not supported with jaccard-* methods")
    anchor_groups: list[tuple[set[str], str]] | None = None
    probe_groups: list[tuple[set[str], str]] | None = None
    if is_anchored:
        # Spec ↔ chains/mode are mutually exclusive
        if args.anchor_spec and args.anchor_chains:
            ap.error("--anchor-spec and --anchor-chains are mutually exclusive")
        if args.probe_spec and args.probe_chains:
            ap.error("--probe-spec and --probe-chains are mutually exclusive")
        # Need one form on each side
        if not args.anchor_spec and not args.anchor_chains:
            ap.error(f"--method {args.method} requires --anchor-spec or --anchor-chains")
        if not args.probe_spec and not args.probe_chains:
            ap.error(f"--method {args.method} requires --probe-spec or --probe-chains")
        if args.cdr:
            ap.error("--cdr is not supported with anchored-* methods "
                     "(anchor/probe selection already controls residue scope)")
        # Parse both sides into uniform group lists.
        try:
            anchor_groups = (
                parse_chain_spec(args.anchor_spec) if args.anchor_spec
                else [(set(re.findall(r"[A-Z]", args.anchor_chains)), args.anchor_mode)]
            )
            probe_groups = (
                parse_chain_spec(args.probe_spec) if args.probe_spec
                else [(set(re.findall(r"[A-Z]", args.probe_chains)), args.probe_mode)]
            )
        except ValueError as exc:
            ap.error(str(exc))
        # Empty-group safety
        if not anchor_groups or not probe_groups:
            ap.error(f"empty anchor or probe spec")
        for chains, _ in anchor_groups + probe_groups:
            if not chains:
                ap.error(f"a group has no chains in spec")

        # Disjoint check at the chain level only when modes are also same
        # for every chain that appears on both sides. The detailed
        # residue-level disjoint check happens after ANARCI runs.
        anchor_chain_set = set().union(*[g[0] for g in anchor_groups])
        probe_chain_set = set().union(*[g[0] for g in probe_groups])
        if (anchor_chain_set & probe_chain_set
                and len(anchor_groups) == 1 and len(probe_groups) == 1
                and anchor_groups[0][1] == probe_groups[0][1]):
            inter = sorted(anchor_chain_set & probe_chain_set)
            if args.allow_self_overlap:
                _log(
                    f"note: anchor and probe both use chains {inter} with same mode "
                    f"({anchor_groups[0][1]!r}); self-overlap allowed via "
                    "--allow-self-overlap (chain-restricted self comparison)."
                )
            else:
                ap.error(
                    f"anchor and probe both use chains {inter} with the same mode "
                    f"({anchor_groups[0][1]!r}) → residue sets would overlap. "
                    "Use different modes (e.g. framework vs cdr), use disjoint chains, "
                    "or pass --allow-self-overlap for chain-restricted self comparison."
                )
    elif args.anchor_chains or args.probe_chains or args.anchor_spec or args.probe_spec:
        _log("note: anchor/probe flags ignored for non-anchored method")
    elif args.anchor_mode != "all" or args.probe_mode != "all":
        _log("note: --anchor-mode / --probe-mode ignored for non-anchored method")

    target_dir = args.target_dir.resolve()
    if not target_dir.is_dir():
        ap.error(f"not a directory: {target_dir}")
    target = args.target_name or target_dir.name

    # In --flat-dir mode the models come from the external bundle, so an
    # in-house results/<sto>/ tree is not required. Fall back to the target
    # dir (base_label "complex") when it's absent instead of bailing out.
    if args.flat_dir is not None:
        try:
            base = resolve_base_dir(target_dir, args.sto)
        except SystemExit:
            base = target_dir
    else:
        base = resolve_base_dir(target_dir, args.sto)
    base_label = base.name if base != target_dir else "complex"

    # Resolve --variant / --variants / --flat-dir → list[str] + items.
    # Three mutually-exclusive input modes:
    #   1. --flat-dir <path>     -> external flat bundle (e.g. MassiveFold)
    #   2. --variants <list|all> -> pool multiple in-house variant subdirs
    #   3. (default) --variant   -> single in-house variant subdir
    flat_mode = args.flat_dir is not None
    if flat_mode:
        if args.variants:
            ap.error("--flat-dir and --variants are mutually exclusive")
        if args.variant != "af3_msa":
            ap.error("--flat-dir and --variant are mutually exclusive")
        flat_tag = args.flat_tag or args.flat_dir.name
        # Auto-detect MassiveFold-style ranking_<TARGET>.csv when not given.
        if args.flat_ranking is None:
            candidates = [
                p for p in args.flat_dir.glob("ranking_[A-Z]*.csv")
                if "_debug" not in p.name and "_iptm" not in p.name
                and "_ptm" not in p.name and "_af3" not in p.name
            ]
            if len(candidates) == 1:
                args.flat_ranking = candidates[0]
                _log(f"  auto-detected --flat-ranking: {args.flat_ranking.name}")
            elif len(candidates) == 0:
                ap.error(
                    f"could not auto-detect ranking CSV in {args.flat_dir}; "
                    f"supply --flat-ranking PATH (need a ranking CSV with a "
                    f"`mapped_name` column)"
                )
            else:
                ap.error(
                    f"multiple candidate ranking CSVs in {args.flat_dir}: "
                    f"{[p.name for p in candidates]}; supply --flat-ranking PATH"
                )
        variants_list = [flat_tag]
        multi_variant = False
        variant_tag_for_path = flat_tag
    else:
        flat_tag = None
        if args.variant != "af3_msa" and args.variants:
            ap.error("--variant and --variants are mutually exclusive")
        if args.variants:
            if args.variants == "all":
                variants_list = list_available_variants(base, args.sto)
                if not variants_list:
                    _log(f"no variant subdirs under {base}/results/{args.sto}/")
                    return 1
            else:
                variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
        else:
            variants_list = [args.variant]
        multi_variant = len(variants_list) > 1
        if not multi_variant:
            variant_tag_for_path = variants_list[0]
        elif len(variants_list) > 5:
            # 6+ variants — too long to enumerate in a path. Use 'all' (full
            # list goes into variants.txt sidecar so info isn't lost).
            variant_tag_for_path = "all"
        else:
            # 2-5 variants: stable 8-hex hash of sorted variant names. Short
            # and collision-safe; variants.txt sidecar still carries the names.
            import hashlib
            h = hashlib.sha1(",".join(sorted(variants_list)).encode()).hexdigest()[:8]
            variant_tag_for_path = f"pool-{h}"

    if flat_mode:
        items = discover_flat_predictions(args.flat_dir, flat_tag)
        if not items:
            _log(f"no Model_<idx>_*.{{pdb,cif}} found under {args.flat_dir}")
            return 1
    else:
        items = discover_predictions(base, args.sto, variants_list)
        if not items:
            _log(f"no model.cif found under {base}/results/{args.sto}/ for variants={variants_list}")
            return 1

    by_variant_model: dict[tuple[str, str], int] = {}
    for v, m, _, _, _, _ in items:
        by_variant_model[(v, m)] = by_variant_model.get((v, m), 0) + 1
    mode_tag = ""
    if args.cdr:
        mode_tag = "  (CDR-excluded)"
    elif is_anchored:
        a_disp = spec_to_tag(anchor_groups) if anchor_groups else (args.anchor_spec or args.anchor_chains)
        p_disp = spec_to_tag(probe_groups) if probe_groups else (args.probe_spec or args.probe_chains)
        mode_tag = f"  (anchor={a_disp}, probe={p_disp})"
    _log(f"target={target}  base={base_label}  sto={args.sto}  "
         f"variants={variants_list}  method={args.method}{mode_tag}")
    if multi_variant:
        # show per-variant counts
        per_variant = {}
        for v, m, _, _, _, _ in items:
            per_variant.setdefault(v, {})[m] = per_variant.setdefault(v, {}).get(m, 0) + 1
        _log(f"  pooled N={len(items)} across {len(variants_list)} variants:")
        for v in variants_list:
            counts = per_variant.get(v, {})
            n = sum(counts.values())
            mod = ", ".join(f"{m}:{c}" for m, c in sorted(counts.items()))
            _log(f"    {v}: N={n} ({mod})")
    else:
        # In-house mode: `m` is the runner (af3/bt2/of3/pt2) — small, useful.
        # Flat mode: `m` is the per-file stem (1 unique value per file —
        # explodes to 14k entries in the log). Group by parsed method via
        # the provenance regex so the breakdown is bounded (~14 buckets for
        # MassiveFold, the parameter sets) and informative.
        if flat_mode:
            from residudes_qa.cluster.analysis.provenance import parse_method
            by_method: dict[str, int] = {}
            for it in items:
                key = key_for(it, multi_variant)
                meth = parse_method(key)
                by_method[meth] = by_method.get(meth, 0) + 1
            _log(f"  pooled N={len(items)} structures (" +
                 ", ".join(f"{m}:{n}" for m, n in sorted(by_method.items())) + ")")
        else:
            by_model = {}
            for v, m, _, _, _, _ in items:
                by_model[m] = by_model.get(m, 0) + 1
            _log(f"  pooled N={len(items)} structures (" +
                 ", ".join(f"{m}:{n}" for m, n in sorted(by_model.items())) + ")")

    # Sequence-consistency guard: aborts on chain/length mismatch unless
    # --allow-mismatch given. Without this, padded GPU kernels happily
    # produce nonsense matrices for mixed inputs.
    # --list is advertised as "no work, no Slurm": return before the
    # consistency guard, which reads EVERY structure (14k gemmi parses on a
    # MassiveFold pool) and would otherwise hammer the login node.
    if args.list:
        return 0

    canonical_shape = verify_sequence_consistency(items, args.allow_mismatch)

    # Output dir layout
    if is_anchored:
        anchor_tag = spec_to_tag(anchor_groups)
        probe_tag = spec_to_tag(probe_groups)
        suffix = f"{args.method}_anchor-{anchor_tag}_probe-{probe_tag}"
    elif args.cdr:
        suffix = f"{args.method}_cdr"
    else:
        suffix = args.method
    # --output-base overrides the default <target>/cluster_results/ location.
    # Used to redirect outputs to e.g. target_notes/<TARGET>/clustering/ for
    # curated analysis archives.
    if args.output_base is not None:
        if base == target_dir:
            out_root = args.output_base / args.sto / variant_tag_for_path / suffix
        else:
            out_root = args.output_base / base.name / args.sto / variant_tag_for_path / suffix
    elif base == target_dir:
        out_root = target_dir / "cluster_results" / args.sto / variant_tag_for_path / suffix
    else:
        out_root = target_dir / "cluster_results" / base.name / args.sto / variant_tag_for_path / suffix
    matrices_dir = out_root / "matrices"
    ranking_csv = out_root / "ranking.csv"
    results_dir = out_root / "results"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    # Multi-variant traceability: save full variant list as a sidecar so the
    # short path (all / pool-<hash>) doesn't lose information.
    if multi_variant:
        (matrices_dir / "variants.txt").write_text(
            "\n".join(variants_list) + "\n"
        )

    # Residue selection (depends on mode):
    #   --cdr           -> exclude_indices = ANARCI CDR set
    #   anchored-*      -> anchor_residues + probe_residues (KEEP sets)
    exclude_indices: dict[str, set[int]] | None = None
    anchor_residues: dict[str, set[int]] | None = None
    probe_residues: dict[str, set[int]] | None = None

    if args.cdr:
        rep_cif = items[0][4]
        seqs = get_chain_sequences(rep_cif)
        cdr_map = detect_cdr_residues(seqs)
        exclude_indices = {ch: set(idxs) for ch, idxs in cdr_map.items()}
        (matrices_dir / "cdr_residues.json").write_text(
            json.dumps(
                {ch: sorted(list(idx)) for ch, idx in exclude_indices.items()},
                indent=2,
            )
        )
        _log("  ANARCI CDR detection (IMGT scheme):")
        for ch, seq in seqs.items():
            n_cdr = len(exclude_indices.get(ch, set()))
            tag = "no antibody/TCR domain found (kept intact)" if n_cdr == 0 else f"{n_cdr} CDR residues excluded"
            _log(f"    chain {ch} (L={len(seq)}): {tag}")
    elif is_anchored:
        rep_cif = items[0][4]
        seqs = get_chain_sequences(rep_cif)

        all_groups = list(anchor_groups) + list(probe_groups)
        all_chains = set().union(*[g[0] for g in all_groups])
        missing = all_chains - seqs.keys()
        if missing:
            raise SystemExit(
                f"chain(s) not in structure: {sorted(missing)}, "
                f"available={sorted(seqs.keys())}"
            )

        # Run ANARCI only if any group needs CDR / framework info
        cdr_modes = {"framework", "cdr", "cdr1", "cdr2", "cdr3"}
        chains_needing_anarci = {
            ch for chains, mode in all_groups if mode in cdr_modes for ch in chains
        }
        cdr_map: dict[str, dict[str, list[int]]] = {}
        if chains_needing_anarci:
            sub_seqs = {ch: seqs[ch] for ch in chains_needing_anarci}
            cdr_map = detect_cdr_residues(sub_seqs)

        def _build_keep_from_groups(
            groups: list[tuple[set[str], str]],
        ) -> dict[str, set[int]]:
            """Union of (chain, mode) keep-sets per chain.
            For 'res=<ranges>' mode, ranges are 1-based residue numbers and
            converted to 0-based in-chain indices here."""
            out: dict[str, set[int]] = {}
            for chains, mode in groups:
                for ch in sorted(chains):
                    L = len(seqs[ch])
                    cmap = cdr_map.get(ch, {})
                    cdr_all = (set(cmap.get("cdr1", []))
                               | set(cmap.get("cdr2", []))
                               | set(cmap.get("cdr3", [])))
                    if mode == "all":
                        add = set(range(L))
                    elif mode == "framework":
                        add = set(range(L)) - cdr_all
                    elif mode == "cdr":
                        add = cdr_all
                    elif mode in ("cdr1", "cdr2", "cdr3"):
                        add = set(cmap.get(mode, []))
                    elif mode.startswith("res="):
                        ranges = _parse_residue_ranges(mode[len("res="):])
                        add = {r - 1 for r in ranges if 0 <= r - 1 < L}
                    else:
                        raise SystemExit(f"internal: unknown mode {mode!r}")
                    out.setdefault(ch, set()).update(add)
            return out

        anchor_residues = _build_keep_from_groups(anchor_groups)
        probe_residues = _build_keep_from_groups(probe_groups)

        # Residue-level disjoint check (covers e.g. AB:cdr1 vs AB:cdr3 OK,
        # but AB:cdr3+E vs AB:cdr3 would overlap → error)
        for ch in anchor_residues.keys() & probe_residues.keys():
            inter = anchor_residues[ch] & probe_residues[ch]
            if inter:
                if args.allow_self_overlap:
                    _log(
                        f"note: chain {ch}: anchor and probe residue sets overlap "
                        f"({len(inter)} residues); allowed via --allow-self-overlap."
                    )
                else:
                    raise SystemExit(
                        f"chain {ch}: anchor and probe residue sets overlap "
                        f"({len(inter)} residues). Check anchor/probe specs, or "
                        "pass --allow-self-overlap for chain-restricted self comparison."
                    )

        # Sanity: enough residues to be meaningful.
        n_anchor_total = sum(len(v) for v in anchor_residues.values())
        n_probe_total = sum(len(v) for v in probe_residues.values())
        if n_anchor_total < 3:
            raise SystemExit(
                f"anchor has {n_anchor_total} residues — need >= 3 for Kabsch. "
                f"Check anchor groups: {anchor_groups}"
            )
        if n_probe_total < 1:
            raise SystemExit(
                f"probe has {n_probe_total} residues — nothing to measure. "
                f"Check probe groups: {probe_groups}"
            )

        anchor_tag_for_log = spec_to_tag(anchor_groups)
        probe_tag_for_log = spec_to_tag(probe_groups)
        # method may be 'anchored-rmsd', 'anchored-tm', or 'com' (no dash)
        _log(f"  {args.method}:  "
             f"anchor=[{anchor_tag_for_log}] ({n_anchor_total} residues),  "
             f"probe=[{probe_tag_for_log}] ({n_probe_total} residues)")
        if cdr_map:
            for ch, per in sorted(cdr_map.items()):
                cdr1n, cdr2n, cdr3n = len(per.get("cdr1", [])), len(per.get("cdr2", [])), len(per.get("cdr3", []))
                _log(f"    chain {ch} CDR (IMGT): cdr1={cdr1n}, cdr2={cdr2n}, cdr3={cdr3n} residues")

        (matrices_dir / "anchor_probe.json").write_text(
            json.dumps({
                "anchor_groups": [
                    {"chains": sorted(g[0]), "mode": g[1]} for g in anchor_groups
                ],
                "probe_groups": [
                    {"chains": sorted(g[0]), "mode": g[1]} for g in probe_groups
                ],
                "anchor_residue_count": n_anchor_total,
                "probe_residue_count": n_probe_total,
                "anchor_residues_per_chain": {
                    ch: sorted(list(idx)) for ch, idx in anchor_residues.items()
                },
                "probe_residues_per_chain": {
                    ch: sorted(list(idx)) for ch, idx in probe_residues.items()
                },
                "cdr_residues_imgt": {
                    ch: {k: sorted(list(v)) for k, v in per.items()}
                    for ch, per in cdr_map.items()
                },
            }, indent=2)
        )

    # Build the input list for the GPU kernels — pass CIFs directly. We
    # *override* the names that get persisted so the matrix index uses the
    # provenance-friendly form (af3_seedX_sampleY.pdb) instead of "model.cif".
    # The kernels call Path(name).name internally, so we hand them a list of
    # CIF paths but rewrite `names` after the fact.
    cif_paths = [it[4] for it in items]
    # Drop 0-byte CIFs (failed inference). Empty files crash the gemmi/PDB
    # loader path with an opaque IndexError or anchor=0 ValueError.
    _n_before = len(items)
    def _readable_nonempty(p):
        try:
            return p.stat().st_size > 0 and os.access(p, os.R_OK)
        except OSError:
            return False
    _keep = [i for i, p in enumerate(cif_paths) if _readable_nonempty(p)]
    if len(_keep) < _n_before:
        _dropped = _n_before - len(_keep)
        items = [items[i] for i in _keep]
        cif_paths = [cif_paths[i] for i in _keep]
        _log(f"  filtered out {_dropped} empty (0-byte) CIFs — proceeding with N={len(items)}")
    matrix_path = matrices_dir / f"{args.method}_matrix.npy"
    names_path = matrices_dir / "structure_names.npy"

    cached = (
        args.skip_existing
        and matrix_path.is_file()
        and names_path.is_file()
        and (matrices_dir / ("cdr_residues.json" if args.cdr else "_dummy")).is_file()
        != bool(args.cdr)  # presence of cdr_residues.json must match args.cdr
    )

    if args.skip_existing and matrix_path.is_file() and names_path.is_file():
        _log(f"  reusing cached matrix at {matrix_path}")
    else:
        # Pre-derive structure_names (provenance-friendly keys) so we can
        # pass them to tcr-angle which needs to match Pierce CSV by name.
        keys_pre = [key_for(it, multi_variant) for it in items]
        _, raw_names = compute_matrix(
            args.method, cif_paths, matrix_path, args.gpu, exclude_indices,
            anchor_residues=anchor_residues,
            probe_residues=probe_residues,
            structure_names=keys_pre,
            ag_chains=([c for c in args.ag_chains.replace(",", "")]
                       if getattr(args, "ag_chains", None) else None),
            ab_chains=([c for c in args.ab_chains.replace(",", "")]
                       if getattr(args, "ab_chains", None) else None),
            iface_cutoff=getattr(args, "iface_cutoff", 5.0),
            tcr_angle_csv=args.tcr_angle_csv,
            tcr_angle_recompute=args.tcr_angle_recompute,
            tcr_angle_group_filter=args.tcr_angle_group_filter,
            tcr_angle_chain_layout=args.tcr_angle_chain_layout,
            tcr_angle_target=args.tcr_angle_target,
            tcr_angle_workers=args.tcr_angle_workers,
        )
        # Most kernels keep all rows. tcr-angle may DROP rows that lack a
        # CSV entry — in that case raw_names is the filtered subset and we
        # save THAT as names_path (not the original keys_pre).
        if args.method == "tcr-angle":
            keys = list(raw_names)
        else:
            keys = keys_pre
            if len(keys) != len(raw_names):
                _log(f"  ! name count mismatch ({len(keys)} != {len(raw_names)})")
        np.save(names_path, np.array(keys))

    # Load names back so that ranking + medoid export use exactly what the
    # matrix indexes.
    keys_in_matrix_order = list(np.load(names_path, allow_pickle=True))

    # Build ranking.csv. In flat mode, point at the MassiveFold-style
    # `ranking_<TARGET>.csv`; otherwise walk per-prediction summary.json.
    build_ranking_csv(items, keys_in_matrix_order, ranking_csv,
                      multi_variant=multi_variant,
                      flat_ranking_path=args.flat_ranking if flat_mode else None,
                      flat_tag=flat_tag)

    # Optional: export PDB view for consensus/medoid extraction.
    pdbs_dir: Path | None = None
    if not args.no_medoid_pdbs:
        pdb_view = out_root / "_pdb_view"
        export_medoid_inputs(items, keys_in_matrix_order, pdb_view,
                             exclude_indices, multi_variant=multi_variant)
        pdbs_dir = pdb_view

    # Comprehensive
    if is_anchored:
        target_tag = f"{target}_{args.sto}_{args.method}_anchor-{anchor_tag}_probe-{probe_tag}"
    else:
        target_tag = f"{target}_{args.sto}_{args.method}{'_cdr' if args.cdr else ''}"
    rc = run_comprehensive(
        matrix_path=matrix_path,
        method=args.method,
        names_path=names_path,
        pdbs_dir=pdbs_dir,
        ranking_csv=ranking_csv,
        out_dir=results_dir,
        target_tag=target_tag,
        cluster_method=args.cluster_method,
    )
    if rc != 0:
        _log(f"  ! massivecluster comprehensive failed (rc={rc})")
        return rc

    # Post-comprehensive: per-cluster confidence distribution figure.
    # Uses the metric-tagged clustering CSV that comprehensive wrote.
    # Label must match the one passed to `--matrix <label>:...` (see
    # run_comprehensive's label_map), since comprehensive prefixes outputs
    # with that label.
    label_map = {
        "tm": "tm", "itm": "itm", "drmsd": "drmsd",
        "anchored-tm": "tm", "anchored-rmsd": "drmsd",
    }
    clustering_label = label_map.get(args.method, args.method)
    clustering_csv = results_dir / f"{target_tag}_{clustering_label}_clustering.csv"
    try:
        conf_png = results_dir / f"{target_tag}_confidence_dist.png"
        plot_confidence_distributions(
            ranking_csv=ranking_csv,
            clustering_csv=clustering_csv,
            out_path=conf_png,
            target_tag=target_tag,
        )
        if conf_png.is_file():
            _log(f"      confidence dist: {conf_png.name}")
    except Exception as exc:
        _log(f"  ! confidence-dist plot failed (non-fatal): {exc}")

    # 3D COM scatter (com method only) — probe COM in anchor frame with
    # anchor backbone drawn. Cluster + provenance views.
    if args.method == 'com':
        try:
            from residudes_qa.cluster.visualization.com_3d import render_com_3d
            com3d_path = matrices_dir / 'com_matrix_com3d.npy'
            names_path_full = matrices_dir / 'com_matrix_names.npy'
            if com3d_path.is_file() and names_path_full.is_file():
                _com3d = np.load(com3d_path)
                _names_short = np.load(names_path_full, allow_pickle=True)
                anchor_chains_tuple = tuple(sorted(anchor_residues.keys())) if anchor_residues else ()
                render_com_3d(
                    com3d=_com3d,
                    names=list(_names_short),
                    clustering_csv=clustering_csv,
                    anchor_cif=Path(cif_paths[0]),
                    anchor_chains=anchor_chains_tuple,
                    out_dir=results_dir,
                    target_tag=target_tag,
                )
        except Exception as exc:
            _log(f'  ! com 3D viz failed (non-fatal): {exc!r}')

    # Variant × cluster heatmap (multi-variant runs only).
    if multi_variant:
        try:
            vc_csv = results_dir / f"{target_tag}_variant_cluster_counts.csv"
            vc_png = results_dir / f"{target_tag}_variant_cluster.png"
            plot_variant_cluster_breakdown(
                clustering_csv=clustering_csv,
                out_csv=vc_csv,
                out_png=vc_png,
                target_tag=target_tag,
            )
            if vc_png.is_file():
                _log(f"      variant×cluster: {vc_png.name}")
        except Exception as exc:
            _log(f"  ! variant×cluster plot failed (non-fatal): {exc}")

        # (msa, template, model) × cluster — finer-grained provenance.
        try:
            mtm_csv = results_dir / f"{target_tag}_msa_template_model_counts.csv"
            mtm_png = results_dir / f"{target_tag}_msa_template_model.png"
            plot_msa_template_model_breakdown(
                clustering_csv=clustering_csv,
                out_csv=mtm_csv,
                out_png=mtm_png,
                target_tag=target_tag,
            )
            if mtm_png.is_file():
                _log(f"      msa×template×model: {mtm_png.name}")
        except Exception as exc:
            _log(f"  ! msa×template×model plot failed (non-fatal): {exc}")

    # Network analysis (graph topology, community detection, transitions).
    # Skipped silently if networkx not installed.
    try:
        _run_network_analysis(
            matrix_path=matrix_path,
            names_path=names_path,
            clustering_csv=clustering_csv,
            method=args.method,
            out_dir=results_dir / "network",
            target_tag=target_tag,
        )
        net_png = results_dir / "network" / "report.png"
        if net_png.is_file():
            _log(f"      network report: {net_png.relative_to(results_dir)}")
    except Exception as exc:
        _log(f"  ! network analysis failed (non-fatal): {exc}")

    # Per-model summary CSV: model_id | variant | cluster | ptm | iptm |
    # cdr_plddt_mean | confidence_score. CDR pLDDT is averaged from B-factor
    # over the AB CDR residues recorded in matrices/anchor_probe.json.
    try:
        summary_csv = results_dir / f"{target_tag}_model_clustering_confidence.csv"
        _build_model_summary_csv(
            run_dir=out_root,
            clustering_csv=clustering_csv,
            ranking_csv=ranking_csv,
            anchor_probe_json=matrices_dir / "anchor_probe.json",
            pdb_view=pdbs_dir,
            out_csv=summary_csv,
        )
        if summary_csv.is_file():
            _log(f"      model summary: {summary_csv.name}")
    except Exception as exc:
        _log(f"  ! model summary CSV failed (non-fatal): {exc}")

    # Strip target_tag prefix from results/ entries — the parent path already
    # encodes target/sto/variant/method, so the redundant prefix on every
    # filename is just noise.
    try:
        prefix = f"{target_tag}_"
        for entry in list(results_dir.iterdir()):
            if entry.name.startswith(prefix):
                stripped = entry.name[len(prefix):]
                if not stripped:
                    continue
                target_path = results_dir / stripped
                if target_path.exists() and target_path != entry:
                    continue  # collision; keep both as-is
                entry.rename(target_path)
    except Exception as exc:
        _log(f"  ! results/ rename pass failed (non-fatal): {exc}")

    _log("")
    _log(f"done. results: {results_dir}")
    _log(f"      matrices: {matrices_dir}")
    if pdbs_dir is not None:
        if args.keep_pdb_view:
            _log(f"      pdb view: {pdbs_dir}  (kept; --keep-pdb-view)")
        elif pdbs_dir.is_dir():
            try:
                shutil.rmtree(pdbs_dir)
                _log(f"      pdb view: removed {pdbs_dir} (use --keep-pdb-view to retain)")
            except Exception as exc:
                _log(f"      pdb view: cleanup failed ({exc}); kept at {pdbs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
