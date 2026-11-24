"""AF3 result directory discovery and file loading (unified output schema)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np


@dataclass
class SampleFiles:
    seed: int
    sample: int
    summary_path: Path       # sample_<i>/summary.json
    model_cif_path: Path     # sample_<i>/model.cif
    arrays_dir: Path         # sample_<i>/arrays/


@dataclass
class PredictionRun:
    label: str               # unique id: '{stoich}_{msa_variant}_{model}', e.g. 'A1_af3_msa_af3'
    af3_dir: Path
    is_complex: bool
    chain_labels: list[str]
    samples: list[SampleFiles]
    seed_dirs: list[Path] = field(default_factory=list)
    chain_letter: str = ""   # single-letter chain dir for monomers; "" for complex
    stoich: str = ""         # stoichiometry dir, e.g. 'A1', 'A1B1C1D1E1'
    msa_variant: str = ""    # MSA pipeline variant, e.g. 'af3_msa', 'mmseqs2_gpu', 'plmmsa_otalign'
    model: str = ""          # prediction model/tool, e.g. 'af3', 'bt2', 'of3', 'pt2'


def discover_prediction_types(target_dir: Path) -> list[PredictionRun]:
    """Scan target_dir and return one PredictionRun per (stoich, msa_variant, model) found.

    Discovery rules:
      - Monomer:  target_dir/{X}/results/{stoich}/{msa_variant}/{model}/   (X = single uppercase letter)
      - Complex:  target_dir/results/{stoich}/{msa_variant}/{model}/

    All MSA variants (af3_msa, mmseqs2_gpu, plmmsa_otalign, …) and all models
    (af3, bt2, of3, pt2, …) under each stoichiometry are discovered.
    Returns list sorted monomers first (alphabetical by chain then stoich), complex last.
    """
    target_dir = target_dir.resolve()
    runs: list[PredictionRun] = []

    # Monomer predictions: single-letter subdirs
    for subdir in sorted(target_dir.iterdir()):
        if not subdir.is_dir() or not re.match(r"^[A-Z]$", subdir.name):
            continue
        for tool_dir in sorted(subdir.glob("results/*/*/*")):
            if not tool_dir.is_dir():
                continue
            samples = _collect_samples(tool_dir)
            if not samples:
                continue
            stoich = tool_dir.parent.parent.name   # e.g. 'A1'
            msa_variant = tool_dir.parent.name      # e.g. 'af3_msa'
            model = tool_dir.name                   # e.g. 'af3'
            run = _build_run(
                label=f"{stoich}_{msa_variant}_{model}",
                af3_dir=tool_dir,
                samples=samples,
                chain_letter=subdir.name,
                stoich=stoich,
                msa_variant=msa_variant,
                model=model,
            )
            runs.append(run)

    # Complex predictions: target_dir/results/{stoich}/{msa_variant}/{model}/
    results_dir = target_dir / "results"
    if results_dir.is_dir():
        for tool_dir in sorted(results_dir.glob("*/*/*")):
            if not tool_dir.is_dir():
                continue
            stoich = tool_dir.parts[-3]
            msa_variant = tool_dir.parts[-2]
            model = tool_dir.name
            samples = _collect_samples(tool_dir)
            if not samples:
                continue
            run = _build_run(
                label=f"{stoich}_{msa_variant}_{model}",
                af3_dir=tool_dir,
                samples=samples,
                chain_letter="",
                stoich=stoich,
                msa_variant=msa_variant,
                model=model,
            )
            runs.append(run)

    if not runs:
        raise FileNotFoundError(
            f"No AF3 prediction types found under {target_dir}"
        )
    return runs


def _build_run(
    label: str,
    af3_dir: Path,
    samples: list[SampleFiles],
    chain_letter: str = "",
    stoich: str = "",
    msa_variant: str = "",
    model: str = "",
) -> PredictionRun:
    best = max(samples, key=lambda s: _quick_ranking_score(s))
    summary = load_summary_confidences(best.summary_path)
    is_complex = summary.get("iptm") is not None

    conf = load_confidences(best)
    chain_labels = _extract_chain_labels(conf)

    # Inner seed dirs: parent of sample_* dirs — distogram_probs.npy lives here.
    seed_dirs = sorted(
        {sf.model_cif_path.parent.parent for sf in samples},
        key=lambda p: int(p.name.split("_")[1]),
    )

    return PredictionRun(
        label=label,
        af3_dir=af3_dir,
        is_complex=is_complex,
        chain_labels=chain_labels,
        samples=samples,
        seed_dirs=seed_dirs,
        chain_letter=chain_letter,
        stoich=stoich,
        msa_variant=msa_variant,
        model=model,
    )


def _collect_samples(af3_dir: Path) -> list[SampleFiles]:
    """Walk af3_dir/seed_<N>/**/sample_<i>/ and build SampleFiles list.

    sample_* dirs may be directly under seed_N/ or nested one level deeper
    (e.g. seed_N/seed_N/sample_i/ or seed_N/<job_tag>/seed_N/sample_i/).
    """
    result: list[SampleFiles] = []
    for seed_top in sorted(af3_dir.glob("seed_*")):
        m = re.match(r"seed_(\d+)$", seed_top.name)
        if not m or not seed_top.is_dir():
            continue
        seed = int(m.group(1))
        for sample_dir in sorted(seed_top.glob("**/sample_*")):
            sm = re.match(r"sample_(\d+)$", sample_dir.name)
            if not sm or not sample_dir.is_dir():
                continue
            sample = int(sm.group(1))

            summary_path = sample_dir / "summary.json"
            cif_path = sample_dir / "model.cif"
            arrays_dir = sample_dir / "arrays"

            if not summary_path.exists() or not cif_path.exists() or not arrays_dir.is_dir():
                continue

            result.append(SampleFiles(
                seed=seed,
                sample=sample,
                summary_path=summary_path,
                model_cif_path=cif_path,
                arrays_dir=arrays_dir,
            ))

    return sorted(result, key=lambda s: (s.seed, s.sample))


def _quick_ranking_score(sf: SampleFiles) -> float:
    try:
        data = load_summary_confidences(sf.summary_path)
        return float(data.get("ranking_score", 0.0))
    except Exception:
        return 0.0


def _extract_chain_labels(conf: dict) -> list[str]:
    seen: list[str] = []
    for cid in conf.get("token_chain_ids", []):
        if cid not in seen:
            seen.append(cid)
    return seen


def _parse_cif_chain_info(
    cif_path: Path,
) -> tuple[list[str], list[int], list[str], list[tuple[int, int]]]:
    """Parse model.cif to get per-token and per-atom chain info.

    Returns:
        token_chain_ids:   one chain label per token (unique chain+residue pair)
        token_res_ids:     residue sequence number per token
        atom_chain_ids:    one chain label per atom
        token_atom_ranges: (atom_start, atom_end) index slice per token
    """
    st = gemmi.read_structure(str(cif_path))
    model_obj = st[0]

    token_chain_ids: list[str] = []
    token_res_ids: list[int] = []
    atom_chain_ids: list[str] = []
    token_atom_ranges: list[tuple[int, int]] = []

    atom_idx = 0
    seen: set[tuple[str, int]] = set()
    for chain in model_obj:
        for res in chain:
            key = (chain.name, res.seqid.num)
            n_res_atoms = sum(1 for _ in res)
            if key not in seen:
                seen.add(key)
                token_chain_ids.append(chain.name)
                token_res_ids.append(res.seqid.num)
                token_atom_ranges.append((atom_idx, atom_idx + n_res_atoms))
            else:
                # Alternate conformations — extend last token's atom range
                prev_start, _ = token_atom_ranges[-1]
                token_atom_ranges[-1] = (prev_start, atom_idx + n_res_atoms)
            for _ in res:
                atom_chain_ids.append(chain.name)
            atom_idx += n_res_atoms

    return token_chain_ids, token_res_ids, atom_chain_ids, token_atom_ranges


def _decode_logits_npz(npz_path: Path, key: str) -> np.ndarray:
    """Decode a quantized logits .npz (int8_softmax scheme) to expected values.

    The npz stores uint8 logits + JSON _meta with scale and bin_axis.
    Returns a 2D expected-value array (N, N).
    """
    d = np.load(npz_path)
    meta = json.loads(d["_meta"].tobytes())
    head_meta = meta["heads"][key]
    scale = float(head_meta["scale"])
    logits = d[key].astype(np.float32) * scale
    # stable softmax
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    # standard AF3 bins: 64 bins, 0–32 Å, centres at 0.25, 0.75, …, 31.75
    n_bins = logits.shape[-1]
    bin_width = 32.0 / n_bins
    bin_centers = np.arange(bin_width / 2, 32.0, bin_width, dtype=np.float32)
    return (probs * bin_centers).sum(axis=-1)


def load_confidences(sf: SampleFiles) -> dict:
    """Load per-sample confidence data from the unified output schema.

    Supports two array layouts:
      af3:           arrays/pae.npy, arrays/pde.npy  (float32, shape NxN)
      bt2/of3/pt2:   arrays/pae_logits.npz, arrays/pde_logits.npz  (int8_softmax)
    """
    token_chain_ids, token_res_ids, atom_chain_ids, token_atom_ranges = (
        _parse_cif_chain_info(sf.model_cif_path)
    )

    plddt_atom = np.load(sf.arrays_dir / "plddt_atom.npy")

    # PAE
    pae_npy = sf.arrays_dir / "pae.npy"
    pae_npz = sf.arrays_dir / "pae_logits.npz"
    if pae_npy.exists():
        pae = np.load(pae_npy)
    elif pae_npz.exists():
        pae = _decode_logits_npz(pae_npz, "pae")
    else:
        raise FileNotFoundError(f"No pae array found in {sf.arrays_dir}")

    # PDE
    pde_npy = sf.arrays_dir / "pde.npy"
    pde_npz = sf.arrays_dir / "pde_logits.npz"
    if pde_npy.exists():
        pde = np.load(pde_npy)
    elif pde_npz.exists():
        pde = _decode_logits_npz(pde_npz, "pde")
    else:
        pde = np.zeros_like(pae)

    contact_path = sf.arrays_dir / "contact_probs.npy"
    contact_probs = np.load(contact_path) if contact_path.exists() else np.zeros_like(pae)

    # Compute per-token pLDDT by averaging atom pLDDTs within each token
    n_tokens = len(token_chain_ids)
    token_plddts = np.zeros(n_tokens, dtype=np.float32)
    for i, (a_start, a_end) in enumerate(token_atom_ranges):
        end = min(a_end, len(plddt_atom))
        if end > a_start:
            token_plddts[i] = float(plddt_atom[a_start:end].mean())

    return {
        "token_chain_ids": token_chain_ids,
        "token_res_ids": token_res_ids,
        "atom_chain_ids": atom_chain_ids,
        "atom_plddts": plddt_atom.tolist(),
        "token_plddts": token_plddts,
        "pae": pae,
        "pde": pde,
        "contact_probs": contact_probs,
    }


def load_summary_confidences(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_ranking_scores(run: PredictionRun) -> list[dict]:
    """Read ranking_score from each sample's summary.json."""
    rows: list[dict] = []
    for sf in run.samples:
        try:
            data = load_summary_confidences(sf.summary_path)
            rows.append({
                "seed": sf.seed,
                "sample": sf.sample,
                "ranking_score": float(data.get("ranking_score", 0.0)),
            })
        except Exception:
            pass
    return sorted(rows, key=lambda r: (r["seed"], r["sample"]))


def find_best_sample(run: PredictionRun) -> SampleFiles:
    """Return SampleFiles with the globally highest ranking_score."""
    all_scores = load_all_ranking_scores(run)
    if not all_scores:
        return run.samples[0]
    best_row = max(all_scores, key=lambda r: r["ranking_score"])
    for sf in run.samples:
        if sf.seed == best_row["seed"] and sf.sample == best_row["sample"]:
            return sf
    return run.samples[0]


def get_msa_paths(run: PredictionRun, target_dir: Path) -> tuple[list[Path], list[int] | None]:
    """Return (a3m_paths, chain_boundaries) for MSA visualization.

    For monomers: returns paths to uniref90/mgnify/small_bfd/uniprot .a3m files.
    For complexes: returns paired MSA files with chain boundary positions.
    chain_boundaries is None for monomers, list of boundary token indices for complexes.
    """
    target_dir = target_dir.resolve()

    if not run.is_complex:
        chain_dir = target_dir / run.chain_letter / "msas" / "af3"
        a3m_names = ["uniref90.a3m", "mgnify.a3m", "small_bfd.a3m", "uniprot.a3m"]
        paths = [chain_dir / name for name in a3m_names if (chain_dir / name).exists()]
        return paths, None

    # Complex: use paired MSA files from this run's inputs directory
    # (inputs/{stoich}/{msa_variant}/), so each variant plots its own MSA.
    inputs_dir = target_dir / "inputs" / run.stoich / run.msa_variant
    if not inputs_dir.exists():
        return [], None

    paired_paths: list[Path] = []
    chain_boundary_positions: list[int] = []
    cumulative = 0
    for chain in run.chain_labels:
        paired = inputs_dir / f"paired_{chain}.a3m"
        if paired.exists():
            paired_paths.append(paired)
            seq_len = _get_a3m_seq_len(paired)
            if seq_len > 0:
                cumulative += seq_len
                chain_boundary_positions.append(cumulative)

    if not paired_paths:
        for chain in run.chain_labels:
            unpaired = inputs_dir / f"unpaired_{chain}.a3m"
            if unpaired.exists():
                paired_paths.append(unpaired)

    return paired_paths, chain_boundary_positions[:-1] if chain_boundary_positions else None


def find_chain_msa_a3ms(target_dir: Path, chain_letter: str) -> list[tuple[str, str, Path]]:
    """Return [(variant_rel_path, stem, a3m_path), ...] for all a3ms under {chain}/msas/.

    variant_rel_path is relative to the msas/ dir, supporting nested layouts such as
    'af3', 'mmseqs2_gpu', 'plmmsa/plmalign', etc.
    """
    result: list[tuple[str, str, Path]] = []
    msas_dir = target_dir / chain_letter / "msas"
    if not msas_dir.is_dir():
        return result
    for a3m in sorted(msas_dir.rglob("*.a3m")):
        if not a3m.is_file():
            continue
        variant = str(a3m.parent.relative_to(msas_dir))
        result.append((variant, a3m.stem, a3m))
    return result


def _get_a3m_seq_len(a3m_path: Path) -> int:
    with open(a3m_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq = "".join(c for c in line if c == "-" or c.isupper())
            return len(seq)
    return 0
