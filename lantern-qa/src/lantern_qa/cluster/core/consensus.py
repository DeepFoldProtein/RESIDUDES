"""
Consensus Structure Extraction Module

For each cluster, computes:
- Per-residue RMSF (structural variability)
- Conserved core (low-RMSF residues)
- Average/consensus coordinates
- Medoid as representative structure
- Local conserved segments (contiguous low-RMSF regions)
- Cross-cluster segment comparison

Usage:
    from lantern_qa.cluster.core.consensus import (
        extract_cluster_consensus,
        find_conserved_segments,
        compare_segments_across_clusters,
    )

    result = extract_cluster_consensus(pdb_paths=pdb_list, cluster_indices=[0, 3, 7, 12])
    segments = find_conserved_segments(result)
    for seg in segments:
        print(f"Segment {seg.segment_id}: res {seg.start}-{seg.end}, "
              f"RMSF={seg.mean_rmsf:.2f}, len={seg.length}")
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class ConsensusResult:
    """Container for consensus structure analysis."""
    cluster_id: int
    n_members: int
    n_residues: int
    # Per-residue metrics
    rmsf: np.ndarray               # (n_residues,) root mean square fluctuation
    mean_coords: np.ndarray         # (n_residues, 3) average coordinates after alignment
    conserved_mask: np.ndarray      # (n_residues,) bool — True if RMSF < threshold
    n_conserved: int
    conserved_fraction: float
    # Representative
    medoid_idx: int                 # index within cluster_indices
    medoid_name: Optional[str]
    # Thresholds
    rmsf_threshold: float
    mean_rmsf: float
    median_rmsf: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalSegment:
    """A contiguous conserved region within a cluster."""
    segment_id: int
    cluster_id: int
    start: int                      # start residue index (0-based)
    end: int                        # end residue index (exclusive)
    length: int
    # Structural metrics
    mean_rmsf: float
    max_rmsf: float
    rmsf_profile: np.ndarray        # (length,) per-residue RMSF within segment
    coords: np.ndarray              # (length, 3) consensus coordinates
    # Residue info (if available)
    residue_indices: List[int]      # original residue indices
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentComparisonResult:
    """Result of comparing conserved segments across clusters."""
    cluster_a: int
    cluster_b: int
    segment_a_id: int
    segment_b_id: int
    # Overlap
    overlap_start: int              # residue index where overlap starts
    overlap_end: int                # residue index where overlap ends
    overlap_length: int
    overlap_fraction_a: float       # fraction of segment A that overlaps
    overlap_fraction_b: float       # fraction of segment B that overlaps
    # Structural similarity within overlap
    rmsd: float                     # CA RMSD of overlapping region
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_pdb_ca(pdb_path: Union[str, Path]) -> np.ndarray:
    """Extract CA atom coordinates from PDB file."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float64)


def _parse_pdb_ca_with_info(pdb_path: Union[str, Path]):
    """Extract CA atoms with residue info for B-factor output."""
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                atoms.append({
                    'line': line,
                    'chain': line[21],
                    'resn': line[17:20].strip(),
                    'resi': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                })
    return atoms


def _kabsch_align(mobile: np.ndarray, target: np.ndarray,
                  mask: Optional[np.ndarray] = None):
    """
    Kabsch alignment of mobile onto target (both N×3).
    If mask is provided (bool array, length N), alignment is computed using
    only masked residues but applied to all residues (stable-core alignment).
    Returns aligned mobile coordinates (full length).
    """
    if mask is not None and mask.sum() >= 4:
        Pm = mobile[mask]
        Qm = target[mask]
    else:
        Pm = mobile
        Qm = target

    mobile_center = Pm.mean(axis=0)
    target_center = Qm.mean(axis=0)
    P = Pm - mobile_center
    Q = Qm - target_center

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    R = Vt.T @ sign_matrix @ U.T

    # Apply to full structure
    aligned = (mobile - mobile_center) @ R.T + target_center
    return aligned


def extract_cluster_consensus(
    pdb_paths: List[Union[str, Path]],
    cluster_indices: List[int],
    reference_idx: Optional[int] = None,
    rmsf_threshold: Optional[float] = None,
    max_structures: int = 200,
    cluster_id: int = 0,
) -> ConsensusResult:
    """
    Extract consensus structure information from a cluster.

    1. Load CA coordinates from cluster members
    2. Align all to reference (medoid) using Kabsch
    3. Compute per-residue RMSF
    4. Identify conserved core (RMSF < threshold)

    Args:
        pdb_paths: List of all PDB file paths
        cluster_indices: Indices of structures in this cluster
        reference_idx: Index of reference structure (medoid). If None, uses first.
        rmsf_threshold: RMSF cutoff for conserved residues (Angstroms).
                        If None, uses median RMSF.
        max_structures: Max structures to use (subsample if larger)
        cluster_id: Cluster identifier

    Returns:
        ConsensusResult with RMSF, conserved core, and mean coordinates
    """
    n_members = len(cluster_indices)

    # Subsample if too many
    if n_members > max_structures:
        rng = np.random.RandomState(42)
        selected = set(rng.choice(n_members, max_structures, replace=False).tolist())
        # Always include reference (reference_idx is a global index)
        if reference_idx is not None:
            # Find local position of reference in cluster_indices
            try:
                ref_local = cluster_indices.index(reference_idx)
                selected.add(ref_local)
            except ValueError:
                pass  # reference not in cluster
        cluster_indices = [cluster_indices[i] for i in sorted(selected)]
        n_members = len(cluster_indices)

    if reference_idx is None:
        reference_idx = cluster_indices[0]

    # Load reference
    ref_coords = _parse_pdb_ca(pdb_paths[reference_idx])
    n_residues = len(ref_coords)

    # Load and align all members
    aligned_coords = []  # list of (n_residues, 3)
    valid_indices = []

    for idx in cluster_indices:
        try:
            coords = _parse_pdb_ca(pdb_paths[idx])
            if len(coords) != n_residues:
                continue  # Skip mismatched lengths
            aligned = _kabsch_align(coords, ref_coords)
            aligned_coords.append(aligned)
            valid_indices.append(idx)
        except Exception:
            continue

    if len(aligned_coords) < 2:
        # Not enough structures
        return ConsensusResult(
            cluster_id=cluster_id,
            n_members=1,
            n_residues=n_residues,
            rmsf=np.zeros(n_residues),
            mean_coords=ref_coords,
            conserved_mask=np.ones(n_residues, dtype=bool),
            n_conserved=n_residues,
            conserved_fraction=1.0,
            medoid_idx=reference_idx,
            medoid_name=str(pdb_paths[reference_idx]),
            rmsf_threshold=0.0,
            mean_rmsf=0.0,
            median_rmsf=0.0,
        )

    all_coords = np.stack(aligned_coords)  # (M, N, 3)

    # ── Iterative Kabsch: refine alignment using conserved core ─────────────
    # Iteration 1: full alignment → initial RMSF → identify stable residues
    mean_coords = all_coords.mean(axis=0)
    deviations = all_coords - mean_coords[np.newaxis]
    rmsf_init = np.sqrt(np.mean(np.sum(deviations ** 2, axis=2), axis=0))

    # Iteration 2: re-align using stable residues (RMSF < min(5Å, 60th pct))
    rmsf_cutoff_iter = min(5.0, float(np.percentile(rmsf_init, 60)))
    stable_mask = rmsf_init < rmsf_cutoff_iter
    if stable_mask.sum() >= max(10, n_residues // 5):
        # Re-align EVERY member (kept in lockstep with valid_indices) to the
        # reference using only the stable residues. NB: do NOT substitute
        # ref_coords into slot 0 — when reference_idx != cluster_indices[0]
        # (the normal medoid-passing path) that desyncs coords from
        # valid_indices, drops the real member 0, and double-counts the
        # reference, corrupting the mean/RMSF and the medoid attribution.
        all_coords = np.stack([
            _kabsch_align(c, ref_coords, mask=stable_mask) for c in aligned_coords
        ])

    # Final mean coordinates and RMSF
    mean_coords = all_coords.mean(axis=0)  # (N, 3)

    # Per-residue RMSF
    # RMSF_i = sqrt( mean( |r_i - <r_i>|^2 ) )
    deviations = all_coords - mean_coords[np.newaxis, :, :]  # (M, N, 3)
    sq_dist = np.sum(deviations ** 2, axis=2)  # (M, N)
    rmsf = np.sqrt(np.mean(sq_dist, axis=0))  # (N,)

    # Conserved core
    if rmsf_threshold is None:
        rmsf_threshold = np.median(rmsf)

    conserved_mask = rmsf < rmsf_threshold
    n_conserved = int(conserved_mask.sum())

    # Find medoid within aligned structures (closest to mean)
    dists_to_mean = np.sqrt(np.sum((all_coords - mean_coords) ** 2, axis=(1, 2)))
    medoid_local = int(np.argmin(dists_to_mean))
    medoid_global = valid_indices[medoid_local]

    return ConsensusResult(
        cluster_id=cluster_id,
        n_members=len(aligned_coords),
        n_residues=n_residues,
        rmsf=rmsf,
        mean_coords=mean_coords,
        conserved_mask=conserved_mask,
        n_conserved=n_conserved,
        conserved_fraction=n_conserved / n_residues if n_residues > 0 else 0.0,
        medoid_idx=medoid_global,
        medoid_name=str(pdb_paths[medoid_global]),
        rmsf_threshold=rmsf_threshold,
        mean_rmsf=float(np.mean(rmsf)),
        median_rmsf=float(np.median(rmsf)),
    )


def write_consensus_pdb(
    consensus: ConsensusResult,
    template_pdb: Union[str, Path],
    output_path: Union[str, Path],
    color_by_rmsf: bool = True,
):
    """
    Write a PDB file with consensus coordinates and RMSF in B-factor column.

    Args:
        consensus: ConsensusResult from extract_cluster_consensus
        template_pdb: Template PDB for chain/residue info
        output_path: Output PDB path
        color_by_rmsf: If True, write RMSF as B-factor; else write conserved (0/1)
    """
    atoms = _parse_pdb_ca_with_info(template_pdb)

    if len(atoms) != consensus.n_residues:
        raise ValueError(
            f"Template has {len(atoms)} CA atoms but consensus has {consensus.n_residues}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    medoid_name = Path(consensus.medoid_name).stem if consensus.medoid_name else "unknown"

    with open(output_path, 'w') as f:
        f.write(f"REMARK   1  Consensus structure from cluster analysis\n")
        f.write(f"REMARK   2\n")
        f.write(f"REMARK   2  CLUSTER ID:          {consensus.cluster_id}\n")
        f.write(f"REMARK   2  CLUSTER MEMBERS:     {consensus.n_members}\n")
        f.write(f"REMARK   2  REFERENCE (MEDOID):   {medoid_name}\n")
        f.write(f"REMARK   2  TOTAL RESIDUES:       {consensus.n_residues}\n")
        f.write(f"REMARK   2\n")
        f.write(f"REMARK   3  RMSF STATISTICS\n")
        f.write(f"REMARK   3  MEAN RMSF:           {consensus.mean_rmsf:.3f} A\n")
        f.write(f"REMARK   3  MEDIAN RMSF:         {consensus.median_rmsf:.3f} A\n")
        f.write(f"REMARK   3  RMSF THRESHOLD:      {consensus.rmsf_threshold:.3f} A\n")
        f.write(f"REMARK   3  CONSERVED CORE:      {consensus.n_conserved}/{consensus.n_residues}"
                f" ({consensus.conserved_fraction:.1%})\n")
        f.write(f"REMARK   3\n")
        f.write(f"REMARK   4  COORDINATES ARE CONSENSUS (MEAN) AFTER KABSCH ALIGNMENT\n")
        f.write(f"REMARK   4  B-FACTOR COLUMN:     "
                f"{'RMSF (A)' if color_by_rmsf else 'CONSERVED (0/100)'}\n")

        for i, atom in enumerate(atoms):
            x, y, z = consensus.mean_coords[i]

            if color_by_rmsf:
                bfactor = min(consensus.rmsf[i], 99.99)
            else:
                bfactor = 0.0 if consensus.conserved_mask[i] else 100.0

            # PDB ATOM format
            line = (
                f"ATOM  {i+1:5d}  CA  {atom['resn']:>3s} {atom['chain']}"
                f"{atom['resi']:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00{bfactor:6.2f}\n"
            )
            f.write(line)

        f.write("END\n")


def extract_all_cluster_consensus(
    pdb_paths: List[Union[str, Path]],
    labels: np.ndarray,
    medoid_indices: Optional[List[int]] = None,
    rmsf_threshold: Optional[float] = None,
    max_structures_per_cluster: int = 200,
) -> List[ConsensusResult]:
    """
    Extract consensus for all clusters.

    Args:
        pdb_paths: All PDB paths
        labels: Cluster label for each structure
        medoid_indices: Medoid index per cluster (optional)
        rmsf_threshold: RMSF cutoff (None = auto per cluster)
        max_structures_per_cluster: Max members to align per cluster

    Returns:
        List of ConsensusResult, one per cluster
    """
    unique_labels = np.unique(labels)
    results = []

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue  # Skip noise

        cluster_idx = np.where(labels == label)[0].tolist()
        ref_idx = medoid_indices[i] if medoid_indices and i < len(medoid_indices) else None

        result = extract_cluster_consensus(
            pdb_paths=pdb_paths,
            cluster_indices=cluster_idx,
            reference_idx=ref_idx,
            rmsf_threshold=rmsf_threshold,
            max_structures=max_structures_per_cluster,
            cluster_id=int(label),
        )
        results.append(result)

    return results


# ============================================================================
# Local Substructure Extraction
# ============================================================================

def find_conserved_segments(
    consensus: ConsensusResult,
    min_segment_length: int = 5,
    smoothing_window: int = 5,
    rmsf_threshold: Optional[float] = None,
    gap_tolerance: int = 2,
) -> List[LocalSegment]:
    """
    Find contiguous conserved segments from per-residue RMSF profile.

    Applies optional smoothing, then identifies runs of low-RMSF residues.
    Short gaps (< gap_tolerance) within a conserved region are bridged.

    Args:
        consensus: ConsensusResult with RMSF and coordinates
        min_segment_length: Minimum segment length to report
        smoothing_window: Window size for moving-average smoothing (0=no smoothing)
        rmsf_threshold: RMSF cutoff (default: use consensus threshold)
        gap_tolerance: Bridge gaps shorter than this within conserved regions

    Returns:
        List of LocalSegment, sorted by start position
    """
    rmsf = consensus.rmsf.copy()
    n = len(rmsf)

    if n == 0:
        return []

    # Smooth RMSF with moving average
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        # Pad to avoid edge effects
        padded = np.pad(rmsf, smoothing_window // 2, mode='edge')
        rmsf_smooth = np.convolve(padded, kernel, mode='valid')[:n]
    else:
        rmsf_smooth = rmsf

    # Threshold
    if rmsf_threshold is None:
        rmsf_threshold = consensus.rmsf_threshold

    conserved = rmsf_smooth < rmsf_threshold

    # Bridge short gaps
    if gap_tolerance > 0:
        conserved = _bridge_gaps(conserved, gap_tolerance)

    # Find contiguous segments
    segments = []
    seg_id = 0
    i = 0
    while i < n:
        if conserved[i]:
            start = i
            while i < n and conserved[i]:
                i += 1
            end = i  # exclusive

            length = end - start
            if length >= min_segment_length:
                seg_rmsf = consensus.rmsf[start:end]
                seg_coords = consensus.mean_coords[start:end]

                segments.append(LocalSegment(
                    segment_id=seg_id,
                    cluster_id=consensus.cluster_id,
                    start=start,
                    end=end,
                    length=length,
                    mean_rmsf=float(np.mean(seg_rmsf)),
                    max_rmsf=float(np.max(seg_rmsf)),
                    rmsf_profile=seg_rmsf,
                    coords=seg_coords,
                    residue_indices=list(range(start, end)),
                ))
                seg_id += 1
        else:
            i += 1

    return segments


def _bridge_gaps(mask: np.ndarray, gap_tolerance: int) -> np.ndarray:
    """Bridge short False gaps within True regions."""
    result = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not result[i]:
            gap_start = i
            while i < n and not result[i]:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            # Bridge if gap is short and flanked by True on both sides
            if gap_len <= gap_tolerance and gap_start > 0 and gap_end < n:
                result[gap_start:gap_end] = True
        else:
            i += 1
    return result


def find_domain_boundaries(
    consensus: ConsensusResult,
    rmsf_jump_threshold: Optional[float] = None,
    min_domain_length: int = 20,
) -> List[int]:
    """
    Detect domain boundaries from RMSF profile.

    Boundaries are positions where RMSF changes sharply (high-variability
    linker regions between low-variability domains).

    Args:
        consensus: ConsensusResult with RMSF
        rmsf_jump_threshold: Minimum RMSF jump to call boundary.
                             Default: mean + 1.5*std of RMSF gradient.
        min_domain_length: Minimum domain size (residues)

    Returns:
        List of boundary residue indices
    """
    rmsf = consensus.rmsf
    n = len(rmsf)

    if n < 2 * min_domain_length:
        return []

    # Smooth first
    kernel = np.ones(7) / 7
    padded = np.pad(rmsf, 3, mode='edge')
    smooth = np.convolve(padded, kernel, mode='valid')[:n]

    # Compute gradient magnitude
    grad = np.abs(np.gradient(smooth))

    if rmsf_jump_threshold is None:
        rmsf_jump_threshold = np.mean(grad) + 1.5 * np.std(grad)

    # Find peaks in gradient (potential boundaries)
    peaks = []
    for i in range(1, n - 1):
        if grad[i] > rmsf_jump_threshold and grad[i] >= grad[i-1] and grad[i] >= grad[i+1]:
            peaks.append(i)

    # Filter: merge nearby peaks, enforce min domain length
    boundaries = []
    last_boundary = 0
    for p in peaks:
        if p - last_boundary >= min_domain_length:
            boundaries.append(p)
            last_boundary = p

    # Also ensure last domain is long enough
    if boundaries and (n - boundaries[-1]) < min_domain_length:
        boundaries.pop()

    return boundaries


def compare_segments_across_clusters(
    consensus_list: List[ConsensusResult],
    min_segment_length: int = 5,
    rmsf_threshold: Optional[float] = None,
    min_overlap: int = 5,
) -> List[SegmentComparisonResult]:
    """
    Compare conserved segments across clusters to find shared substructures.

    For each pair of clusters, find overlapping conserved segments and
    compute structural similarity (RMSD) within the overlap.

    Args:
        consensus_list: List of ConsensusResult from different clusters
        min_segment_length: Min segment length for detection
        rmsf_threshold: RMSF threshold (None = per-cluster auto)
        min_overlap: Minimum overlap length to report

    Returns:
        List of SegmentComparisonResult, sorted by RMSD
    """
    # Extract segments for each cluster
    all_segments = {}
    for cons in consensus_list:
        threshold = rmsf_threshold if rmsf_threshold is not None else None
        segs = find_conserved_segments(
            cons, min_segment_length=min_segment_length,
            rmsf_threshold=threshold,
        )
        all_segments[cons.cluster_id] = (cons, segs)

    comparisons = []
    cluster_ids = list(all_segments.keys())

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cid_a, cid_b = cluster_ids[i], cluster_ids[j]
            cons_a, segs_a = all_segments[cid_a]
            cons_b, segs_b = all_segments[cid_b]

            # Only compare if same number of residues (same protein)
            if cons_a.n_residues != cons_b.n_residues:
                continue

            for sa in segs_a:
                for sb in segs_b:
                    # Compute overlap
                    overlap_start = max(sa.start, sb.start)
                    overlap_end = min(sa.end, sb.end)
                    overlap_len = overlap_end - overlap_start

                    if overlap_len < min_overlap:
                        continue

                    # RMSD of overlapping consensus coordinates
                    coords_a = cons_a.mean_coords[overlap_start:overlap_end]
                    coords_b = cons_b.mean_coords[overlap_start:overlap_end]

                    # Align overlapping region
                    aligned_b = _kabsch_align(coords_b, coords_a)
                    rmsd = np.sqrt(np.mean(np.sum((coords_a - aligned_b) ** 2, axis=1)))

                    comparisons.append(SegmentComparisonResult(
                        cluster_a=cid_a,
                        cluster_b=cid_b,
                        segment_a_id=sa.segment_id,
                        segment_b_id=sb.segment_id,
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        overlap_length=overlap_len,
                        overlap_fraction_a=overlap_len / sa.length,
                        overlap_fraction_b=overlap_len / sb.length,
                        rmsd=float(rmsd),
                    ))

    # Sort by RMSD (most similar first)
    comparisons.sort(key=lambda x: x.rmsd)
    return comparisons


def write_segment_pdb(
    consensus: ConsensusResult,
    segment: LocalSegment,
    template_pdb: Union[str, Path],
    output_path: Union[str, Path],
):
    """
    Write a single conserved segment as a PDB file.

    B-factor column contains per-residue RMSF within the segment.

    Args:
        consensus: Parent ConsensusResult
        segment: LocalSegment to write
        template_pdb: Template PDB for residue info
        output_path: Output PDB path
    """
    atoms = _parse_pdb_ca_with_info(template_pdb)

    if len(atoms) != consensus.n_residues:
        raise ValueError(
            f"Template has {len(atoms)} CA atoms but consensus has {consensus.n_residues}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Residue range info from template
    start_atom = atoms[segment.start]
    end_atom = atoms[segment.end - 1]
    chain_id = start_atom['chain']
    start_resi = start_atom['resi']
    end_resi = end_atom['resi']
    medoid_name = Path(consensus.medoid_name).stem if consensus.medoid_name else "unknown"

    with open(output_path, 'w') as f:
        f.write(f"REMARK   1  Conserved local segment from cluster analysis\n")
        f.write(f"REMARK   2\n")
        f.write(f"REMARK   2  CLUSTER ID:        {consensus.cluster_id}\n")
        f.write(f"REMARK   2  CLUSTER MEMBERS:   {consensus.n_members}\n")
        f.write(f"REMARK   2  REFERENCE (MEDOID): {medoid_name}\n")
        f.write(f"REMARK   2\n")
        f.write(f"REMARK   3  SEGMENT INFO\n")
        f.write(f"REMARK   3  SEGMENT ID:        {segment.segment_id}\n")
        f.write(f"REMARK   3  CHAIN:             {chain_id}\n")
        f.write(f"REMARK   3  RESIDUE RANGE:     {start_resi}-{end_resi} "
                f"({segment.length} residues)\n")
        f.write(f"REMARK   3  MEAN RMSF:         {segment.mean_rmsf:.3f} A\n")
        f.write(f"REMARK   3  MAX RMSF:          {segment.max_rmsf:.3f} A\n")
        f.write(f"REMARK   3  RMSF THRESHOLD:    {consensus.rmsf_threshold:.3f} A\n")
        f.write(f"REMARK   3\n")
        f.write(f"REMARK   4  COORDINATES ARE CONSENSUS (MEAN) AFTER KABSCH ALIGNMENT\n")
        f.write(f"REMARK   4  B-FACTOR COLUMN CONTAINS PER-RESIDUE RMSF (ANGSTROMS)\n")

        atom_serial = 1
        for i in range(segment.start, segment.end):
            atom = atoms[i]
            x, y, z = consensus.mean_coords[i]
            bfactor = min(consensus.rmsf[i], 99.99)

            line = (
                f"ATOM  {atom_serial:5d}  CA  {atom['resn']:>3s} {atom['chain']}"
                f"{atom['resi']:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00{bfactor:6.2f}\n"
            )
            f.write(line)
            atom_serial += 1

        f.write("END\n")


def write_consensus_cif(
    consensus: ConsensusResult,
    template_pdb: Union[str, Path],
    output_path: Union[str, Path],
    target_name: str = "Unknown",
):
    """
    Write consensus structure as mmCIF with rich metadata.

    Includes cluster info, medoid reference, RMSF stats, and
    per-residue RMSF in the B_iso_or_equiv field.

    Args:
        consensus: ConsensusResult from extract_cluster_consensus
        template_pdb: Template PDB for chain/residue info
        output_path: Output .cif path
        target_name: Target identifier (e.g., T1212)
    """
    atoms = _parse_pdb_ca_with_info(template_pdb)

    if len(atoms) != consensus.n_residues:
        raise ValueError(
            f"Template has {len(atoms)} CA atoms but consensus has {consensus.n_residues}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    medoid_name = Path(consensus.medoid_name).stem if consensus.medoid_name else "unknown"

    with open(output_path, 'w') as f:
        entry_id = f"{target_name}_cluster{consensus.cluster_id}_consensus"
        f.write(f"data_{entry_id}\n#\n")

        # Entry
        f.write(f"_entry.id   '{entry_id}'\n#\n")

        # Cluster metadata
        f.write("_mc_cluster_info.cluster_id             {}\n".format(consensus.cluster_id))
        f.write("_mc_cluster_info.n_members              {}\n".format(consensus.n_members))
        f.write("_mc_cluster_info.n_residues             {}\n".format(consensus.n_residues))
        f.write("_mc_cluster_info.medoid_name            '{}'\n".format(medoid_name))
        f.write("_mc_cluster_info.target_name            '{}'\n".format(target_name))
        f.write("_mc_cluster_info.n_conserved            {}\n".format(consensus.n_conserved))
        f.write("_mc_cluster_info.conserved_fraction     {:.4f}\n".format(
            consensus.conserved_fraction))
        f.write("_mc_cluster_info.rmsf_threshold         {:.4f}\n".format(
            consensus.rmsf_threshold))
        f.write("_mc_cluster_info.mean_rmsf              {:.4f}\n".format(consensus.mean_rmsf))
        f.write("_mc_cluster_info.median_rmsf            {:.4f}\n".format(consensus.median_rmsf))
        f.write("_mc_cluster_info.coordinate_type        'consensus_mean'\n")
        f.write("_mc_cluster_info.alignment_method       'Kabsch'\n")
        f.write("_mc_cluster_info.biso_content           'per_residue_rmsf'\n")
        f.write("#\n")

        # atom_site loop
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.label_atom_id\n")
        f.write("_atom_site.label_alt_id\n")
        f.write("_atom_site.label_comp_id\n")
        f.write("_atom_site.label_asym_id\n")
        f.write("_atom_site.label_entity_id\n")
        f.write("_atom_site.label_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        f.write("_atom_site.occupancy\n")
        f.write("_atom_site.B_iso_or_equiv\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.pdbx_mc_rmsf\n")
        f.write("_atom_site.pdbx_mc_conserved\n")

        for i, atom in enumerate(atoms):
            x, y, z = consensus.mean_coords[i]
            rmsf_val = consensus.rmsf[i]
            bfactor = min(rmsf_val, 999.99)
            conserved = "Y" if consensus.conserved_mask[i] else "N"

            f.write(
                f"ATOM {i+1:5d} C CA . {atom['resn']:>3s} {atom['chain']} 1 "
                f"{atom['resi']:4d} ? "
                f"{x:8.3f} {y:8.3f} {z:8.3f} "
                f"1.00 {bfactor:6.2f} "
                f"{atom['resi']:4d} {atom['chain']} "
                f"{rmsf_val:.4f} {conserved}\n"
            )

        f.write("#\n")


def write_segment_cif(
    consensus: ConsensusResult,
    segment: LocalSegment,
    template_pdb: Union[str, Path],
    output_path: Union[str, Path],
    target_name: str = "Unknown",
):
    """
    Write a conserved segment as mmCIF with full provenance metadata.

    Metadata includes:
    - Which cluster this segment belongs to
    - Which medoid structure was used as reference
    - Residue range (original numbering from template)
    - Per-residue RMSF and conserved status
    - Alignment and consensus details

    Args:
        consensus: Parent ConsensusResult
        segment: LocalSegment to write
        template_pdb: Template PDB for chain/residue info
        output_path: Output .cif path
        target_name: Target identifier (e.g., T1212)
    """
    atoms = _parse_pdb_ca_with_info(template_pdb)

    if len(atoms) != consensus.n_residues:
        raise ValueError(
            f"Template has {len(atoms)} CA atoms but consensus has {consensus.n_residues}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    medoid_name = Path(consensus.medoid_name).stem if consensus.medoid_name else "unknown"
    start_atom = atoms[segment.start]
    end_atom = atoms[segment.end - 1]

    with open(output_path, 'w') as f:
        entry_id = (f"{target_name}_cluster{consensus.cluster_id}"
                    f"_seg{segment.segment_id}")
        f.write(f"data_{entry_id}\n#\n")

        # Entry
        f.write(f"_entry.id   '{entry_id}'\n#\n")

        # Cluster provenance
        f.write("_mc_cluster_info.cluster_id             {}\n".format(consensus.cluster_id))
        f.write("_mc_cluster_info.n_members              {}\n".format(consensus.n_members))
        f.write("_mc_cluster_info.n_residues_total       {}\n".format(consensus.n_residues))
        f.write("_mc_cluster_info.medoid_name            '{}'\n".format(medoid_name))
        f.write("_mc_cluster_info.target_name            '{}'\n".format(target_name))
        f.write("_mc_cluster_info.mean_rmsf_full         {:.4f}\n".format(consensus.mean_rmsf))
        f.write("_mc_cluster_info.rmsf_threshold         {:.4f}\n".format(
            consensus.rmsf_threshold))
        f.write("_mc_cluster_info.coordinate_type        'consensus_mean'\n")
        f.write("_mc_cluster_info.alignment_method       'Kabsch'\n")
        f.write("#\n")

        # Segment provenance
        f.write("_mc_segment_info.segment_id             {}\n".format(segment.segment_id))
        f.write("_mc_segment_info.chain_id               '{}'\n".format(start_atom['chain']))
        f.write("_mc_segment_info.start_residue_index    {}\n".format(segment.start))
        f.write("_mc_segment_info.end_residue_index      {}\n".format(segment.end - 1))
        f.write("_mc_segment_info.start_residue_auth     {}\n".format(start_atom['resi']))
        f.write("_mc_segment_info.end_residue_auth       {}\n".format(end_atom['resi']))
        f.write("_mc_segment_info.start_residue_name     '{}'\n".format(start_atom['resn']))
        f.write("_mc_segment_info.end_residue_name       '{}'\n".format(end_atom['resn']))
        f.write("_mc_segment_info.length                 {}\n".format(segment.length))
        f.write("_mc_segment_info.mean_rmsf              {:.4f}\n".format(segment.mean_rmsf))
        f.write("_mc_segment_info.max_rmsf               {:.4f}\n".format(segment.max_rmsf))
        f.write("_mc_segment_info.biso_content           'per_residue_rmsf'\n")
        f.write("#\n")

        # atom_site loop
        f.write("loop_\n")
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.label_atom_id\n")
        f.write("_atom_site.label_alt_id\n")
        f.write("_atom_site.label_comp_id\n")
        f.write("_atom_site.label_asym_id\n")
        f.write("_atom_site.label_entity_id\n")
        f.write("_atom_site.label_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        f.write("_atom_site.occupancy\n")
        f.write("_atom_site.B_iso_or_equiv\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.pdbx_mc_rmsf\n")
        f.write("_atom_site.pdbx_mc_conserved\n")

        atom_serial = 1
        for i in range(segment.start, segment.end):
            atom = atoms[i]
            x, y, z = consensus.mean_coords[i]
            rmsf_val = consensus.rmsf[i]
            bfactor = min(rmsf_val, 999.99)
            conserved = "Y" if consensus.conserved_mask[i] else "N"

            f.write(
                f"ATOM {atom_serial:5d} C CA . {atom['resn']:>3s} {atom['chain']} 1 "
                f"{atom['resi']:4d} ? "
                f"{x:8.3f} {y:8.3f} {z:8.3f} "
                f"1.00 {bfactor:6.2f} "
                f"{atom['resi']:4d} {atom['chain']} "
                f"{rmsf_val:.4f} {conserved}\n"
            )
            atom_serial += 1

        f.write("#\n")


def extract_local_substructures(
    pdb_paths: List[Union[str, Path]],
    labels: np.ndarray,
    medoid_indices: Optional[List[int]] = None,
    rmsf_threshold: Optional[float] = None,
    min_segment_length: int = 5,
    max_structures_per_cluster: int = 200,
) -> Dict[str, Any]:
    """
    Full pipeline: consensus → segments → cross-cluster comparison.

    Args:
        pdb_paths: All PDB paths
        labels: Cluster labels
        medoid_indices: Medoid indices per cluster
        rmsf_threshold: RMSF cutoff
        min_segment_length: Minimum conserved segment length
        max_structures_per_cluster: Max structures to align

    Returns:
        Dict with keys:
            'consensus': List[ConsensusResult]
            'segments': Dict[cluster_id, List[LocalSegment]]
            'comparisons': List[SegmentComparisonResult]
            'shared_cores': List of segment pairs with RMSD < 2.0 A
    """
    # Step 1: Consensus for each cluster
    consensus_list = extract_all_cluster_consensus(
        pdb_paths=pdb_paths,
        labels=labels,
        medoid_indices=medoid_indices,
        rmsf_threshold=rmsf_threshold,
        max_structures_per_cluster=max_structures_per_cluster,
    )

    # Step 2: Find conserved segments per cluster
    all_segments = {}
    for cons in consensus_list:
        segs = find_conserved_segments(
            cons,
            min_segment_length=min_segment_length,
            rmsf_threshold=rmsf_threshold,
        )
        all_segments[cons.cluster_id] = segs

    # Step 3: Cross-cluster comparison
    comparisons = compare_segments_across_clusters(
        consensus_list,
        min_segment_length=min_segment_length,
        rmsf_threshold=rmsf_threshold,
    )

    # Step 4: Identify shared cores (low RMSD overlaps)
    shared_cores = [c for c in comparisons if c.rmsd < 2.0]

    return {
        'consensus': consensus_list,
        'segments': all_segments,
        'comparisons': comparisons,
        'shared_cores': shared_cores,
    }
