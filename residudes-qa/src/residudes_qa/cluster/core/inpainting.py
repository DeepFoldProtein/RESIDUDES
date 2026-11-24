"""
Superposition core → PatchR inpainting pipeline.

Converts per-residue TM-weight data from superposition core analysis
into PatchR input format (YAML + CIF template + inpainting metadata JSON),
and provides analysis utilities for inpainting results.

PatchR determines fixed vs inpainted residues by atom *presence* in the CIF
template — residues with atoms are fixed, missing residues are inpainted.
The metadata JSON is informational only.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_pdb_chains(pdb_path: str) -> tuple[dict, dict]:
    """Parse PDB to get chain → residue list mapping and sequences.

    Returns:
        chains: {chain_id: [resnum, ...]}
        sequences: {chain_id: [resname_3letter, ...]}
    """
    chains = {}
    sequences = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21]
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()

            if chain_id not in chains:
                chains[chain_id] = []
                sequences[chain_id] = []
            if resnum not in chains[chain_id]:
                chains[chain_id].append(resnum)
                sequences[chain_id].append(resname)

    return chains, sequences


def classify_residues(
    chains: dict,
    weights: np.ndarray,
    threshold: float,
) -> dict:
    """Classify residues as fixed or inpainted based on TM-weight threshold.

    Args:
        chains: {chain_id: [resnum, ...]} from parse_pdb_chains
        weights: per-residue TM-weight array (global ordering)
        threshold: weight threshold — residues with w >= threshold are fixed

    Returns:
        metadata dict with per-chain fixed/inpainted residue lists
    """
    chain_ids = sorted(chains.keys())

    # Map global residue index → (chain, resnum)
    global_to_chain = []
    for cid in chain_ids:
        for resnum in chains[cid]:
            global_to_chain.append((cid, resnum))

    metadata = {"chains": {}}
    for cid in chain_ids:
        fixed = []
        inpainted = []
        chain_indices = [i for i, (c, _) in enumerate(global_to_chain) if c == cid]
        for idx in chain_indices:
            _, resnum = global_to_chain[idx]
            if weights[idx] >= threshold:
                fixed.append(resnum)
            else:
                inpainted.append(resnum)
        metadata["chains"][cid] = {
            "fully_fixed_residues": fixed,
            "partially_fixed_residues": [],
            "fully_inpainted_residues": inpainted,
        }

    return metadata


def generate_patchr_input(
    pdb_path: str,
    core_csv: str,
    threshold: float,
    target: str,
    output_dir: str,
    weight_col: str = "weight_mean",
    n_samples: int = 5,
) -> tuple[Path, Path, Path]:
    """Generate PatchR inpainting input from superposition core data.

    Creates:
      1. Template CIF (converted from PDB, with inpainted residues removed)
      2. Inpainting metadata JSON (fixed vs inpainted residues per chain)
      3. YAML config for PatchR predict

    Args:
        pdb_path: path to medoid PDB file
        core_csv: path to cross_cluster_core.csv
        threshold: TM-weight threshold for fixed residues
        target: target name (used for output filenames)
        output_dir: output directory
        weight_col: column name for weight in core CSV
        n_samples: number of diffusion samples (written to YAML)

    Returns:
        (yaml_path, cif_path, meta_path)
    """
    import gemmi

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load core data and PDB
    core_df = pd.read_csv(core_csv)
    n_residues = len(core_df)
    # Support both old and new column names
    if weight_col not in core_df.columns:
        fallback_map = {
            "weight_mean": "mean_weight_across_clusters",
            "mean_weight_across_clusters": "weight_mean",
        }
        alt = fallback_map.get(weight_col)
        if alt and alt in core_df.columns:
            weight_col = alt
        else:
            raise KeyError(f"Column '{weight_col}' not found. Available: {list(core_df.columns)}")
    weights = core_df[weight_col].values

    chains, sequences = parse_pdb_chains(pdb_path)
    chain_ids = sorted(chains.keys())

    # Validate
    global_to_chain = []
    for cid in chain_ids:
        for resnum in chains[cid]:
            global_to_chain.append((cid, resnum))
    assert len(global_to_chain) == n_residues, \
        f"Mismatch: PDB has {len(global_to_chain)} residues, core has {n_residues}"

    # Classify residues
    metadata = classify_residues(chains, weights, threshold)

    # Save metadata JSON
    meta_path = out / f"{target}_inpainting_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Convert PDB → CIF with atom removal for inpainted residues
    st = gemmi.read_pdb(pdb_path)

    # Capture full sequence BEFORE removing residues
    # (predicted PDBs lack SEQRES, so gemmi can't auto-populate entity sequences)
    full_sequences_by_chain = {}
    for chain in st[0]:
        full_sequences_by_chain[chain.name] = [res.name for res in chain]

    # Build inpaint set and remove atoms
    inpaint_set = set()
    for cid in chain_ids:
        for resnum in metadata["chains"][cid]["fully_inpainted_residues"]:
            inpaint_set.add((cid, resnum))

    removed_count = 0
    for model in st:
        for chain in model:
            residues_to_remove = []
            for i, residue in enumerate(chain):
                if (chain.name, residue.seqid.num) in inpaint_set:
                    residues_to_remove.append(i)
            for i in reversed(residues_to_remove):
                del chain[i]
                removed_count += 1

    st.setup_entities()
    st.assign_label_seq_id()

    # Fix subchain IDs (gemmi auto-generates 'Bxp', 'Cxp' etc.)
    for model in st:
        for chain in model:
            for residue in chain:
                residue.subchain = chain.name
    for entity in st.entities:
        entity.subchains = [s.replace('xp', '') if s.endswith('xp') else s
                            for s in entity.subchains]

    # Populate entity full_sequence from pre-removal ATOM records
    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and not entity.full_sequence:
            for sub_id in entity.subchains:
                if sub_id in full_sequences_by_chain:
                    entity.full_sequence = full_sequences_by_chain[sub_id]
                    break

    cif_path = out / f"{target}_template.cif"
    st.make_mmcif_document().write_file(str(cif_path))

    # Fix space group (predicted structures don't have one)
    subprocess.run(['sed', '-i',
                    "s/^_symmetry.space_group_name_H-M .*/"
                    "_symmetry.space_group_name_H-M             'P 1'/",
                    str(cif_path)], check=True)

    # Build YAML
    seq_entries = []
    for cid in chain_ids:
        seq = ''.join(THREE_TO_ONE.get(r, 'X') for r in sequences[cid])
        seq_entries.append(
            f"  - protein:\n      id: '{cid}'\n      sequence: {seq}\n      msa: empty"
        )

    yaml_content = f"""version: 1

sequences:
{chr(10).join(seq_entries)}

templates:
  - cif: {cif_path.name}
    chain_id: {json.dumps(chain_ids)}
    inpainting_metadata: {meta_path.name}
"""
    yaml_path = out / f"{target}.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Summary
    fix_count = sum(len(metadata['chains'][c]['fully_fixed_residues']) for c in chain_ids)
    inp_count = sum(len(metadata['chains'][c]['fully_inpainted_residues']) for c in chain_ids)

    print(f"  Removed {removed_count} residues from template CIF")
    print(f"  Fixed: {fix_count}/{n_residues} ({fix_count/n_residues:.1%})")
    print(f"  Inpainted: {inp_count}/{n_residues} ({inp_count/n_residues:.1%})")
    for cid in chain_ids:
        n_fix = len(metadata['chains'][cid]['fully_fixed_residues'])
        n_inp = len(metadata['chains'][cid]['fully_inpainted_residues'])
        print(f"  Chain {cid}: {n_fix} fixed, {n_inp} inpainted")

    return yaml_path, cif_path, meta_path


def run_patchr(
    yaml_path: str,
    results_dir: str,
    n_samples: int = 5,
    gpu: int = 0,
) -> int:
    """Run PatchR predict.

    Returns:
        return code (0 = success)
    """
    import os
    cmd = [
        "patchr", "predict", str(yaml_path),
        "--out_dir", str(results_dir),
        "--diffusion_samples", str(n_samples),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _superpose_on_fixed(coords_ref, coords_mov, fixed_mask):
    """Superpose coords_mov onto coords_ref using fixed residues.

    Returns:
        aligned coords_mov
    """
    fix_ref = coords_ref[fixed_mask]
    fix_mov = coords_mov[fixed_mask]
    c_ref = fix_ref.mean(0)
    c_mov = fix_mov.mean(0)
    H = (fix_mov - c_mov).T @ (fix_ref - c_ref)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (coords_mov - c_mov) @ R.T + c_ref


def analyze_inpainting(
    pred_dir: str,
    metadata_path: str,
    target: str,
    output_dir: str,
    native_pdb: Optional[str] = None,
) -> dict:
    """Analyze PatchR inpainting results.

    Computes:
      - Structural diversity (pairwise RMSD of fixed vs inpainted regions)
      - Per-residue RMSF
      - Confidence scores
      - Optional DockQ vs native

    Args:
        pred_dir: directory containing model CIFs and confidence JSONs
        metadata_path: inpainting_metadata.json
        target: target name
        output_dir: where to save analysis outputs
        native_pdb: optional native PDB for DockQ

    Returns:
        dict with analysis summary
    """
    import gemmi
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pred_dir = Path(pred_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    inpaint_resnums = {}
    for cid, chain_data in metadata["chains"].items():
        inpaint_resnums[cid] = set(chain_data["fully_inpainted_residues"])

    # Load model CIFs
    model_cifs = sorted(pred_dir.glob(f"{target}_model_*.cif"))
    n_models = len(model_cifs)
    if n_models == 0:
        print("  [ERROR] No model CIFs found")
        return {}

    # Extract CA coordinates
    all_ca = []
    fixed_mask = None
    for cif_path in model_cifs:
        st = gemmi.read_structure(str(cif_path))
        ca_coords = []
        mask = []
        for chain in st[0]:
            cname = chain.name
            for res in chain:
                ca = res.find_atom('CA', '*')
                if ca:
                    ca_coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
                    is_inp = cname in inpaint_resnums and res.seqid.num in inpaint_resnums[cname]
                    mask.append(not is_inp)
        all_ca.append(np.array(ca_coords))
        if fixed_mask is None:
            fixed_mask = np.array(mask)

    n_total = len(fixed_mask)
    n_fixed = int(fixed_mask.sum())
    n_inp = n_total - n_fixed

    # Confidence scores
    conf_scores = []
    for i in range(n_models):
        conf_file = pred_dir / f"confidence_{target}_model_{i}.json"
        if conf_file.exists():
            with open(conf_file) as f:
                conf_scores.append(json.load(f))

    # Pairwise RMSD after superposition on fixed residues
    rmsds_fix, rmsds_inp = [], []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            aligned_j = _superpose_on_fixed(all_ca[i], all_ca[j], fixed_mask)
            diff_fix = aligned_j[fixed_mask] - all_ca[i][fixed_mask]
            diff_inp = aligned_j[~fixed_mask] - all_ca[i][~fixed_mask]
            rmsds_fix.append(np.sqrt(np.mean(np.sum(diff_fix**2, axis=1))))
            if (~fixed_mask).sum() > 0:
                rmsds_inp.append(np.sqrt(np.mean(np.sum(diff_inp**2, axis=1))))

    # Per-residue RMSF (aligned to model 0)
    aligned_all = [all_ca[0].copy()]
    for k in range(1, n_models):
        aligned_all.append(_superpose_on_fixed(all_ca[0], all_ca[k], fixed_mask))

    coords_stack = np.stack(aligned_all)
    per_res_rmsf = np.sqrt(np.mean(np.sum(
        (coords_stack - coords_stack.mean(axis=0, keepdims=True))**2, axis=2
    ), axis=0))

    # DockQ evaluation
    dockq_results = None
    if native_pdb:
        dockq_results = []
        for i, cif_path in enumerate(model_cifs):
            try:
                result = subprocess.run(
                    ["DockQ", str(cif_path), native_pdb, "--short"],
                    capture_output=True, text=True, timeout=60
                )
                for line in result.stdout.strip().split('\n'):
                    if line.startswith("DockQ"):
                        dockq_results.append({"model": i, "dockq": float(line.split()[1])})
                        break
                else:
                    dockq_results.append({"model": i, "dockq": None})
            except Exception as e:
                dockq_results.append({"model": i, "dockq": None})

    # Save analysis CSV
    analysis_rows = []
    for i in range(n_models):
        row = {
            "target": target, "model": i,
            "n_fixed": n_fixed, "n_inpainted": n_inp, "n_total": n_total,
            "confidence": conf_scores[i]["confidence_score"] if i < len(conf_scores) else None,
            "ptm": conf_scores[i].get("ptm") if i < len(conf_scores) else None,
            "iptm": conf_scores[i].get("iptm") if i < len(conf_scores) else None,
            "rmsf_fixed_mean": float(per_res_rmsf[fixed_mask].mean()),
            "rmsf_inpainted_mean": float(per_res_rmsf[~fixed_mask].mean()),
        }
        if dockq_results and i < len(dockq_results):
            row["dockq"] = dockq_results[i]["dockq"]
        analysis_rows.append(row)

    analysis_df = pd.DataFrame(analysis_rows)
    analysis_csv = out / f"{target}_inpainting_analysis.csv"
    analysis_df.to_csv(analysis_csv, index=False)

    # Save per-residue RMSF
    rmsf_df = pd.DataFrame({
        "residue_idx": range(n_total),
        "is_fixed": fixed_mask.astype(int),
        "rmsf": per_res_rmsf,
    })
    rmsf_csv = out / f"{target}_inpainting_rmsf.csv"
    rmsf_df.to_csv(rmsf_csv, index=False)

    # Generate figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [1, 2, 1]})
    fig.suptitle(f'{target} Inpainting Analysis', fontsize=14, fontweight='bold')
    residues = np.arange(n_total)

    # Panel 1: Fixed/Inpainted map
    ax = axes[0]
    colors = ['#1565c0' if m else '#d32f2f' for m in fixed_mask]
    ax.bar(residues, np.ones(n_total), width=1.0, color=colors, edgecolor='none')
    ax.set_yticks([])
    ax.set_title(f'Fixed (blue): {n_fixed} | Inpainted (red): {n_inp}', fontsize=10)
    ax.set_xlim(0, n_total)

    # Panel 2: Per-residue RMSF
    ax = axes[1]
    bar_colors = ['#1565c0' if m else '#d32f2f' for m in fixed_mask]
    ax.bar(residues, per_res_rmsf, width=1.0, color=bar_colors, edgecolor='none', alpha=0.8)
    ax.set_ylabel('RMSF (Å)')
    ax.set_title(f'Per-residue RMSF ({n_models} models) — '
                 f'Fixed: {per_res_rmsf[fixed_mask].mean():.2f}Å, '
                 f'Inpainted: {per_res_rmsf[~fixed_mask].mean():.2f}Å', fontsize=10)
    ax.set_xlim(0, n_total)

    # Panel 3: Confidence per model
    ax = axes[2]
    model_indices = range(n_models)
    confs = [cs['confidence_score'] for cs in conf_scores[:n_models]]
    bar_colors_conf = plt.cm.RdYlGn([c for c in confs])
    ax.bar(model_indices, confs, color=bar_colors_conf, edgecolor='black', linewidth=0.5)
    if dockq_results:
        dqs = [dr['dockq'] if dr['dockq'] is not None else 0 for dr in dockq_results[:n_models]]
        ax.bar([x + 0.3 for x in model_indices], dqs,
               width=0.3, color='steelblue', alpha=0.7, label='DockQ')
        ax.legend(fontsize=8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Confidence' + (' & DockQ' if dockq_results else ''), fontsize=10)
    ax.set_xticks(model_indices)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = out / f"{target}_inpainting_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Build summary dict
    summary = {
        "target": target,
        "n_fixed": n_fixed,
        "n_inpainted": n_inp,
        "n_total": n_total,
        "n_models": n_models,
        "rmsd_fixed_mean": float(np.mean(rmsds_fix)) if rmsds_fix else 0.0,
        "rmsd_inpainted_mean": float(np.mean(rmsds_inp)) if rmsds_inp else 0.0,
        "rmsf_fixed_mean": float(per_res_rmsf[fixed_mask].mean()),
        "rmsf_inpainted_mean": float(per_res_rmsf[~fixed_mask].mean()),
        "confidence_scores": [cs.get("confidence_score") for cs in conf_scores],
        "analysis_csv": str(analysis_csv),
        "rmsf_csv": str(rmsf_csv),
        "figure": str(fig_path),
    }
    if dockq_results:
        summary["dockq_scores"] = [dr["dockq"] for dr in dockq_results]

    return summary
