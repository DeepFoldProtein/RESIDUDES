#!/usr/bin/env python3
"""
ProteinMPNN-based backbone quality scoring (general QA tool).

Uses ProteinMPNN sequence recovery as a structure-independent confidence metric.
For MassiveFold predictions (100% sequence identity), per-residue recovery
measures how well each backbone region encodes the original sequence —
i.e., backbone "designability" as a proxy for structural quality.

Key insight for 100% seq identity:
    All predictions share the same sequence, so recovery differences
    are purely due to backbone geometry differences.
    High recovery = physically plausible backbone
    Low recovery = unnatural backbone geometry (likely mispredicted)

Usage (default — auto-finds bundled proteinmpnn/run/):
    from residudes_qa.ema.scorers.proteinmpnn_score import MPNNScorer
    scorer = MPNNScorer()
    result = scorer.score_structure("model.pdb")
    print(f"Recovery: {result.overall_recovery:.3f}")
    print(f"Per-residue NLL: {result.per_residue_score}")

Usage (explicit):
    scorer = MPNNScorer(proteinmpnn_dir="/path/to/proteinmpnn/run")
"""

import os
import sys
import json
from pathlib import Path


def _find_proteinmpnn_dir() -> str:
    """Resolve the proteinmpnn/run/ directory.

    Priority:
      1. $EMA_PROTEINMPNN_DIR env var
      2. MassiveClustering_tool/proteinmpnn/run/ relative to repo root
    """
    env = os.environ.get('EMA_PROTEINMPNN_DIR')
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    # repo root = ../../.. from EMA/scorers/proteinmpnn_score.py
    candidate = here.parents[2] / 'mpnn' / 'run'  # residudes-qa: vendored ProteinMPNN code
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        'ProteinMPNN run/ directory not found. Either:\n'
        '  - bash MassiveClustering_tool/proteinmpnn/setup_weights.sh\n'
        '  - or set EMA_PROTEINMPNN_DIR to your existing install.'
    )
import copy
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class MPNNResult:
    """Result of ProteinMPNN scoring for a single structure."""
    pdb_path: str
    overall_recovery: float          # Global sequence recovery rate (0-1)
    per_residue_recovery: np.ndarray  # Shape (L,), per-residue recovery
    mpnn_score: float                # Negative log-likelihood (lower = better fit)
    per_residue_score: np.ndarray    # Shape (L,), per-residue NLL
    designed_sequences: List[str]    # Generated sequences
    native_sequence: str             # Original sequence from PDB
    n_residues: int                  # Protein length


@dataclass
class ClusterConfidence:
    """Confidence assessment for a cluster."""
    cluster_id: int
    consensus_result: Optional[MPNNResult] = None
    medoid_result: Optional[MPNNResult] = None
    member_results: List[MPNNResult] = field(default_factory=list)
    mean_recovery: float = 0.0
    std_recovery: float = 0.0
    per_residue_mean_recovery: Optional[np.ndarray] = None
    per_residue_std_recovery: Optional[np.ndarray] = None


class MPNNScorer:
    """
    ProteinMPNN-based backbone quality scorer.

    Wraps ProteinMPNN to compute per-residue sequence recovery
    as a backbone geometry quality metric.

    Args:
        proteinmpnn_dir: Path to proteinmpnn/run/ directory
            (contains run_protein_mpnn.py, model_weights/, protein_mpnn/)
        model_name: Which ProteinMPNN model to use
        device: 'cuda' or 'cpu'
        num_sequences: Number of sequences to sample per structure
        sampling_temp: Sampling temperature (0.1 = near-greedy, good for recovery)
    """

    def __init__(
        self,
        proteinmpnn_dir: str | None = None,
        model_name: str = "v_48_020",
        device: str = "auto",
        num_sequences: int = 8,
        sampling_temp: float = 0.1,
        skip_sampling: bool = False,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for ProteinMPNN scoring.")

        if proteinmpnn_dir is None:
            proteinmpnn_dir = _find_proteinmpnn_dir()
        self.proteinmpnn_dir = str(Path(proteinmpnn_dir).resolve())
        self.model_name = model_name
        self.num_sequences = num_sequences
        self.sampling_temp = sampling_temp
        # When True, score_structure() runs only forward() (native NLL +
        # per-residue NLL) and skips the autoregressive sample() loop.
        # `overall_recovery`, `per_residue_recovery`, `designed_sequences`
        # are returned as placeholders (0.0 / empty).
        self.skip_sampling = skip_sampling

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Add proteinmpnn to sys.path for imports
        if self.proteinmpnn_dir not in sys.path:
            sys.path.insert(0, self.proteinmpnn_dir)

        # Verify model weights exist
        # residudes-qa: weights live in the data root, code in the package
        weight_dir = os.environ.get("EMA_PROTEINMPNN_WEIGHTS") or os.path.join(self.proteinmpnn_dir, "model_weights")
        ckpt_path = os.path.join(weight_dir, f"{model_name}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Model weights not found: {ckpt_path}\n"
                f"Available: {os.listdir(weight_dir) if os.path.isdir(weight_dir) else 'dir not found'}"
            )

        # Load model once
        self.model, self.model_config = self._load_model(ckpt_path)
        print(f"MPNNScorer initialized: model={model_name}, device={self.device}, "
              f"num_seq={num_sequences}, temp={sampling_temp}, "
              f"skip_sampling={skip_sampling}")

    def _load_model(self, ckpt_path: str):
        """Load ProteinMPNN model from checkpoint."""
        from protein_mpnn.protein_mpnn_utils import ProteinMPNN

        config = {"hidden_dim": 128, "num_layers": 3}
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        num_edges = ckpt["num_edges"]

        model = ProteinMPNN(
            num_letters=21,
            node_features=config["hidden_dim"],
            edge_features=config["hidden_dim"],
            hidden_dim=config["hidden_dim"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            augment_eps=0.0,
            k_neighbors=num_edges,
        )
        model.to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return model, config

    # 3-letter to 1-letter amino acid mapping
    _AA3TO1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        # Common modified residues
        "MSE": "M", "SEC": "U", "SEP": "S", "TPO": "T", "PTR": "Y",
    }

    def _parse_pdb_gemmi(self, pdb_path: str) -> dict:
        """
        Parse a PDB/CIF file using gemmi → ProteinMPNN dict format.

        Produces the exact dict format ProteinMPNN's tied_featurize expects:
            - name: str
            - num_of_chains: int
            - seq: str (concatenated)
            - seq_chain_{letter}: str (per-chain sequence)
            - coords_chain_{letter}: dict with N/CA/C/O coord lists
        """
        import gemmi

        path = str(pdb_path)
        if path.endswith(".cif") or path.endswith(".mmcif"):
            doc = gemmi.cif.read(path)
            structure = gemmi.make_structure_from_block(doc.sole_block())
        else:
            structure = gemmi.read_pdb(path)

        structure.setup_entities()

        my_dict = {}
        concat_seq = ""
        n_chains = 0

        for model in structure:
            for chain in model:
                chain_id = chain.name if chain.name.strip() else "A"
                # Collect residues with all 4 backbone atoms
                residues_data = []  # (seq_char, N, CA, C, O)

                for residue in chain:
                    # Check if it's a standard amino acid
                    res_info = gemmi.find_tabulated_residue(residue.name)
                    if not res_info.is_amino_acid():
                        continue
                    aa1 = self._AA3TO1.get(residue.name, "X")

                    # Find backbone atoms
                    atoms = {}
                    for atom_name in ["N", "CA", "C", "O"]:
                        atom = residue.find_atom(atom_name, "*")
                        if atom:
                            atoms[atom_name] = [atom.pos.x, atom.pos.y, atom.pos.z]

                    if len(atoms) == 4:
                        residues_data.append((
                            aa1,
                            atoms["N"], atoms["CA"], atoms["C"], atoms["O"],
                        ))

                if not residues_data:
                    continue

                seq = "".join(r[0] for r in residues_data)
                my_dict[f"seq_chain_{chain_id}"] = seq
                concat_seq += seq

                coords = {
                    f"N_chain_{chain_id}": [r[1] for r in residues_data],
                    f"CA_chain_{chain_id}": [r[2] for r in residues_data],
                    f"C_chain_{chain_id}": [r[3] for r in residues_data],
                    f"O_chain_{chain_id}": [r[4] for r in residues_data],
                }
                my_dict[f"coords_chain_{chain_id}"] = coords
                n_chains += 1

            break  # Only first model

        if n_chains == 0:
            raise ValueError(f"No protein chains with backbone atoms found in: {pdb_path}")

        # Set name from filename
        name = Path(pdb_path).stem
        my_dict["name"] = name
        my_dict["num_of_chains"] = n_chains
        my_dict["seq"] = concat_seq

        # Wrap through StructureDatasetPDB to get the final tensor-ready format
        from protein_mpnn.protein_mpnn_utils import StructureDatasetPDB
        dataset = StructureDatasetPDB([my_dict], max_length=20000)

        if len(dataset) == 0:
            raise ValueError(f"Structure too long or invalid: {pdb_path}")

        return dataset[0]

    def score_structure(
        self,
        pdb_path: str,
        original_sequence: Optional[str] = None,
    ) -> MPNNResult:
        """
        Score a single PDB structure using ProteinMPNN.

        Computes:
        1. Native score (NLL of original sequence given backbone)
        2. Designed sequences with sampling
        3. Per-residue sequence recovery

        Args:
            pdb_path: Path to PDB file
            original_sequence: If None, extracted from PDB

        Returns:
            MPNNResult with recovery rates and scores
        """
        from protein_mpnn.protein_mpnn_utils import (
            tied_featurize, _scores, _S_to_seq
        )

        pdb_path = str(Path(pdb_path).resolve())
        protein = self._parse_pdb_gemmi(pdb_path)

        # Build chain info
        all_chains = [
            item[-1:]
            for item in list(protein)
            if item[:9] == "seq_chain"
        ]

        # For "design all" mode: no fixed positions, no tied positions
        chain_id_dict = None
        fixed_positions_dict = None
        tied_positions_dict = None
        pssm_dict = None
        bias_by_res_dict = None

        # Build omit_AA_dict (only omit 'X')
        omit_AA_dict = {protein["name"]: {}}
        for chain in all_chains:
            omit_AA_dict[protein["name"]][chain] = []

        # Featurize
        batch_clones = [copy.deepcopy(protein)]
        (
            X, S, mask, _, chain_M, chain_encoding_all, _,
            visible_list_list, masked_list_list,
            masked_chain_length_list_list, chain_M_pos,
            omit_AA_mask, residue_idx, _,
            tied_pos_list_of_lists_list,
            pssm_coef, pssm_bias, pssm_log_odds_all,
            bias_by_res_all, tied_beta,
        ) = tied_featurize(
            batch_clones, self.device,
            chain_id_dict, fixed_positions_dict,
            omit_AA_dict, tied_positions_dict,
            pssm_dict, bias_by_res_dict,
        )

        # Design ALL positions
        chain_M_pos = torch.ones_like(chain_M_pos)

        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        omit_AAs_np = np.array(
            [AA == "X" for AA in alphabet]
        ).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))

        with torch.no_grad():
            # 1. Score the native sequence
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = self.model(
                X, S, mask, chain_M * chain_M_pos,
                residue_idx, chain_encoding_all, randn_1,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            native_scores, native_scores_per_res = _scores(S, log_probs, mask_for_loss)
            native_score = float(native_scores.cpu().numpy().mean())

            # Per-residue native score (only for designed positions)
            per_res_score = native_scores_per_res[0].cpu().numpy()  # (L,)
            score_mask = mask_for_loss[0].cpu().numpy()  # (L,)

            # 2. Sample sequences and compute recovery (skippable for
            # scoring-only use cases where forward NLL is sufficient).
            valid_mask = score_mask > 0.5
            if self.skip_sampling:
                # NaN placeholders → callers can detect "not computed" and
                # propagate NaN to feature tables (XGB then imputes via
                # train_means).  Matches the ablation experiment exactly.
                mean_per_res_recovery = np.full_like(per_res_score, np.nan)
                overall_recovery = float("nan")
                designed_seqs: List[str] = []
            else:
                all_recoveries = []
                designed_seqs = []

                for seq_idx in range(self.num_sequences):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    sample_dict = self.model.sample(
                        X, randn_2, S, chain_M, chain_encoding_all,
                        residue_idx, mask=mask, temperature=self.sampling_temp,
                        omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef, pssm_bias=pssm_bias,
                        pssm_multi=0.0,
                        pssm_log_odds_flag=False,
                        pssm_log_odds_mask=pssm_log_odds_all,
                        pssm_bias_flag=False,
                        bias_by_res=bias_by_res_all,
                    )
                    S_sample = sample_dict["S"]

                    # Per-residue recovery
                    match = (
                        torch.nn.functional.one_hot(S[0], 21)
                        * torch.nn.functional.one_hot(S_sample[0], 21)
                    ).sum(dim=-1) * mask_for_loss[0]
                    per_res_recovery = match.cpu().numpy()  # 1 where match, 0 where not
                    all_recoveries.append(per_res_recovery)

                    # Decode sequence
                    seq = _S_to_seq(S_sample[0], chain_M[0])
                    designed_seqs.append(seq)

                # Average per-residue recovery across samples
                all_recoveries = np.stack(all_recoveries, axis=0)  # (num_seq, L)
                mean_per_res_recovery = all_recoveries.mean(axis=0)  # (L,)

                if valid_mask.sum() > 0:
                    overall_recovery = float(mean_per_res_recovery[valid_mask].mean())
                else:
                    overall_recovery = 0.0

            # Extract native sequence
            native_seq = _S_to_seq(S[0], chain_M[0])

        # Trim to valid residues only
        per_res_recovery_valid = mean_per_res_recovery[valid_mask]
        per_res_score_valid = per_res_score[valid_mask]

        return MPNNResult(
            pdb_path=pdb_path,
            overall_recovery=overall_recovery,
            per_residue_recovery=per_res_recovery_valid,
            mpnn_score=native_score,
            per_residue_score=per_res_score_valid,
            designed_sequences=designed_seqs,
            native_sequence=native_seq,
            n_residues=int(valid_mask.sum()),
        )

    def score_multiple(
        self,
        pdb_paths: List[str],
        progress: bool = True,
    ) -> List[MPNNResult]:
        """Score multiple PDB structures."""
        results = []
        for i, pdb_path in enumerate(pdb_paths):
            if progress:
                print(f"  Scoring {i+1}/{len(pdb_paths)}: {Path(pdb_path).name}")
            try:
                result = self.score_structure(pdb_path)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to score {pdb_path}: {e}")
        return results

    def score_cluster(
        self,
        consensus_pdb: Optional[str] = None,
        medoid_pdb: Optional[str] = None,
        member_pdbs: Optional[List[str]] = None,
        n_sample_members: int = 5,
        cluster_id: int = 0,
    ) -> ClusterConfidence:
        """
        Comprehensive confidence assessment for a cluster.

        Scores consensus structure, medoid, and optionally a sample of members.

        Args:
            consensus_pdb: Path to consensus PDB (from consensus extraction)
            medoid_pdb: Path to medoid PDB
            member_pdbs: List of all member PDB paths
            n_sample_members: Number of random members to score
            cluster_id: Cluster identifier

        Returns:
            ClusterConfidence with per-residue recovery statistics
        """
        result = ClusterConfidence(cluster_id=cluster_id)

        # Score consensus
        if consensus_pdb and os.path.isfile(consensus_pdb):
            print(f"  Scoring consensus: {Path(consensus_pdb).name}")
            result.consensus_result = self.score_structure(consensus_pdb)

        # Score medoid
        if medoid_pdb and os.path.isfile(medoid_pdb):
            print(f"  Scoring medoid: {Path(medoid_pdb).name}")
            result.medoid_result = self.score_structure(medoid_pdb)

        # Score sample members
        if member_pdbs and n_sample_members > 0:
            n_sample = min(n_sample_members, len(member_pdbs))
            rng = np.random.default_rng(42)
            sampled = rng.choice(member_pdbs, size=n_sample, replace=False)
            result.member_results = self.score_multiple(list(sampled))

        # Compute aggregate statistics
        all_results = []
        if result.consensus_result:
            all_results.append(result.consensus_result)
        if result.medoid_result:
            all_results.append(result.medoid_result)
        all_results.extend(result.member_results)

        if all_results:
            recoveries = [r.overall_recovery for r in all_results]
            result.mean_recovery = float(np.mean(recoveries))
            result.std_recovery = float(np.std(recoveries))

            # Per-residue stats (only if all have same length)
            lengths = [r.n_residues for r in all_results]
            if len(set(lengths)) == 1:
                per_res = np.stack(
                    [r.per_residue_recovery for r in all_results], axis=0
                )
                result.per_residue_mean_recovery = per_res.mean(axis=0)
                result.per_residue_std_recovery = per_res.std(axis=0)

        return result


def save_mpnn_results(
    results: Union[MPNNResult, List[MPNNResult], ClusterConfidence],
    output_dir: str,
    target_name: str = "unknown",
) -> None:
    """
    Save ProteinMPNN scoring results to disk.

    Creates:
    - {target}_mpnn_summary.csv: Overall scores per structure
    - {target}_mpnn_per_residue.npz: Per-residue recovery and scores
    - {target}_mpnn_sequences.fasta: Designed sequences
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(results, MPNNResult):
        results = [results]
    elif isinstance(results, ClusterConfidence):
        all_results = []
        if results.consensus_result:
            all_results.append(results.consensus_result)
        if results.medoid_result:
            all_results.append(results.medoid_result)
        all_results.extend(results.member_results)
        results = all_results

    # Summary CSV
    import csv
    summary_path = output_dir / f"{target_name}_mpnn_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pdb_file", "n_residues", "overall_recovery",
            "mpnn_score", "mean_per_res_recovery", "std_per_res_recovery",
        ])
        for r in results:
            writer.writerow([
                Path(r.pdb_path).name,
                r.n_residues,
                f"{r.overall_recovery:.4f}",
                f"{r.mpnn_score:.4f}",
                f"{r.per_residue_recovery.mean():.4f}",
                f"{r.per_residue_recovery.std():.4f}",
            ])
    print(f"  Summary saved: {summary_path}")

    # Per-residue data
    per_res_path = output_dir / f"{target_name}_mpnn_per_residue.npz"
    per_res_data = {}
    for i, r in enumerate(results):
        name = Path(r.pdb_path).stem
        per_res_data[f"{name}_recovery"] = r.per_residue_recovery
        per_res_data[f"{name}_score"] = r.per_residue_score
    np.savez(per_res_path, **per_res_data)
    print(f"  Per-residue data saved: {per_res_path}")

    # FASTA with designed sequences
    fasta_path = output_dir / f"{target_name}_mpnn_designs.fasta"
    with open(fasta_path, "w") as f:
        for r in results:
            name = Path(r.pdb_path).stem
            f.write(f">{name}_native\n{r.native_sequence}\n")
            for j, seq in enumerate(r.designed_sequences):
                f.write(f">{name}_design_{j:02d}\n{seq}\n")
    print(f"  Designed sequences saved: {fasta_path}")
