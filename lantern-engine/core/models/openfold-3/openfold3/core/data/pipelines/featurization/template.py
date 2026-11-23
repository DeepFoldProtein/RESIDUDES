# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains preprocessing pipelines for template data."""

import logging
from typing import Any, Sequence

import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.padding import pad_token_dim
from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
)
from openfold3.core.data.primitives.featurization.template import (
    create_template_distogram,
    create_template_feature_precursor_of3,
    create_template_restype,
    create_template_unit_vector,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.template import TemplateSliceCollection

logger = logging.getLogger(__name__)


def _build_cross_chain_template_rows_of3(
    multimer_templates: Sequence[Any],
    chain_ids_token: np.ndarray,
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build precursor rows (res_names, pb_coords, frame_coords) for each
    multimer template. `templateIndices` is a flat numbering across the cif's
    polymer chains in cif iteration order (gen_inputs/combine.py convention),
    so we concat per-residue arrays across chains and index with `t_i`.

    Returns None if no rows could be built.
    """
    try:
        from Bio.PDB import MMCIFParser
    except ImportError:
        logger.warning(
            "cross-chain template: Bio.PDB unavailable; skipping multimer_templates"
        )
        return None

    chain_offsets: dict[str, int] = {}
    chain_lens: dict[str, int] = {}
    for i, cid in enumerate(chain_ids_token):
        cid_s = str(cid)
        if cid_s not in chain_offsets:
            chain_offsets[cid_s] = int(i)
        chain_lens[cid_s] = chain_lens.get(cid_s, 0) + 1

    parser = MMCIFParser(QUIET=True)
    rows_res: list[np.ndarray] = []
    rows_pb: list[np.ndarray] = []
    rows_frame: list[np.ndarray] = []

    for mt in multimer_templates:
        cif_path = getattr(mt, "mmcif_path", None) or (mt.get("mmcif_path") if isinstance(mt, dict) else None)
        qi_dict = getattr(mt, "query_indices", None) or (mt.get("query_indices") if isinstance(mt, dict) else None)
        ti_dict = getattr(mt, "template_indices", None) or (mt.get("template_indices") if isinstance(mt, dict) else None)
        if not (cif_path and qi_dict and ti_dict):
            continue

        try:
            structure = parser.get_structure("cc", str(cif_path))
        except Exception as e:
            logger.warning(
                f"cross-chain template: failed to parse {cif_path}: {e!r}"
            )
            continue
        try:
            model = next(structure.get_models())
        except StopIteration:
            continue

        # Flat per-residue arrays across all polymer chains in cif iteration order.
        flat_res: list[str] = []
        flat_pb: list[np.ndarray] = []
        flat_frame: list[np.ndarray] = []
        for chain in model.get_chains():
            for res in chain.get_residues():
                if res.id[0] != " ":
                    continue
                rn = res.get_resname()
                pos_n = pos_ca = pos_c = pos_cb = None
                for atom in res.get_atoms():
                    name = atom.get_name()
                    if name == "N":
                        pos_n = atom.get_coord()
                    elif name == "CA":
                        pos_ca = atom.get_coord()
                    elif name == "C":
                        pos_c = atom.get_coord()
                    elif name == "CB":
                        pos_cb = atom.get_coord()
                pb = pos_ca if rn == "GLY" else pos_cb
                if pb is None:
                    pb = np.full(3, np.nan, dtype=np.float32)
                if pos_n is None or pos_ca is None or pos_c is None:
                    frame = np.full((3, 3), np.nan, dtype=np.float32)
                else:
                    frame = np.stack(
                        [pos_n, pos_ca, pos_c], axis=0
                    ).astype(np.float32)
                flat_res.append(rn)
                flat_pb.append(np.asarray(pb, dtype=np.float32))
                flat_frame.append(frame)

        if not flat_res:
            continue

        flat_res_arr = np.array(flat_res, dtype=np.dtype("U3"))
        flat_pb_arr = np.stack(flat_pb, axis=0)
        flat_frame_arr = np.stack(flat_frame, axis=0)

        row_res = np.full(n_tokens, "GAP", dtype=np.dtype("U3"))
        row_pb = np.full((n_tokens, 3), np.nan, dtype=np.float32)
        row_frame = np.full((n_tokens, 3, 3), np.nan, dtype=np.float32)

        n_filled = 0
        chains_filled: set = set()
        for q_chain, q_idxs in qi_dict.items():
            t_idxs = ti_dict.get(q_chain, [])
            if q_chain not in chain_offsets:
                logger.warning(
                    f"cross-chain template: query chain {q_chain!r} not in "
                    f"atom_array; skipping"
                )
                continue
            q_offset = chain_offsets[q_chain]
            q_n = chain_lens[q_chain]
            for q_i, t_i in zip(q_idxs, t_idxs):
                if not (0 <= q_i < q_n and 0 <= t_i < flat_res_arr.shape[0]):
                    continue
                global_q = q_offset + q_i
                if not (0 <= global_q < n_tokens):
                    continue
                row_res[global_q] = flat_res_arr[t_i]
                row_pb[global_q] = flat_pb_arr[t_i]
                row_frame[global_q] = flat_frame_arr[t_i]
                n_filled += 1
                chains_filled.add(q_chain)

        if n_filled == 0:
            logger.warning(
                f"cross-chain template: {cif_path} produced 0 filled tokens"
            )
            continue

        logger.info(
            f"cross-chain template: {cif_path} → filled {n_filled} tokens "
            f"across chains_filled={sorted(chains_filled)} "
            f"(requested {list(qi_dict.keys())})"
        )
        rows_res.append(row_res)
        rows_pb.append(row_pb)
        rows_frame.append(row_frame)

    if not rows_res:
        return None
    return (
        np.stack(rows_res, axis=0),
        np.stack(rows_pb, axis=0),
        np.stack(rows_frame, axis=0),
    )


def featurize_templates_dummy_of3(n_templ, n_token):
    """Temporary function to generate dummy template features."""
    return {
        "template_restype": torch.ones((n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones((n_templ, n_token)).float(),
        "template_distogram": torch.ones((n_templ, n_token, n_token, 39)).int(),
        "template_unit_vector": torch.ones((n_templ, n_token, n_token, 3)).float(),
    }


@log_runtime_memory(runtime_dict_key="runtime-template-feat")
def featurize_template_structures_of3(
    atom_array: AtomArray,
    template_slice_collection: TemplateSliceCollection,
    n_templates: int,
    n_tokens: int,
    min_bin: float,
    max_bin: float,
    n_bins: int,
    multimer_templates: Sequence[Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Featurizes template data for AF3.

    Returned features:
        - template_pseudo_beta_mask:
            Mask indicating if pseudo beta atoms (C-alpha atom for GLY, C-beta
            otherwise) are present in the corresponding token.
        - template_backbone_frame_mask:
            Mask indicating if backbone frame atoms (N, CA, C) are present in the
            corresponding token.
        - template_restype:
            One-hot encoded residue types.
        - template_distogram:
            Distogram constructed from pseudo beta atoms distances.
        - template_unit_vector:
            Unit vector constructed from backbone frames.

    Args:
        template_slice_collection (TemplateSliceCollection):
            The collection of cropped template atom arrays per chain, per template.
        n_templates (int):
            Number of templates.
        n_tokens (int):
            Number of tokens in the target structure.
        min_bin (float):
            The minimum distance for the distogram bins.
        max_bin (float):
            The maximum distance for the distogram bins.
        n_bins (int):
            The number of bins in the distogram.

    Returns:
        dict[str, torch.Tensor]:
            The featurized template data.
    """
    template_feature_precursor = create_template_feature_precursor_of3(
        template_slice_collection,
        n_templates,
        n_tokens,
    )

    # Create asym ID to mask inter-chain features and mask
    token_starts_with_stop, _ = extract_starts_entities(atom_array)
    token_starts = token_starts_with_stop[:-1]
    chain_ids_token = atom_array.chain_id[token_starts]
    _, renum_ids = np.unique(chain_ids_token, return_inverse=True)
    asym_id = pad_token_dim(
        {"asym_id": torch.tensor(renum_ids + 1, dtype=torch.int32)}, n_tokens
    )["asym_id"]

    # Optional cross-chain template rows. Built from `multimer_templates` by
    # parsing each cif and projecting its atoms (concat'd across all polymer
    # chains in cif iteration order) onto the multi-chain query token grid.
    # Cross-chain rows are appended after per-chain rows; `template_is_cross_chain`
    # marks them so downstream (template_embedders.py) widens the per-pair
    # mask only on those rows.
    res_names = template_feature_precursor.res_names
    pb_coords = template_feature_precursor.pseudo_beta_atom_coords
    frame_coords = template_feature_precursor.frame_atom_coords

    if multimer_templates:
        extra = _build_cross_chain_template_rows_of3(
            multimer_templates, chain_ids_token, n_tokens
        )
        if extra is not None:
            extra_res, extra_pb, extra_frame = extra
            res_names = np.concatenate([res_names, extra_res], axis=0)
            pb_coords = np.concatenate([pb_coords, extra_pb], axis=0)
            frame_coords = np.concatenate([frame_coords, extra_frame], axis=0)

    n_total_templates = res_names.shape[0]
    is_cross_chain = np.zeros(n_total_templates, dtype=bool)
    if multimer_templates and n_total_templates > n_templates:
        is_cross_chain[n_templates:] = True

    # Create features
    features = {}
    features["template_pseudo_beta_mask"] = torch.tensor(
        ~np.isnan(pb_coords).any(axis=-1),
        dtype=torch.float,
    )
    features["template_backbone_frame_mask"] = torch.tensor(
        ~np.isnan(frame_coords).any(axis=(-2, -1)),
        dtype=torch.float,
    )
    features["template_restype"] = create_template_restype(
        res_names,
        features["template_pseudo_beta_mask"],
    )

    # Per-template multichain_pair_mask: non-cross-chain rows get block_diag
    # only (existing behaviour); cross-chain rows get block_diag ∨ pair_cov,
    # where pair_cov is the outer product of this row's pseudo-beta mask
    # (so any pair of tokens both covered by the template is unmasked).
    block_diag = (asym_id[..., None] == asym_id[..., None, :]).to(torch.float)
    if bool(is_cross_chain.any()):
        pb_mask_t = features["template_pseudo_beta_mask"]
        pair_cov = pb_mask_t[..., :, None] * pb_mask_t[..., None, :]
        is_cc_t = torch.tensor(is_cross_chain, dtype=torch.float)[..., None, None]
        multichain_pair_mask = (
            block_diag[None, ...] + is_cc_t * pair_cov
        ).clamp(max=1.0)
        multichain_pair_mask = multichain_pair_mask[..., None]
    else:
        multichain_pair_mask = block_diag[..., None, :, :, None]

    features["template_distogram"] = create_template_distogram(
        pb_coords,
        features["template_pseudo_beta_mask"],
        multichain_pair_mask,
        min_bin,
        max_bin,
        n_bins,
    )
    features["template_unit_vector"] = create_template_unit_vector(
        frame_coords,
        features["template_backbone_frame_mask"],
        multichain_pair_mask,
    )
    features["template_is_cross_chain"] = torch.tensor(is_cross_chain)
    return features
