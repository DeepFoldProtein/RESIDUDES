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

"""
Inference class template for first inference pipeline prototype.
"""

import itertools
import logging
import traceback

import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from torch.utils.data import Dataset

from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    pad_to_world_size,
)
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.msa import (
    MsaFeaturizerOF3,
    MsaFeaturizerOF3Config,
)
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.query import (
    StructureWithReferenceMolecules,
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    get_token_count,
    tokenize_atom_array,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)

logger = logging.getLogger(__name__)


@register_dataset
class InferenceDataset(Dataset):
    """Dataset class for running inference on a set of queries."""

    # TODO: Can accept a dataset_config here if we want
    def __init__(
        self,
        dataset_config,  # type : DefaultDatasetConfigSection
        world_size: int | None = None,
    ) -> None:
        """Initializes the InferenceDataset."""
        super().__init__()
        self.query_set = dataset_config.query_set
        self.query_cache = self.query_set.queries

        self.seeds: list = dataset_config.seeds
        self.world_size = world_size

        # Main alignments
        self.msa_settings = dataset_config.msa
        self.msa_sample_processor_inference = MsaSampleProcessorInference(
            config=self.msa_settings
        )
        self.msa_featurizer_of3 = MsaFeaturizerOF3(
            config=MsaFeaturizerOF3Config(
                max_rows=self.msa_settings.max_rows,
                max_rows_paired=self.msa_settings.max_rows_paired,
                subsample_with_bands=self.msa_settings.subsample_with_bands,
            )
        )

        # Templates
        self.template_settings = dataset_config.template
        self.template_preprocessor_settings = (
            dataset_config.template_preprocessor_settings
        )
        if self.template_preprocessor_settings.preparse_structures:
            self.template_preprocessor_settings.structure_file_format = "npz"

        # Parse CCD
        if dataset_config.ccd_file_path is not None:
            logger.debug("Parsing CCD file.")
            self.ccd = pdbx.CIFFile.read(dataset_config.ccd_file_path)
        else:
            self.ccd = BiotiteCCDWrapper()

        # Create individual datapoint cache (allows rerunning the same query with
        # different seeds)
        self.create_datapoint_cache()

    def create_datapoint_cache(self) -> None:
        qids = self.query_cache.keys()

        # Order by total sequence length (excluding ligands) so that the run times
        # are more consistent across GPUs
        query_len_map = {}
        for k, v in self.query_cache.items():
            total_poly_seq_len = sum(
                [
                    len(c.sequence) * len(c.chain_ids) if c.sequence is not None else 0
                    for c in v.chains
                ]
            )
            query_len_map[k] = total_poly_seq_len

        qids = sorted(
            qids,
            key=lambda q: query_len_map[q],
        )

        qid_values, seed_values = zip(
            *[(q, s) for q, s in itertools.product(qids, self.seeds)], strict=True
        )

        _datapoint_cache = pd.DataFrame(
            {
                "query_id": qid_values,
                "seed": seed_values,
            }
        )

        self.datapoint_cache = pad_to_world_size(_datapoint_cache, self.world_size)

    @staticmethod
    def get_structure_with_ref_mols(query: Query) -> StructureWithReferenceMolecules:
        """Creates a preprocessed AtomArray and reference molecules from the query.

        Parses the Query object into a full AtomArray and processed reference molecules
        (RDKit mol objects with atom names and computed conformers) matching the
        molecule components in the query. The returned AtomArray follows the chain IDs
        given in the Query object. If a chain specifies multiple chain IDs, repeated
        identical chains with those IDs will be constructed and given the same entity
        ID. Residue names will be inferred from the sequence or CCD codes. If a ligand
        is specified through a SMILES string, it will be named as "LIG-X", where X
        starts at 1 and is incremented for each unnamed ligand entity found in the
        Query.

        Additionally, this method adds tokenization information (token IDs) and token
        positions to the AtomArray, which are required by other functions in the
        featurization pipeline.

        Args:
            query (Query):
                The Query object containing the chains to construct the structure from.

        Returns:
            StructureWithReferenceMolecules:
                A named tuple containing the tokenized AtomArray and a list of processed
                reference molecules.
        """
        # Gets AtomArray and processed reference molecules with conformers
        atom_array, processed_reference_molecules = structure_with_ref_mols_from_query(
            query=query,
        )

        # Add token-related IDs
        tokenize_atom_array(atom_array)
        add_token_positions(atom_array)

        return StructureWithReferenceMolecules(
            atom_array, processed_reference_molecules
        )

    def create_structure_features(
        self,
        atom_array: AtomArray,
        processed_reference_molecules: list[ProcessedReferenceMolecule],
        n_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Creates the target structure features."""

        target_structure_features = featurize_structure_of3(
            atom_array=atom_array,
            n_tokens=n_tokens,
            is_gt=False,
            add_perm_features=False,
        )

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=processed_reference_molecules,
            add_ref_space_uid_to_perm=False,
        )

        # Wrap up features
        structure_features = target_structure_features | reference_conformer_features

        return structure_features

    def create_msa_features(self, query, atom_array, n_tokens) -> dict:
        """Creates the MSA features."""

        # [of3-input] per-chain MSA + template summary, one block per chain.
        # Mirrors the format AF3's folding_input.py emits so cross-model
        # comparisons share a structure: source paths + seq counts + size.
        from pathlib import Path as _Path

        def _count_a3m_seqs(p):
            # OF3 may swap a3m paths for binary .npz caches at runtime; skip
            # those (only count text-format a3m / fasta files).
            sp = str(p)
            if not sp.endswith((".a3m", ".fasta", ".fa")):
                return None
            try:
                with open(p, encoding="utf-8", errors="ignore") as fp:
                    return sum(1 for line in fp if line.startswith(">"))
            except OSError:
                return None

        # Featurizer-time per-chain view → inputs_runtime.json under
        # $OF3_OUTPUT_DIR (set by openfold3/inference.py shim).
        runtime_msa: dict = {}
        runtime_tpls: dict = {}

        for chain in query.chains:
            mt = getattr(chain, "molecule_type", None)
            mt_name = mt.name if hasattr(mt, "name") else str(mt)
            seq = getattr(chain, "sequence", "") or ""
            main_paths = chain.main_msa_file_paths or []
            paired_paths = chain.paired_msa_file_paths or []
            tpl_path = getattr(chain, "template_alignment_file_path", None)
            ent_ids = getattr(chain, "template_entry_chain_ids", None)

            logger.info("[of3-input] chain_ids=%s  (%s, len=%d)",
                        chain.chain_ids, mt_name, len(seq))
            cid_list = list(chain.chain_ids) if isinstance(chain.chain_ids, (list, tuple)) else [str(chain.chain_ids)]
            chain_runtime = {"unpaired": None, "paired": None}
            for label, paths, slot in (("main_msa  ", main_paths, "unpaired"),
                                       ("paired_msa", paired_paths, "paired")):
                if not paths:
                    logger.info("  %s: -", label)
                    continue
                for p in paths:
                    n = _count_a3m_seqs(p)   # None for non-a3m (e.g. .npz cache)
                    sz = (_Path(p).stat().st_size if _Path(p).exists() else 0)
                    seq_str = f"seqs={n:,}  " if n is not None else "seqs=?(npz)  "
                    logger.info("  %s: src=%s  %ssize=%dB", label, p, seq_str, sz)
                    if chain_runtime[slot] is None:
                        chain_runtime[slot] = {"src": str(p), "depth": n}
            # Expand homomer entry over all output chain ids — each shares the
            # same MSA / templates by construction (single AF3 entry with
            # `id: ["A","B"]` becomes chains A and B with identical inputs).
            for cid_key in cid_list:
                runtime_msa[cid_key] = chain_runtime
            tpl_entries: list[dict] = []
            if tpl_path:
                # Delegate to inputs_meta.parse_hmmsearch_a3m_hits so we
                # also get template_st/template_en (via the new
                # `template_range:N-M` header field).
                hits: list = []
                tpl_str = str(tpl_path)
                if tpl_str.endswith((".a3m", ".fasta", ".fa")):
                    try:
                        import sys as _sys
                        _scripts = str(_Path(__file__).resolve().parents[6] / "scripts")
                        if _scripts not in _sys.path:
                            _sys.path.insert(0, _scripts)
                        from inputs_meta import parse_hmmsearch_a3m_hits as _phh  # type: ignore
                        hits = _phh(tpl_path)
                    except Exception:
                        hits = []
                logger.info("  templates: src=%s  hits=%d  entry_ids=%s",
                            tpl_path, len(hits), ent_ids or [])
                for i, h in enumerate(hits[:8]):
                    qs, qe = h.get("qs"), h.get("qe")
                    aln = (qe - qs + 1) if (qs is not None and qe is not None) else "?"
                    logger.info(f"    [{i:02d}] {h.get('pdb_chain')}  range={qs}-{qe}  aln_len={aln}")
                for h in hits:
                    entry = dict(h)
                    entry["selected"] = True
                    tpl_entries.append(entry)
                # ColabFold path stores templates in .npz cache (no a3m
                # headers); fall back to entry_chain_ids list — alignment
                # metadata is unavailable but PDB ids are.
                if not tpl_entries and ent_ids:
                    for ec in ent_ids:
                        tpl_entries.append({
                            "pdb_chain": str(ec),
                            "qs": None, "qe": None,
                            "length": None,
                            "selected": True,
                        })
                if len(hits) > 8:
                    logger.info(f"    ... ({len(hits) - 8} more)")
            elif ent_ids:
                logger.info("  templates: entry_ids=%s (no alignment file)", ent_ids)
            else:
                logger.info("  templates: -")
            for cid_key in cid_list:
                runtime_tpls[cid_key] = tpl_entries

        # Create MSA precursor input
        input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
            inference_query=query
        )
        msa_array_collection = self.msa_sample_processor_inference(input=input)

        # After parsing, summarise effective row counts per rep_id and
        # merge those into runtime_msa so summary.json reports the actual
        # featurizer-loaded depth (matters for ColabFold .npz where the
        # input-time count is unavailable, and for any future cap setting).
        for rep_id, main in getattr(msa_array_collection, "rep_id_to_main_msa", {}).items():
            main_depths = {k: len(getattr(v, "aatype", [])) for k, v in main.items()}
            paired = getattr(msa_array_collection, "rep_id_to_paired_msa", {}).get(rep_id, {})
            paired_depths = {k: len(getattr(v, "aatype", [])) for k, v in paired.items()}
            logger.info(
                "MSA depths: rep_id=%s main=%s paired=%s",
                rep_id, main_depths, paired_depths,
            )
            # rep_id maps to a chain (or set of chains for homo-multimer);
            # apply the same depth to every chain that shares this rep_id.
            for chain in query.chains:
                if str(getattr(chain, "rep_id", None)) != str(rep_id):
                    continue
                cid_list = (list(chain.chain_ids)
                            if isinstance(chain.chain_ids, (list, tuple)) and chain.chain_ids
                            else [str(chain.chain_ids)])
                # Take max across rep_id's per-source depths.
                main_total = max(main_depths.values(), default=0) if main_depths else 0
                paired_total = max(paired_depths.values(), default=0) if paired_depths else 0
                for cid in cid_list:
                    if cid not in runtime_msa:
                        continue
                    if runtime_msa[cid].get("unpaired") is not None:
                        runtime_msa[cid]["unpaired"]["depth"] = int(main_total)
                    if runtime_msa[cid].get("paired") is not None and paired_total:
                        runtime_msa[cid]["paired"]["depth"] = int(paired_total)

        # Dump runtime view to $OF3_OUTPUT_DIR/inputs_runtime.json so
        # OF3OutputWriter can attach it to summary.json.inputs.
        try:
            import os as _os, sys as _sys
            out_dir = _os.environ.get("OF3_OUTPUT_DIR")
            if out_dir:
                _scripts = str(_Path(__file__).resolve().parents[6] / "scripts")
                if _scripts not in _sys.path:
                    _sys.path.insert(0, _scripts)
                from inputs_meta import write_runtime_inputs as _wri  # type: ignore
                _wri(out_dir, {"msa": runtime_msa, "templates": runtime_tpls})
        except Exception as _e:
            logger.warning(f"[of3 runtime_inputs] {type(_e).__name__}: {_e}")

        msa_features = self.msa_featurizer_of3(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            n_tokens=n_tokens,
        )

        return msa_features

    def create_template_features(
        self, query: Query, atom_array: AtomArray, n_tokens: int, *args, **kwargs
    ) -> dict:
        """Creates the template features."""

        # Expand pre-chain template data
        assembly_data = {}
        for chain in query.chains:
            tpl_path = chain.template_alignment_file_path
            logger.info(
                "Templates: chain_ids=%s alignment=%s entry_chain_ids=%s",
                chain.chain_ids,
                str(tpl_path) if tpl_path else None,
                chain.template_entry_chain_ids,
            )
            for chain_id in chain.chain_ids:
                assembly_data[chain_id] = {
                    "template_ids": chain.template_entry_chain_ids,
                    "cache_entry_file_path": chain.template_alignment_file_path,
                }

        # Sample processing
        template_slice_collection = process_template_structures_of3(
            atom_array=atom_array,
            n_templates=self.template_settings.n_templates,
            take_top_k=self.template_settings.take_top_k,
            min_n_tokens_per_chain=self.template_settings.min_n_tokens_per_chain,
            template_cache_directory=None,
            assembly_data=assembly_data,
            template_structures_directory=self.template_preprocessor_settings.structure_directory,
            template_structure_array_directory=self.template_preprocessor_settings.structure_array_directory,
            template_file_format=self.template_preprocessor_settings.structure_file_format,
            ccd=self.ccd,
        )

        # Featurization. multimer_templates (if any) flows through to
        # featurize_template_structures_of3, which appends one cross-chain
        # row per entry to the per-chain Templates stack.
        template_features = featurize_template_structures_of3(
            atom_array=atom_array,
            template_slice_collection=template_slice_collection,
            n_templates=self.template_settings.n_templates,
            n_tokens=n_tokens,
            min_bin=self.template_settings.distogram.min_bin,
            max_bin=self.template_settings.distogram.max_bin,
            n_bins=self.template_settings.distogram.n_bins,
            multimer_templates=getattr(query, "multimer_templates", None),
        )

        return template_features

    def create_all_features(
        self,
        query: Query,
    ) -> dict:
        """Creates all features for a single datapoint."""

        features = {}

        # Create initial AtomArray and ReferenceMolecules from query entry
        structure_objs = self.get_structure_with_ref_mols(
            query=query,
        )
        preprocessed_atom_array, processed_reference_molecules = structure_objs

        # TODO: At some point, think about a cleaner way to pass this through the model
        # runner than as a pseudo-feature
        features["atom_array"] = preprocessed_atom_array
        n_tokens = get_token_count(preprocessed_atom_array)

        # Target structure and conformer features
        structure_features = self.create_structure_features(
            atom_array=preprocessed_atom_array,
            processed_reference_molecules=processed_reference_molecules,
            n_tokens=n_tokens,
        )
        features.update(structure_features)

        # MSA features
        msa_features = self.create_msa_features(
            query, preprocessed_atom_array, n_tokens
        )
        features.update(msa_features)

        # Template features
        template_features = self.create_template_features(
            query, preprocessed_atom_array, n_tokens
        )
        features.update(template_features)

        return features

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        # Get query ID and seed information
        datapoint = self.datapoint_cache.iloc[index]
        query_id = datapoint["query_id"]
        query = self.query_cache[query_id]
        seed = datapoint["seed"]
        is_repeated_sample = bool(datapoint["repeated_sample"])

        try:
            features = self.create_all_features(query)
            features["query_id"] = query_id
            features["seed"] = torch.tensor([seed])
            features["repeated_sample"] = torch.tensor(
                [is_repeated_sample], dtype=torch.bool
            )
            features["valid_sample"] = torch.tensor([True], dtype=torch.bool)

            return features
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                "-" * 40
                + "\n"
                + f"Failed to process {query_id} with preferred"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            features = {
                "query_id": query_id,
                "repeated_sample": torch.tensor([is_repeated_sample], dtype=torch.bool),
                "valid_sample": torch.tensor([False], dtype=torch.bool),
            }

            return features

    def __len__(self):
        return len(self.datapoint_cache)
