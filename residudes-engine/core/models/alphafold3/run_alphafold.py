# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import json
import gc
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.data.tools import shards
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import tokamax


_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_SMALL_BFD_Z_VALUE = flags.DEFINE_integer(
    'small_bfd_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_MGNIFY_Z_VALUE = flags.DEFINE_integer(
    'mgnify_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_Z_VALUE = flags.DEFINE_integer(
    'uniprot_cluster_annot_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_UNIREF90_Z_VALUE = flags.DEFINE_integer(
    'uniref90_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_NTRNA_Z_VALUE = flags.DEFINE_float(
    'ntrna_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RFAM_Z_VALUE = flags.DEFINE_float(
    'rfam_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_RNA_CENTRAL_Z_VALUE = flags.DEFINE_float(
    'rna_central_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Jackhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_JACKHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'jackhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Jackhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Nhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_NHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'nhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Nhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)

# Data pipeline configuration.
_RESOLVE_MSA_OVERLAPS = flags.DEFINE_bool(
    'resolve_msa_overlaps',
    True,
    'Whether to deduplicate unpaired MSA against paired MSA. The default'
    ' behaviour matches the method described in the AlphaFold 3 paper. Set this'
    ' to false if providing custom paired MSA using the unpaired MSA field to'
    ' keep it exactly as is as deduplication against the paired MSA could break'
    ' the manually crafted pairing between MSA sequences.',
)
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All'
    ' templates released after this date will be ignored. Controls also whether'
    ' to allow use of model coordinates for a chemical component from the CCD'
    ' if RDKit conformer generation fails and the component does not have ideal'
    ' coordinates set. Only for components that have been released before this'
    ' date the model coordinates can be used as a fallback.',
)
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
    lower_bound=0,
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference, uses zero-based'
    ' indexing. Defaults to the 0th GPU on the system. Useful on multi-GPU'
    ' systems to pin each run to a specific GPU. Note that if GPUs are already'
    ' pre-filtered by the environment (e.g. by using CUDA_VISIBLE_DEVICES),'
    ' this flag refers to the GPU index after the filtering has been done.',
)
_CP_SIZE = flags.DEFINE_integer(
    'cp_size',
    1,
    'Context-parallel world size for the trunk pair modules. Must be 1 (CP off)'
    ' or a perfect square (4, 9, 16, …). When > 1, AF3 splits the pair tensor'
    ' across a √cp_size × √cp_size mesh of GPUs — visible devices are taken in'
    ' order, or pinned via AF3_CP_DEVICES. --gpu_device is ignored in this mode.'
    ' The jit-cached model is bound to the chosen mesh, so toggling between'
    ' values within one process forces a re-trace.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.'
    ' Note that the embeddings are large float16 arrays: num_tokens * 384'
    ' + num_tokens * num_tokens * 128.',
)
_INJECT_EMBEDDINGS_NPZ = flags.DEFINE_string(
    'inject_embeddings_npz',
    None,
    'If set, load (single, pair) embeddings from this npz and feed them to'
    ' the heads DIRECTLY, SKIPPING the trunk. Arrays are passed as JIT'
    ' inputs (NOT HLO constants) to avoid backend-compile segfaults.',
)

# AF3_STREAMING_OUTPUT (default '1'): interleave featurise → infer → write
# per seed so long seed sweeps (e.g. 1000+ seed diffusion-only sampling)
# drop completed seed_<N>/ subtrees onto disk every ~30–60s instead of
# buffering all featurised batches + inference results in memory and only
# writing after the entire sweep finishes (which OOMs the host on multi-
# seed multimer batches and loses everything on stop/kill). Set
# AF3_STREAMING_OUTPUT=0 to fall back to the legacy eager path.
_STREAMING_OUTPUT = os.environ.get('AF3_STREAMING_OUTPUT', '1') == '1'


def _load_inject_embeddings_npz(path):
  """Load injection npz; return dict of uint16 (raw bf16 bits) arrays."""
  if not path:
    return None
  print(f'[af3 inject] loading {path}', flush=True)
  d = np.load(path)

  def _to_uint16(arr):
    if arr.dtype == np.dtype('V2'):
      return np.frombuffer(arr.tobytes(), dtype=np.uint16).reshape(arr.shape)
    if arr.dtype == np.float32:
      return (arr.view(np.uint32) >> 16).astype(np.uint16)
    if arr.dtype == np.float16:
      return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    raise ValueError(f'unsupported dtype for injection: {arr.dtype}')

  pair_u16 = _to_uint16(d['pair_embeddings'])
  single_u16 = _to_uint16(d['single_embeddings'])
  print(f'[af3 inject] pair {pair_u16.shape}, single {single_u16.shape}'
        f'  (uint16 bf16-bits; trunk WILL BE SKIPPED)', flush=True)
  return {'inject_pair_bf16_bits': pair_u16,
          'inject_single_bf16_bits': single_u16}


_SAVE_EMBEDDINGS_DTYPE = flags.DEFINE_enum(
    'save_embeddings_dtype',
    'float16',
    ['float16', 'bfloat16', 'float32'],
    'Numeric dtype for saved trunk embeddings (single/pair). Stored once per'
    ' seed (trunk output is shared across diffusion samples).'
    ' float16 (default upstream): smallest, but limited dynamic range can'
    ' saturate large values.'
    ' bfloat16: same dynamic range as float32, bit-identical to AF3 trunk'
    ' internals — recommended when reloading embeddings to feed the heads.'
    ' NOTE: numpy npz strips the bfloat16 dtype tag on load (returns as'
    ' |V2 opaque void). Reload with `arr.view(ml_dtypes.bfloat16)` to'
    ' recover; bytes are preserved exactly.'
    ' float32: largest (e.g. T1313 A7: ~10.4 GB pair, vs ~5.2 GB at fp16).',
)


def _resolve_embeddings_dtype(name: str):
  """Map --save_embeddings_dtype flag value to numpy/ml_dtypes dtype.

  bfloat16 is not native to numpy; ml_dtypes provides it (always installed
  as a JAX dependency). float16/float32 are numpy-native.
  """
  if name == 'bfloat16':
    import ml_dtypes  # pylint: disable=g-import-not-at-top
    return ml_dtypes.bfloat16
  return np.dtype(name)


_SAVE_DISTOGRAM = flags.DEFINE_bool(
    'save_distogram',
    False,
    'Whether to save the final distogram in the output. Note that the distogram'
    ' is a large float16 array: num_tokens * num_tokens * 64.',
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    'force_output_dir',
    False,
    'Whether to force the output directory to be used even if it already exists'
    ' and is non-empty. Useful to set this to True to run the data pipeline and'
    ' the inference separately, but use the same output directory.',
)
_COMPRESS_LARGE_OUTPUT_FILES = flags.DEFINE_bool(
    'compress_large_output_files',
    False,
    'If True, compresses the output mmCIF and confidences JSON files (the two'
    ' largest files) using zstandard. Note that embeddings and distogram, if'
    ' saved, are already stored in a compressed format.',
)


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  config.return_distogram = return_distogram
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir
    # When CP is active the model is jitted across the whole mesh — there is
    # no single owning device. We still keep `_device` around for diagnostics
    # but skip jax.device_put / jit(device=...) below.
    from alphafold3.model.network import parallelism as _parallelism
    self._cp_enabled = _parallelism.is_enabled()

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
      return model.Model(self._model_config)(batch)

    if self._cp_enabled:
      jitted = jax.jit(forward_fn.apply)
    else:
      jitted = jax.jit(forward_fn.apply, device=self._device)
    return functools.partial(jitted, self.model_params)

  def run_inference(
      self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.tree_util.tree_map(
        jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
    )
    if not self._cp_enabled:
      featurised_example = jax.device_put(featurised_example, self._device)

    result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result = dict(result)
    identifier = self.model_params['__meta__']['__identifier__'].tobytes()
    result['__identifier__'] = identifier
    return result

  def extract_inference_results(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> list[model.InferenceResult]:
    """Extracts inference results from model outputs."""
    return list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )

  def extract_embeddings(
      self,
      result: model.ModelResult,
      num_tokens: int,
      dtype: 'np.dtype | str' = np.float16,
  ) -> dict[str, np.ndarray] | None:
    """Extracts embeddings from model outputs.

    Embeddings come from the trunk (Evoformer + recycle), shared across
    diffusion samples within a seed — saved once per seed.
    """
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][
          :num_tokens
      ].astype(dtype)
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ].astype(dtype)
    return embeddings or None

  def extract_distogram(
      self, result: model.ModelResult, num_tokens: int
  ) -> np.ndarray | None:
    """Extracts distogram from model outputs."""
    if 'distogram' not in result['distogram']:
      return None
    distogram = result['distogram']['distogram'][:num_tokens, :num_tokens, :]
    return distogram


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
    distogram: The token distance histogram, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None
  distogram: np.ndarray | None = None
  runtime_s: float | None = None


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    output_dir: os.PathLike[str] | str | None = None,
    compress_large_output_files: bool = False,
    job_name: str | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
  # Inject embeddings (diffusion-only mode). Loaded once; reused for every seed.
  _inject_arrays = _load_inject_embeddings_npz(_INJECT_EMBEDDINGS_NPZ.value)

  if _STREAMING_OUTPUT and output_dir is not None:
    # Lazy interleave: featurise → infer → write per-seed. Each completed
    # seed_<N>/ subtree hits disk before the next seed starts featurising,
    # so an external watcher / partial pull can make incremental progress
    # and SIGTERM only loses one in-flight seed.
    print(
        f'[af3 streaming] interleaved featurise/infer/write for'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    streaming_iter = featurisation.featurise_input_streaming(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=True,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    _job_name = job_name or fold_input.sanitised_name()
    for seed, example in streaming_iter:
      print(f'Running model inference with seed {seed}...')
      inference_start_time = time.time()
      rng_key = jax.random.PRNGKey(seed)
      if _inject_arrays is not None:
        example = dict(example)
        example.update(_inject_arrays)
      result = model_runner.run_inference(example, rng_key)
      print(
          f'Running model inference with seed {seed} took'
          f' {time.time() - inference_start_time:.2f} seconds.'
      )
      print(f'Extracting inference results with seed {seed}...')
      extract_structures = time.time()
      inference_results = model_runner.extract_inference_results(
          batch=example, result=result, target_name=fold_input.name
      )
      num_tokens = len(inference_results[0].metadata['token_chain_ids'])
      embeddings = model_runner.extract_embeddings(
          result=result, num_tokens=num_tokens,
          dtype=_resolve_embeddings_dtype(_SAVE_EMBEDDINGS_DTYPE.value),
      )
      distogram = model_runner.extract_distogram(
          result=result, num_tokens=num_tokens
      )
      print(
          f'Extracting {len(inference_results)} inference samples with'
          f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
      )
      seed_runtime_s = float(time.time() - inference_start_time)
      results_for_seed = ResultsForSeed(
          seed=seed,
          inference_results=inference_results,
          full_fold_input=fold_input,
          embeddings=embeddings,
          distogram=distogram,
          runtime_s=seed_runtime_s,
      )
      print(f'[af3 streaming] writing seed {seed} to disk...')
      write_outputs(
          all_inference_results=[results_for_seed],
          output_dir=output_dir,
          job_name=_job_name,
          compress_large_output_files=compress_large_output_files,
          streaming=True,
      )
      # After per-seed write the data lives on disk; drop in-memory refs
      # so 999-seed sweeps don't accumulate ~14 GB/seed (pair embeddings
      # + distogram + per-sample structures) until the host OOMs. Track
      # only the seed int so the return type still implies "N seeds done"
      # for the caller. JAX result arrays are released too — without an
      # explicit del they're held alive by `result` until the next iter
      # overwrites it, doubling device-buffer pressure across the loop.
      all_inference_results.append(seed)
      del (
          result,
          inference_results,
          embeddings,
          distogram,
          results_for_seed,
          example,
      )
      gc.collect()
    print(
        'Running model inference and extracting output structures with'
        f' {len(fold_input.rng_seeds)} seed(s) took'
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
      resolve_msa_overlaps=resolve_msa_overlaps,
  )
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()

  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    if _inject_arrays is not None:
      example = dict(example)
      example.update(_inject_arrays)
    result = model_runner.run_inference(example, rng_key)
    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results = model_runner.extract_inference_results(
        batch=example, result=result, target_name=fold_input.name
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = model_runner.extract_embeddings(
        result=result, num_tokens=num_tokens,
        dtype=_resolve_embeddings_dtype(_SAVE_EMBEDDINGS_DTYPE.value),
    )
    distogram = model_runner.extract_distogram(
        result=result, num_tokens=num_tokens
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    seed_runtime_s = float(time.time() - inference_start_time)
    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
            distogram=distogram,
            runtime_s=seed_runtime_s,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
    compress_large_output_files: bool = False,
    streaming: bool = False,
) -> None:
  """Writes outputs to the specified output directory.

  In streaming mode (one ResultsForSeed per call, fired from
  predict_structure as each seed completes), the per-seed dir is written
  every call but cross-seed trailers (top-ranked copy at output_dir, full
  ranking_scores.csv) are deferred — the CSV is APPENDED to so partial
  progress survives a SIGTERM, and the top-ranked copy is skipped (the
  "best" is a moving target during a long sweep; user can derive it from
  the CSV after stopping).
  """
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  # If output_dir is already named `seed_<S>` (CASP17 single-seed-per-sbatch
  # convention), don't add another seed_<S>/ nesting layer. Only add it for
  # multi-seed bulk runs where samples from different seeds would collide.
  _basename = os.path.basename(os.path.abspath(output_dir))
  _single_seed_dir = (
      len(all_inference_results) == 1
      and _basename == f'seed_{all_inference_results[0].seed}'
      and not streaming  # streaming always lands in seed_<N>/ subdir
  )
  # Drop the featurizer-time per-chain view (populated by folding_input
  # while parsing each chain) into <output_dir>/inputs_runtime.json so the
  # unified writer reads it back into summary.json.inputs. In streaming
  # mode this only fires on the first call (the queue is consumed once).
  try:
    _runtime = folding_input.consume_runtime_inputs()
    if _runtime['msa'] or _runtime['templates']:
      with open(os.path.join(output_dir, 'inputs_runtime.json'), 'w') as _f:
        json.dump(_runtime, _f, indent=2)
  except Exception as _e:
    print(f'[af3 inputs_runtime] {type(_e).__name__}: {_e}', flush=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    seed_dir = output_dir if _single_seed_dir else os.path.join(output_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    # Reorder samples within this seed by ranking_score (descending) so
    # `sample_0` is always the top-ranked sample — matches pt2/bt2, fixes
    # of3/af3 native submission-order quirk for cross-model parity.
    sorted_results = sorted(
        results_for_seed.inference_results,
        key=lambda r: float(r.metadata['ranking_score']),
        reverse=True,
    )
    for sample_idx, result in enumerate(sorted_results):
      sample_dir = os.path.join(seed_dir, f'sample_{sample_idx}')
      post_processing.write_output(
          inference_result=result,
          output_dir=sample_dir,
          name=None,  # unified writer doesn't use prefix
          compress=compress_large_output_files,
          sample_idx=sample_idx,
          seed=seed,
          job_name=job_name,
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(seed_dir, 'embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings,
          output_dir=embeddings_dir,
          name=f'{job_name}_seed-{seed}',
      )

    if (distogram := results_for_seed.distogram) is not None:
      # Unified: distogram once per seed (sample-independent) at seed_<seed>/distogram_probs.npy.
      post_processing.write_distogram(distogram=distogram, output_dir=seed_dir)

    if results_for_seed.runtime_s is not None:
      with open(os.path.join(seed_dir, 'timing.json'), 'w') as f:
        json.dump({'runtime_s': float(results_for_seed.runtime_s)}, f, indent=4)

  if streaming:
    # Append-only ranking_scores.csv so partial sweep results survive a
    # mid-run SIGTERM. Skip the cross-seed top-ranked copy at output_dir
    # (would just thrash on every seed; user can derive the best from the
    # CSV after stopping). Header written only on first append.
    csv_path = os.path.join(output_dir, f'{job_name}_ranking_scores.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
      writer = csv.writer(f)
      if write_header:
        writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)
  elif max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
        compress=compress_large_output_files,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(
        os.path.join(output_dir, f'{job_name}_ranking_scores.csv'), 'wt'
    ) as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if (sharded_paths := shards.get_sharded_paths(path_with_db_dir)) is not None:
    db_exists = all(os.path.exists(p) for p in sharded_paths)
  else:
    db_exists = os.path.exists(path_with_db_dir)
  if not db_exists:
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    *,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
    compress_large_output_files: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
      MSA. The default behaviour matches the method described in the AlphaFold 3
      paper. Set this to false if providing custom paired MSA using the unpaired
      MSA field to keep it exactly as is as deduplication against the paired MSA
      could break the manually crafted pairing between MSA sequences.
    force_output_dir: If True, do not create a new output directory even if the
      existing one is non-empty. Instead use the existing output directory and
      potentially overwrite existing files. If False, create a new timestamped
      output directory instead if the existing one is non-empty.
    compress_large_output_files: If True, compress large output files (mmCIF and
      confidences JSON) using zstandard.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'\nRunning fold job {fold_input.name}...')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if (
      not force_output_dir
      and os.path.exists(output_dir)
      and os.listdir(output_dir)
  ):
    new_output_dir = (
        f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
        output_dir=output_dir,
        compress_large_output_files=compress_large_output_files,
        job_name=fold_input.sanitised_name(),
    )
    if not _STREAMING_OUTPUT:
      print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
      write_outputs(
          all_inference_results=all_inference_results,
          output_dir=output_dir,
          job_name=fold_input.sanitised_name(),
          compress_large_output_files=compress_large_output_files,
      )
    else:
      print(
          f'[af3 streaming] {len(all_inference_results)} seed(s) already'
          ' written incrementally; skipping batch write_outputs.'
      )
    output = all_inference_results

  print(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
  return output


def main(_):
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )

  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
    # Surface json_path for the unified-writer's MSA / template provenance.
    # post_processing.write_output reads BIO_TRT_INPUT_PATH lazily.
    os.environ['BIO_TRT_INPUT_PATH'] = str(pathlib.Path(_JSON_PATH.value).resolve())
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  if _RUN_INFERENCE.value and _CP_SIZE.value > 1:
    # Enable the CP mesh before any model construction so that ModelRunner
    # picks up parallelism.is_enabled()=True at __init__ time (its
    # `_cp_enabled` flag is captured then).
    from alphafold3.model.network import parallelism as _af3_parallelism
    _af3_parallelism.enable_cp(size_cp=_CP_SIZE.value)
    print(
        f'CP enabled: size={_CP_SIZE.value} '
        f'(mesh shape={_af3_parallelism.get_mesh().shape if _af3_parallelism.get_mesh() else None})'
    )

  if _RUN_INFERENCE.value:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      compute_capability = float(
          gpu_devices[_GPU_DEVICE.value].compute_capability
      )
      if compute_capability < 6.0:
        raise ValueError(
            'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
            ' https://developer.nvidia.com/cuda-gpus).'
        )
      elif 7.0 <= compute_capability < 8.0:
        xla_flags = os.environ.get('XLA_FLAGS')
        required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
        if not xla_flags or required_flag not in xla_flags:
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
              f' include "{required_flag}".'
          )
        if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the'
              ' --flash_attention_implementation must be set to "xla".'
          )

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n' + '\n'.join(notice) + '\n')

  max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
  if _RUN_DATA_PIPELINE.value:
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
        small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
        small_bfd_z_value=_SMALL_BFD_Z_VALUE.value,
        mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
        mgnify_z_value=_MGNIFY_Z_VALUE.value,
        uniprot_cluster_annot_database_path=expand_path(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniprot_cluster_annot_z_value=_UNIPROT_CLUSTER_ANNOT_Z_VALUE.value,
        uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
        uniref90_z_value=_UNIREF90_Z_VALUE.value,
        ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
        ntrna_z_value=_NTRNA_Z_VALUE.value,
        rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
        rfam_z_value=_RFAM_Z_VALUE.value,
        rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
        rna_central_z_value=_RNA_CENTRAL_Z_VALUE.value,
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        jackhmmer_max_parallel_shards=_JACKHMMER_MAX_PARALLEL_SHARDS.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
        nhmmer_max_parallel_shards=_NHMMER_MAX_PARALLEL_SHARDS.value,
        max_template_date=max_template_date,
    )
  else:
    data_pipeline_config = None

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend='gpu')
    print(
        f'Found local devices: {devices}, using device {_GPU_DEVICE.value}:'
        f' {devices[_GPU_DEVICE.value]}'
    )

    print('Building model from scratch...')
    print(
        f'AF3 num_recycles = {_NUM_RECYCLES.value}  '
        f'num_diffusion_samples = {_NUM_DIFFUSION_SAMPLES.value}',
        flush=True,
    )
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                tokamax.DotProductAttentionImplementation,
                _FLASH_ATTENTION_IMPLEMENTATION.value,
            ),
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
            return_distogram=_SAVE_DISTOGRAM.value,
        ),
        device=devices[_GPU_DEVICE.value],
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
  else:
    model_runner = None

  num_fold_inputs = 0
  for fold_input in fold_inputs:
    if _NUM_SEEDS.value is not None:
      print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
      fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
    # For --input_dir (bulk) mode, refresh BIO_TRT_INPUT_PATH per fold_input
    # so post_processing.write_output sees the right input.json (not the
    # dir set above).
    if _INPUT_DIR.value is not None:
      _candidate = pathlib.Path(_INPUT_DIR.value) / f'{fold_input.sanitised_name()}.json'
      if _candidate.exists():
        os.environ['BIO_TRT_INPUT_PATH'] = str(_candidate.resolve())
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
        buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
        ref_max_modified_date=max_template_date,
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
        resolve_msa_overlaps=_RESOLVE_MSA_OVERLAPS.value,
        force_output_dir=_FORCE_OUTPUT_DIR.value,
        compress_large_output_files=_COMPRESS_LARGE_OUTPUT_FILES.value,
    )
    num_fold_inputs += 1

  print(f'Done running {num_fold_inputs} fold jobs.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
