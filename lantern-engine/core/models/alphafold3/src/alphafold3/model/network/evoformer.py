# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Evoformer network."""

import functools
import os
import threading

import numpy as np

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from alphafold3.model.network import atom_cross_attention
from alphafold3.model.network import featurization
from alphafold3.model.network import modules
from alphafold3.model.network import template_modules
import haiku as hk
import jax
import jax.numpy as jnp


# Opt-in: when AF3_CROSS_CHAIN_TEMPLATES=1, allow template pair features to
# populate inter-chain (off-diagonal) entries of the pair grid. Default off
# preserves upstream bit-for-bit. See _embed_template_pair below for the
# branch and templates.py for the matching data-layer relaxation.
_ALLOW_CROSS_CHAIN_TEMPLATES = (
    os.environ.get('AF3_CROSS_CHAIN_TEMPLATES', '0') == '1'
)

# Opt-in: gate the template embedder contribution to the first N recycles
# only (0-indexed; recycle_idx < N → active, else multiplied by 0). Default
# unset → template active in every recycle (upstream behavior). Lower N
# yields higher seed/sample diversity since later recycles refine without
# template prior.
def _max_template_recycles() -> int | None:
  v = os.environ.get('AF3_MAX_TEMPLATE_RECYCLES')
  if v is None or v == '':
    return None
  try:
    return int(v)
  except ValueError:
    return None
_MAX_TEMPLATE_RECYCLES = _max_template_recycles()


# Opt-in: AF3_TPL_WEIGHT=<float> scales the template embedder contribution
# to the pair representation before it is added back (pair + w * template_act).
# >1 lets a docking-quality multimer template dominate over MSA signal without
# enabling full diffusion guidance; <1 softens it. Default unset → 1.0 =
# upstream behavior.
def _tpl_weight() -> float:
  v = os.environ.get('AF3_TPL_WEIGHT')
  if v is None or v == '':
    return 1.0
  try:
    return float(v)
  except ValueError:
    return 1.0
_TPL_WEIGHT = _tpl_weight()


# Opt-in: when AF3_TEMPLATE_DUMP_PATH=<file.npz> is set, dump a snapshot of
# the post-embedder template_act (channel-norm), multichain_mask, and
# asym_id to that file on the first forward pass of each process. Used
# offline by gen_inputs/viz_template_mask.py --runtime to confirm that
# template signal actually populates the off-diagonal entries the mask
# permits. Default unset = no dump (zero overhead).
_TEMPLATE_DUMP_PATH = os.environ.get('AF3_TEMPLATE_DUMP_PATH', '') or None
_template_dump_lock = threading.Lock()
_template_dump_done = False


def _maybe_dump_template_runtime(
    template_act_norm: jnp.ndarray,
    multichain_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
) -> None:
  """Dump first-call snapshot to AF3_TEMPLATE_DUMP_PATH and ignore later
  calls. JIT-compatible: routed via jax.debug.callback so the host write
  happens after the device computation. The file written:
    template_act_norm : (N, N) float32   ||template_act[i,j,:]||_2
    multichain_mask   : (N, N) int8      0/1 gating mask
    asym_id           : (N,)   int32     chain index per token
  """
  if not _TEMPLATE_DUMP_PATH:
    return

  def _host_write(act_norm_np, mask_np, asym_np):
    global _template_dump_done
    with _template_dump_lock:
      if _template_dump_done:
        return
      out_dir = os.path.dirname(_TEMPLATE_DUMP_PATH) or '.'
      os.makedirs(out_dir, exist_ok=True)
      np.savez(
          _TEMPLATE_DUMP_PATH,
          template_act_norm=np.asarray(act_norm_np).astype(np.float32),
          multichain_mask=np.asarray(mask_np).astype(np.int8),
          asym_id=np.asarray(asym_np).astype(np.int32),
      )
      _template_dump_done = True

  jax.debug.callback(
      _host_write, template_act_norm, multichain_mask, asym_id
  )


# Opt-in: when AF3_CYCLIC_PERIOD_CHAIN=N (N >= 2), apply modulo-N wrap to
# the inter-chain rel_sym_id in create_relative_encoding. Encodes Cn-cyclic
# homomer assemblies (e.g. T1313 A7 → N=7) so the trunk sees chain pair
# (A,G) as a cyclic neighbor of A. Default 0 = upstream behavior.
def _cyclic_period_chain() -> int:
  v = os.environ.get('AF3_CYCLIC_PERIOD_CHAIN', '0')
  try:
    return int(v)
  except ValueError:
    return 0
_CYCLIC_PERIOD_CHAIN = _cyclic_period_chain()


# Opt-in: AF3_CYCLIC_CHAIN_BIN_AMPLIFY="m0,m1,m2,m3,m4,m5" (6 floats)
# applies per-bin multiplier to rel_chain one-hot before the position
# Linear consumes it. With max_relative_chain=2, 6 bins map to:
#   bin 0: rel<=-2 (far backward)
#   bin 1: rel=-1  (cyclic neighbor backward)
#   bin 2: rel=0   (intra)
#   bin 3: rel=+1  (cyclic neighbor forward; AF3_CYCLIC_PERIOD_CHAIN sends
#                   wrapped (A,G) rel=-6 → +1 here)
#   bin 4: rel>=+2 (far forward)
#   bin 5: sentinel (different entity)
# Recommended for Cn ring: "0.2,2.5,1.0,2.5,0.2,1.0" — amplify cyclic
# neighbors (bins 1,3) ×2.5, dampen far chains (bins 0,4) ×0.2.
def _cyclic_chain_bin_amplify():
  v = os.environ.get('AF3_CYCLIC_CHAIN_BIN_AMPLIFY')
  if not v:
    return None
  try:
    return [float(x) for x in v.split(',')]
  except ValueError:
    return None
_CYCLIC_CHAIN_BIN_AMPLIFY = _cyclic_chain_bin_amplify()


class Evoformer(hk.Module):
  """Creates 'single' and 'pair' embeddings."""

  class PairformerConfig(modules.PairFormerIteration.Config):  # pytype: disable=invalid-function-definition
    block_remat: bool = False
    remat_block_size: int = 8

  class Config(base_config.BaseConfig):
    """Configuration for Evoformer."""

    max_relative_chain: int = 2
    msa_channel: int = 64
    seq_channel: int = 384
    max_relative_idx: int = 32
    num_msa: int = 1024
    pair_channel: int = 128
    pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
        single_transition=base_config.autocreate(),
        single_attention=base_config.autocreate(),
        num_layer=48,
    )
    per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
        base_config.autocreate(
            per_token_channels=384,
            per_atom_channels=128,
            atom_transformer=base_config.autocreate(
                num_intermediate_factor=2,
                num_blocks=3,
            ),
            per_atom_pair_channels=16,
        )
    )
    template: template_modules.TemplateEmbedding.Config = (
        base_config.autocreate()
    )
    msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='evoformer',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(
      self, batch: feat_batch.Batch, pair_activations: jnp.ndarray
  ) -> jnp.ndarray:
    """Add relative position encodings."""
    rel_feat = featurization.create_relative_encoding(
        seq_features=batch.token_features,
        max_relative_idx=self.config.max_relative_idx,
        max_relative_chain=self.config.max_relative_chain,
        cyclic_period_chain=_CYCLIC_PERIOD_CHAIN,
        cyclic_chain_bin_amplify=_CYCLIC_CHAIN_BIN_AMPLIFY,
    )
    rel_feat = rel_feat.astype(pair_activations.dtype)

    pair_activations += hm.Linear(
        self.config.pair_channel, name='position_activations'
    )(rel_feat)
    return pair_activations

  @hk.transparent
  def _seq_pair_embedding(
      self,
      token_features: features.TokenFeatures,
      target_feat: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generated Pair embedding from sequence."""
    left_single = hm.Linear(self.config.pair_channel, name='left_single')(
        target_feat
    )[:, None]
    right_single = hm.Linear(self.config.pair_channel, name='right_single')(
        target_feat
    )[None]
    dtype = left_single.dtype
    pair_activations = left_single + right_single
    num_residues = pair_activations.shape[0]
    assert pair_activations.shape == (
        num_residues,
        num_residues,
        self.config.pair_channel,
    )
    mask = token_features.mask
    pair_mask = (mask[:, None] * mask[None, :]).astype(dtype)
    assert pair_mask.shape == (num_residues, num_residues)
    return pair_activations, pair_mask  # pytype: disable=bad-return-type  # jax-ndarray

  @hk.transparent
  def _embed_bonds(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
  ) -> jnp.ndarray:
    """Embeds bond features and merges into pair activations."""
    # Construct contact matrix.
    num_tokens = batch.token_features.token_index.shape[0]
    contact_matrix = jnp.zeros((num_tokens, num_tokens))

    tokens_to_polymer_ligand_bonds = (
        batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    )
    gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
    gather_mask_polymer_ligand = (
        tokens_to_polymer_ligand_bonds.gather_mask.prod(axis=1).astype(
            gather_idxs_polymer_ligand.dtype
        )[:, None]
    )
    # If valid mask then it will be all 1's, so idxs should be unchanged.
    gather_idxs_polymer_ligand = (
        gather_idxs_polymer_ligand * gather_mask_polymer_ligand
    )

    tokens_to_ligand_ligand_bonds = (
        batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    )
    gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
    gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
        axis=1
    ).astype(gather_idxs_ligand_ligand.dtype)[:, None]
    gather_idxs_ligand_ligand = (
        gather_idxs_ligand_ligand * gather_mask_ligand_ligand
    )

    gather_idxs = jnp.concatenate(
        [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
    )
    contact_matrix = contact_matrix.at[
        gather_idxs[:, 0], gather_idxs[:, 1]
    ].set(1.0)

    # Because all the padded index's are 0's.
    contact_matrix = contact_matrix.at[0, 0].set(0.0)

    bonds_act = hm.Linear(self.config.pair_channel, name='bond_embedding')(
        contact_matrix[:, :, None].astype(pair_activations.dtype)
    )
    return pair_activations + bonds_act

  @hk.transparent
  def _embed_template_pair(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
      recycle_idx=None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embeds Templates and merges into pair activations."""
    dtype = pair_activations.dtype
    key, subkey = jax.random.split(key)
    template_module = template_modules.TemplateEmbedding(
        self.config.template, self.global_config
    )
    templates = batch.templates
    asym_id = batch.token_features.asym_id
    # Selective multichain_mask. Block-diagonal (intra-chain) is always
    # unmasked. For pair (i, j) with chain(i) != chain(j), unmask iff some
    # *cross-chain* template covers both endpoints. We restrict the OR to
    # rows flagged is_cross_chain because per-chain templates from
    # different chains share a template row index after the data
    # pipeline's pad-axis-0 + concat-axis-1 (features.py) — OR-ing over
    # all rows would spuriously open A×B whenever A's k-th hit and B's
    # k-th hit happen to land in the same row. Cross-chain templates
    # (declared via `multimerTemplates` in the input JSON, or
    # auto-detected when AF3_CROSS_CHAIN_TEMPLATES=1) carry genuine
    # inter-chain geometry and rightfully open the off-diagonal blocks
    # they cover.
    # templates.atom_mask: (n_templates, n_tokens, n_atoms), dtype=bool
    tok_covered = jnp.any(templates.atom_mask, axis=-1).astype(jnp.int32)
    cc_only = tok_covered * templates.is_cross_chain.astype(jnp.int32)[:, None]
    pair_covered = (cc_only.T @ cc_only > 0).astype(dtype)
    block_diag = (asym_id[:, None] == asym_id[None, :]).astype(dtype)
    multichain_mask = jnp.maximum(block_diag, pair_covered)

    template_fn = functools.partial(template_module, key=subkey)
    template_act = template_fn(
        query_embedding=pair_activations,
        templates=templates,
        multichain_mask_2d=multichain_mask,
        padding_mask_2d=pair_mask,
    )

    # Optional: snapshot template_act (channel-norm) + mask for offline
    # visualization. Gated by AF3_TEMPLATE_DUMP_PATH; first call only.
    if _TEMPLATE_DUMP_PATH:
      template_act_norm = jnp.linalg.norm(template_act, axis=-1)
      _maybe_dump_template_runtime(template_act_norm, multichain_mask, asym_id)

    # Optional: gate template contribution to the first N recycles only.
    # When AF3_MAX_TEMPLATE_RECYCLES=N is set and recycle_idx is threaded
    # from the recycle loop, mask out template_act once recycle_idx >= N
    # so later recycles refine without the template prior — boosts diversity
    # across diffusion samples / seeds. Default unset = upstream (always on).
    if _MAX_TEMPLATE_RECYCLES is not None and recycle_idx is not None:
      active = jnp.less(
          jnp.asarray(recycle_idx), _MAX_TEMPLATE_RECYCLES
      ).astype(template_act.dtype)
      template_act = template_act * active

    # AF3_TPL_WEIGHT: scale the template contribution before it is added to
    # the pair representation. Applied after the recycle gate so it composes
    # with AF3_MAX_TEMPLATE_RECYCLES. Default 1.0 = no change.
    if _TPL_WEIGHT != 1.0:
      template_act = template_act * jnp.asarray(
          _TPL_WEIGHT, dtype=template_act.dtype
      )

    return pair_activations + template_act, key

  @hk.transparent
  def _embed_process_msa(
      self,
      msa_batch: features.MSA,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
      target_feat: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Processes MSA and returns updated pair activations."""
    dtype = pair_activations.dtype
    msa_batch, key = featurization.shuffle_msa(key, msa_batch)
    msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
    msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

    msa_activations = hm.Linear(
        self.config.msa_channel, name='msa_activations'
    )(msa_feat)

    msa_activations += hm.Linear(
        self.config.msa_channel, name='extra_msa_target_feat'
    )(target_feat)[None]
    msa_mask = msa_batch.mask.astype(dtype)

    # Evoformer MSA stack.
    evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
    masks = {'msa': msa_mask, 'pair': pair_mask}

    def evoformer_fn(x):
      return modules.EvoformerIteration(
          self.config.msa_stack, self.global_config, name='msa_stack'
      )(
          activations=x,
          masks=masks,
      )

    evoformer_stack = hk.experimental.layer_stack(
        self.config.msa_stack.num_layer
    )(evoformer_fn)

    evoformer_output = evoformer_stack(evoformer_input)

    return evoformer_output['pair'], key

  def __call__(
      self,
      batch: feat_batch.Batch,
      prev: dict[str, jnp.ndarray],
      target_feat: jnp.ndarray,
      key: jnp.ndarray,
      recycle_idx=None,
  ) -> dict[str, jnp.ndarray]:

    assert self.global_config.bfloat16 in {'all', 'none'}

    num_residues = target_feat.shape[0]
    assert batch.token_features.aatype.shape == (num_residues,)

    dtype = (
        jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32
    )

    with utils.bfloat16_context():
      pair_activations, pair_mask = self._seq_pair_embedding(
          batch.token_features, target_feat
      )

      pair_activations += hm.Linear(
          pair_activations.shape[-1],
          name='prev_embedding',
          initializer=self.global_config.final_init,
      )(
          hm.LayerNorm(name='prev_embedding_layer_norm')(
              prev['pair'].astype(pair_activations.dtype)
          )
      )

      pair_activations = self._relative_encoding(batch, pair_activations)

      pair_activations = self._embed_bonds(
          batch=batch, pair_activations=pair_activations
      )

      pair_activations, key = self._embed_template_pair(
          batch=batch,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
          recycle_idx=recycle_idx,
      )
      pair_activations, key = self._embed_process_msa(
          msa_batch=batch.msa,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
          target_feat=target_feat,
      )
      del key  # Unused after this point.

      single_activations = hm.Linear(
          self.config.seq_channel, name='single_activations'
      )(target_feat)

      single_activations += hm.Linear(
          single_activations.shape[-1],
          name='prev_single_embedding',
          initializer=self.global_config.final_init,
      )(
          hm.LayerNorm(name='prev_single_embedding_layer_norm')(
              prev['single'].astype(single_activations.dtype)
          )
      )

      def pairformer_fn(x):
        pairformer_iteration = modules.PairFormerIteration(
            self.config.pairformer,
            self.global_config,
            with_single=True,
            name='trunk_pairformer',
        )
        pair_act, single_act = x
        return pairformer_iteration(
            act=pair_act,
            single_act=single_act,
            pair_mask=pair_mask,
            seq_mask=batch.token_features.mask.astype(dtype),
        )

      pairformer_stack = hk.experimental.layer_stack(
          self.config.pairformer.num_layer
      )(pairformer_fn)

      pair_activations, single_activations = pairformer_stack(
          (pair_activations, single_activations)
      )

      assert pair_activations.shape == (
          num_residues,
          num_residues,
          self.config.pair_channel,
      )
      assert single_activations.shape == (num_residues, self.config.seq_channel)
      assert len(target_feat.shape) == 2
      assert target_feat.shape[0] == num_residues
      output = {
          'single': single_activations,
          'pair': pair_activations,
          'target_feat': target_feat,
      }

    return output
