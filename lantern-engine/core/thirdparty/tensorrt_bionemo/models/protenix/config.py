# SPDX-License-Identifier: Apache-2.0
"""Protenix model config — same dims as OF3 but used for weight-name mapping."""

from tensorrt_bionemo.configs import (BaseConfig, DiffusionTransformerConfig,
                                      PairformerConfig)


class ProtenixConfig(BaseConfig):
    c_z: int = 128
    c_s: int = 384

    trunk: BaseConfig = BaseConfig(pairformer=PairformerConfig(
        token_s=384,
        token_z=128,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=48,
        num_heads=16,
        trimul_high_precision=False,
        version="v1",
        dtype="float32"), )

    structure_module: BaseConfig = BaseConfig(
        score_model=BaseConfig(token_transformer=DiffusionTransformerConfig(
            num_blocks=24,
            num_heads=16,
            dim=768,
            dim_single_cond=384,
            dim_pairwise=128,
            expansion_factor=2,
            bias_proj=True,
            conditioned_transition_using_silu=True,
            attention_initial_norm=False,
            post_layer_norm=False,
            version="v1",
            dtype="float32"), ))
