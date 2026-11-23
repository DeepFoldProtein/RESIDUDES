# SPDX-License-Identifier: Apache-2.0
"""Per-model PyTorch state_dict -> Haiku param tree converters.

Each model exposes:
  pairformer_rules(num_blocks) -> list[Rule]
  token_transformer_rules(num_blocks) -> list[Rule]

Use ``convert.py`` to apply rules end-to-end.
"""
