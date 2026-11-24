# SPDX-License-Identifier: Apache-2.0
"""Haiku building blocks for the JAX hot-module backend.

Layout:
  primitives.py  - Linear, LayerNorm matching AF3 weight conventions
  triangle.py    - TriangleMultiplication + TriangleAttention (=GridSelfAttention)
  transition.py  - SwiGLU TransitionBlock + AttentionPairBias (diffusion track)
  pairformer.py  - PairformerBlock + PairformerStack (per-model variant config)

Reference: alphafold3/src/alphafold3/model/network/{modules,evoformer}.py.
We vendor a minimal subset rather than depend on the full alphafold3 package
because AF3 has additional license terms on its source tree.

Triangle attention, AdaLN single-track attention, and windowed atom
cross-attention all route through ``_tokamax.dot_product_attention``. When
the JAX-only env (.venv-jax / pyproject-jax.toml) is active, tokamax's Pallas
Triton flash kernel runs (~5x faster than XLA at N=512, BF16). When tokamax
is absent (legacy .venv), the wrapper falls back to
``jax.nn.dot_product_attention(impl='cudnn')`` and finally XLA. See
``docs/jax_venv.md``.
"""
