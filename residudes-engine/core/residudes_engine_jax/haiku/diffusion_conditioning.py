# SPDX-License-Identifier: Apache-2.0
"""Diffusion conditioning modules in JAX (Algorithms 21, 22, 3 of AF3).

Reference:
  models/protenix/protenix/model/modules/embedders.py::FourierEmbedding
  models/protenix/protenix/model/modules/embedders.py::RelativePositionEncoding
  models/protenix/protenix/model/modules/diffusion.py::DiffusionConditioning

Building blocks for the JAX port of ``DiffusionModule.forward`` — these
prepare the per-step conditioning vectors (single ``s`` and pair ``z``)
that feed into the atom-attention encoder/decoder and token transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp

from residudes_engine_jax.haiku.primitives import LayerNorm, Linear


# ── FourierEmbedding (Algorithm 22) ───────────────────────────────────────

class FourierEmbedding(hk.Module):
    """``cos(2π * (t·w + b))`` with non-trainable random ``w, b``.

    PT stores ``w`` and ``b`` as ``nn.Parameter(requires_grad=False)``, so the
    state_dict has them. We treat them as Haiku params too so the converter
    can copy them verbatim.
    """

    def __init__(self, *, c: int, name: str | None = None):
        super().__init__(name=name)
        self.c = c

    def __call__(self, t_hat_noise_level: jax.Array) -> jax.Array:
        # PT init: torch.randn with seed=42 — we just expose params, the
        # converter copies the values from the loaded state_dict.
        w = hk.get_parameter("w", (self.c,), jnp.float32, init=jnp.zeros)
        b = hk.get_parameter("b", (self.c,), jnp.float32, init=jnp.zeros)
        # t_hat_noise_level: [..., N_sample]
        # output:           [..., N_sample, c]
        return jnp.cos(2.0 * jnp.pi * (t_hat_noise_level[..., None] * w + b))


# ── RelativePositionEncoding (Algorithm 3) ────────────────────────────────

@dataclass(frozen=True)
class RelPEConfig:
    r_max: int = 32
    s_max: int = 2
    c_z: int = 128

    @property
    def in_features(self) -> int:
        return 4 * self.r_max + 2 * self.s_max + 7


class RelativePositionEncoding(hk.Module):
    """Single Linear projection of the precomputed relp feature.

    The 'relp' tensor (one-hot relative position + chain + entity features) is
    computed once on the host side and supplied as an input. We don't
    re-implement ``generate_relp`` in JAX; it has no learnable params and is
    cheap on the PT side.
    """

    def __init__(self, *, config: RelPEConfig, name: str | None = None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, relp_feature: jax.Array) -> jax.Array:
        # relp_feature: [..., N_token, N_token, in_features]
        # output:       [..., N_token, N_token, c_z]
        return Linear(
            self.config.c_z, use_bias=False, name="linear_no_bias",
        )(relp_feature)


# ── Transition (SwiGLU, Algorithm 11) - reused for diffusion conditioning ──

class TransitionResidual(hk.Module):
    """Protenix's Transition block. Two parallel Linears, SwiGLU, output.

    This duplicates the structural Transition in haiku/transition.py but uses
    Protenix-style param names (``layernorm1`` / ``linear_no_bias_a`` /
    ``linear_no_bias_b`` / ``linear_no_bias``) so the converter has a 1:1
    mapping.

    Forward is a residual-friendly *update* — caller does ``x = x + module(x)``.
    """

    def __init__(self, *, c_in: int, n: int = 2, name: str | None = None):
        super().__init__(name=name)
        self.c_in = c_in
        self.n = n

    def __call__(self, x: jax.Array) -> jax.Array:
        x = LayerNorm(name="layernorm1")(x)
        a = Linear(self.n * self.c_in, use_bias=False, initializer="relu",
                   name="linear_no_bias_a")(x)
        b = Linear(self.n * self.c_in, use_bias=False, initializer="relu",
                   name="linear_no_bias_b")(x)
        return Linear(self.c_in, use_bias=False, initializer="zeros",
                      name="linear_no_bias")(jax.nn.silu(a) * b)


# ── DiffusionConditioning (Algorithm 21) ──────────────────────────────────

@dataclass(frozen=True)
class DiffusionConditioningConfig:
    sigma_data: float = 16.0
    c_z: int = 128
    c_s: int = 384
    c_s_inputs: int = 449
    c_noise_embedding: int = 256
    relpe: RelPEConfig = RelPEConfig()


class DiffusionConditioning(hk.Module):
    """Algorithm 21: assemble ``(s_single, pair_z)`` conditioning.

    Pair path:
        z = concat([z_trunk, RelPE(relp)], -1)
        z = Linear(LayerNorm(z))
        z = z + Transition(z)            # x2

    Single path:
        s = concat([s_trunk, s_inputs], -1)
        s = Linear(LayerNorm(s))
        n = FourierEmbedding(log(t_hat / sigma_data) / 4)
        s = s.unsqueeze(N_sample) + Linear(LayerNorm(n)).unsqueeze(N_token)
        s = s + Transition(s)            # x2

    Returns ``(s_single, pair_z)`` of shapes
    ``[..., N_sample, N_token, c_s]`` and ``[..., N_token, N_token, c_z]``.
    """

    def __init__(
        self, *, config: DiffusionConditioningConfig, name: str = "diffusion_conditioning",
    ):
        super().__init__(name=name)
        self.config = config

    @hk.transparent
    def _pair_conditioning(self, z_trunk, relp_feature):
        cfg = self.config
        relp_z = RelativePositionEncoding(config=cfg.relpe, name="relpe")(relp_feature)
        pair_z = jnp.concatenate([z_trunk, relp_z], axis=-1)  # [..., N, N, 2c_z]
        pair_z = Linear(
            cfg.c_z, use_bias=False, name="linear_no_bias_z",
        )(LayerNorm(create_offset=False, name="layernorm_z")(pair_z))
        pair_z = pair_z + TransitionResidual(c_in=cfg.c_z, n=2, name="transition_z1")(pair_z)
        pair_z = pair_z + TransitionResidual(c_in=cfg.c_z, n=2, name="transition_z2")(pair_z)
        return pair_z

    def __call__(
        self,
        t_hat_noise_level: jax.Array,   # [..., N_sample]
        relp_feature: jax.Array,        # [..., N_token, N_token, in_features]
        s_inputs: jax.Array,            # [..., N_token, c_s_inputs]
        s_trunk: jax.Array,             # [..., N_token, c_s]
        z_trunk: jax.Array,             # [..., N_token, N_token, c_z]
        pair_z: jax.Array | None = None,
        use_conditioning: bool = True,
    ):
        cfg = self.config

        if not use_conditioning:
            s_trunk = jnp.zeros_like(s_trunk)
            z_trunk = jnp.zeros_like(z_trunk)

        # --- pair path ---
        if pair_z is None:
            pair_z = self._pair_conditioning(z_trunk, relp_feature)
        # else: pair_z passed in as cache → use directly

        # --- single path ---
        single_s = jnp.concatenate([s_trunk, s_inputs], axis=-1)  # [..., N, c_s + c_s_inputs]
        single_s = Linear(
            cfg.c_s, use_bias=False, name="linear_no_bias_s",
        )(LayerNorm(create_offset=False, name="layernorm_s")(single_s))

        # Noise level Fourier embedding
        # t_hat -> log(t_hat / sigma_data) / 4
        log_t = jnp.log(t_hat_noise_level / cfg.sigma_data) / 4.0
        noise_n = FourierEmbedding(c=cfg.c_noise_embedding, name="fourier_embedding")(log_t)
        # noise_n: [..., N_sample, c_noise]
        noise_n = Linear(
            cfg.c_s, use_bias=False, name="linear_no_bias_n",
        )(LayerNorm(create_offset=False, name="layernorm_n")(noise_n))
        # noise_n: [..., N_sample, c_s]

        # single_s broadcast: [..., N_token, c_s] -> [..., 1, N_token, c_s]
        # noise_n broadcast:  [..., N_sample, c_s] -> [..., N_sample, 1, c_s]
        # sum: [..., N_sample, N_token, c_s]
        single_s = single_s[..., None, :, :] + noise_n[..., :, None, :]

        single_s = single_s + TransitionResidual(c_in=cfg.c_s, n=2, name="transition_s1")(single_s)
        single_s = single_s + TransitionResidual(c_in=cfg.c_s, n=2, name="transition_s2")(single_s)
        return single_s, pair_z
