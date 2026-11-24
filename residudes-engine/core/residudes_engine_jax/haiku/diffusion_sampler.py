# SPDX-License-Identifier: Apache-2.0
"""Diffusion sampler in pure JAX.

Reference:
  alphafold3/src/alphafold3/model/network/diffusion_head.py::sample
  models/protenix/protenix/model/generator.py::sample_diffusion

We follow AF3's ``hk.scan``-based design but implement Protenix's specific
algorithm (Algorithm 18) — the math differs from AF3 only in:
  - ``gamma`` is a *fixed* gamma_0 when c_tau > gamma_min, else 0
    (AF3 multiplies gamma_0 by the boolean directly — same result).
  - The "noise_level_prev" used in ``t_hat`` and the noise scale formula is
    ``c_tau_last`` here (the previous step's level), matching Protenix.

The whole loop becomes a single ``jax.lax.scan`` so that JIT can fuse the
200 denoising steps + augmentation + EDM-Heun update into one CUDA graph.
The denoising function is passed in as a callable; the caller decides
whether it's pure-JAX (fast path) or a host-callback into PyTorch.

Usage::

    schedule = inference_noise_schedule(n_steps=200, ...)
    positions = sample(
        denoising_step=my_denoise,        # callable: (pos[N,3], t_hat) -> pos
        noise_schedule=schedule,
        mask=atom_mask,                    # [N_atom]
        key=jax.random.PRNGKey(seed),
        num_samples=5,
    )
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# ── Sampler helpers ───────────────────────────────────────────────────────

def random_rotation(key: jax.Array) -> jax.Array:
    """Haar-uniform 3D rotation matrix via QR decomposition.

    Matches Protenix's ``uniform_random_rotation`` (also haar-uniform).
    AF3 uses a Gram-Schmidt approximation which is slightly biased; we
    pick QR for parity with Protenix.
    """
    a = jax.random.normal(key, (3, 3))
    q, r = jnp.linalg.qr(a)
    # Force sign of diag(r) positive so q is uniformly distributed
    q = q * jnp.sign(jnp.diag(r))
    # If det == -1, flip last column to get a proper rotation (det == +1)
    det = jnp.linalg.det(q)
    q = q.at[:, -1].set(q[:, -1] * det)
    return q


def centre_random_augmentation(
    key: jax.Array,
    positions: jax.Array,        # [N_atom, 3]
    mask: jax.Array | None = None,
    s_trans: float = 1.0,
) -> jax.Array:
    """Algorithm 19: center, random rotation, random translation."""
    rot_key, trans_key = jax.random.split(key)
    if mask is not None:
        denom = mask.sum(axis=-1, keepdims=True) + 1e-12
        centre = (positions * mask[..., None]).sum(axis=-2, keepdims=True) / denom[..., None]
        positions = positions - centre
    else:
        positions = positions - positions.mean(axis=-2, keepdims=True)

    rot = random_rotation(rot_key)
    trans = s_trans * jax.random.normal(trans_key, (3,))
    out = jnp.einsum("...i,ij->...j", positions, rot) + trans
    if mask is not None:
        out = out * mask[..., None]
    return out


def inference_noise_schedule(
    n_steps: int = 200,
    s_min: float = 0.0004,
    s_max: float = 160.0,
    rho: float = 7.0,
    sigma_data: float = 16.0,
) -> jax.Array:
    """Protenix's inference schedule (Karras EDM, descending).

    Returns array of length ``n_steps + 1`` where index 0 is the largest
    noise level (s_max) and index n_steps is approximately s_min.
    """
    t = jnp.linspace(0.0, 1.0, n_steps + 1)
    inv_rho = 1.0 / rho
    return sigma_data * (
        s_max ** inv_rho + t * (s_min ** inv_rho - s_max ** inv_rho)
    ) ** rho


# ── Main sampler (JIT-friendly via lax.scan) ──────────────────────────────

def sample(
    denoising_step: Callable[[jax.Array, jax.Array], jax.Array],
    noise_schedule: jax.Array,        # [steps + 1]
    mask: jax.Array,                  # [N_atom]
    key: jax.Array,
    *,
    num_samples: int = 5,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
) -> jax.Array:
    """Run ``len(noise_schedule) - 1`` denoising steps.

    Args:
      denoising_step: pure function ``(positions[N_atom, 3], t_hat) ->
          denoised_positions[N_atom, 3]``. Will be vmap'd over n_samples.
      noise_schedule: descending noise levels, length ``n_steps + 1``.
      mask: per-atom mask, shape ``[N_atom]`` (0/1 floats).
      key: PRNG key.
      num_samples: number of parallel diffusion samples.
      gamma_0, gamma_min, noise_scale_lambda, step_scale_eta: Algorithm 18
        hyperparameters.

    Returns:
      Final atom positions, shape ``[num_samples, N_atom, 3]``.
    """
    n_atom = mask.shape[0]

    init_key, scan_key = jax.random.split(key)

    # Initial fully-noised positions
    positions = noise_schedule[0] * jax.random.normal(
        init_key, (num_samples, n_atom, 3)
    )

    # vmap denoiser over the n_samples axis (z, t_hat shared across samples)
    denoise_v = jax.vmap(denoising_step, in_axes=(0, None))

    def step(carry, x):
        positions, key = carry
        c_tau_last, c_tau = x[0], x[1]

        key, key_aug, key_noise = jax.random.split(key, 3)

        # Per-sample augmentation (different rotation/translation per sample)
        keys_aug = jax.random.split(key_aug, num_samples)
        positions = jax.vmap(
            centre_random_augmentation, in_axes=(0, 0, None)
        )(keys_aug, positions, mask)

        # γ schedule (Protenix gates on c_tau > gamma_min)
        gamma = jnp.where(c_tau > gamma_min, gamma_0, 0.0)
        t_hat = c_tau_last * (1.0 + gamma)

        # Add noise: σ_λ * sqrt(t_hat² - c_tau_last²) · ε
        delta_noise = jnp.sqrt(jnp.maximum(t_hat ** 2 - c_tau_last ** 2, 0.0))
        noise = (noise_scale_lambda * delta_noise) * jax.random.normal(
            key_noise, positions.shape
        )
        x_noisy = positions + noise

        # Denoise (vmap'd over n_samples)
        x_denoised = denoise_v(x_noisy, t_hat)

        # EDM-Heun first-order update
        delta = (x_noisy - x_denoised) / t_hat
        dt = c_tau - t_hat
        positions = x_noisy + step_scale_eta * dt * delta

        return (positions, key), None

    # noise_schedule[i] -> x[0]; noise_schedule[i+1] -> x[1]
    pairs = jnp.stack([noise_schedule[:-1], noise_schedule[1:]], axis=-1)
    (positions_out, _), _ = jax.lax.scan(step, (positions, scan_key), pairs)

    return positions_out


def make_jit_sampler(denoising_step, **sample_kwargs):
    """Convenience: bind denoising_step + kwargs and jit the result.

    Returns a function ``f(noise_schedule, mask, key) -> positions`` ready
    to be called repeatedly with cached compilation.
    """
    def f(noise_schedule, mask, key):
        return sample(
            denoising_step=denoising_step,
            noise_schedule=noise_schedule,
            mask=mask,
            key=key,
            **sample_kwargs,
        )
    return jax.jit(f, static_argnames=())
