# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Diffusion head and sampler (alphafold3.model.network.diffusion_head) in PyTorch.

The denoiser runs the diffusion samples as a leading batch axis; conditioning
that does not depend on the noise level (the pair conditioning) is computed once
per sampling run. Random numbers come from a torch.Generator seeded with the
run's seed, so samples are statistically but not bit-for-bit equivalent to
JAX's."""
from __future__ import annotations

import math

import torch

from alphafold3.model.network import noise_level_embeddings as af3_noise

from . import nn
from .atom_cross_attention import decoder, encoder
from .diffusion_transformer import transformer, transition_block
from .featurization import create_relative_encoding
from .nn import Scope

SIGMA_DATA = 16.0
PER_TOKEN_CHANNELS = 768


def noise_schedule(t: torch.Tensor, smin: float = 0.0004, smax: float = 160.0, p: int = 7) -> torch.Tensor:
    return SIGMA_DATA * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p


def noise_embeddings(sigma_scaled: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(af3_noise._WEIGHT, dtype=torch.float32, device=sigma_scaled.device)
    b = torch.tensor(af3_noise._BIAS, dtype=torch.float32, device=sigma_scaled.device)
    emb = (0.25 * torch.log(sigma_scaled))[..., None] * w + b
    return torch.cos(2 * math.pi * emb)


def pair_conditioning(batch, embeddings: dict, s: Scope) -> torch.Tensor:
    pair = embeddings["pair"].float()
    rel = create_relative_encoding(batch.token_features, 32, 2).to(pair.dtype)
    features_2d = torch.cat([pair, rel], dim=-1)
    pair_cond = nn.linear(nn.layer_norm(features_2d, s, "pair_cond_initial_norm", offset=False), s, "pair_cond_initial_projection")
    for idx in range(2):
        pair_cond = pair_cond + transition_block(pair_cond, 2, s, f"pair_transition_{idx}")
    return pair_cond


def single_conditioning(embeddings: dict, noise_level: torch.Tensor, s: Scope) -> torch.Tensor:
    """(B, N, 384): trunk single + target features, plus the Fourier noise embedding per sample."""
    single = embeddings["single"].float()
    features_1d = torch.cat([single, embeddings["target_feat"].float()], dim=-1)
    sc = nn.linear(nn.layer_norm(features_1d, s, "single_cond_initial_norm", offset=False), s, "single_cond_initial_projection")
    ne = noise_embeddings(noise_level / SIGMA_DATA)                                            # (B, 256)
    ne = nn.linear(nn.layer_norm(ne, s, "noise_embedding_initial_norm", offset=False), s, "noise_embedding_initial_projection")
    sc = sc[None] + ne[:, None, :]
    for idx in range(2):
        sc = sc + transition_block(sc, 2, s, f"single_transition_{idx}")
    return sc


def denoise(positions_noisy: torch.Tensor, noise_level: torch.Tensor, batch, embeddings: dict, pair_cond: torch.Tensor,
            s: Scope, engines=None) -> torch.Tensor:
    """One denoising step for a batch of samples: positions (B, T, D, 3), noise_level (B,)."""
    single_cond = single_conditioning(embeddings, noise_level, s)
    seq_mask = batch.token_features.mask.float()
    atom_mask = batch.predicted_structure_info.atom_mask.float()
    nl = noise_level[:, None, None, None]
    act = positions_noisy * atom_mask[..., None]
    act = act / torch.sqrt(nl ** 2 + SIGMA_DATA ** 2)

    enc = encoder(act, embeddings["single"], pair_cond, batch, s, "diffusion", per_token_channels=PER_TOKEN_CHANNELS)
    act = enc.token_act.float()
    act = act + nn.linear(nn.layer_norm(single_cond, s, "single_cond_embedding_norm", offset=False), s, "single_cond_embedding_projection")
    if engines is not None and engines.token is not None:
        act = engines.token_transformer(act, single_cond, pair_cond, seq_mask)
    else:
        # bf16 like the TensorRT engine and the JAX model's bfloat16 context; the residual returns to float32 after
        dt = torch.bfloat16 if act.is_cuda else act.dtype
        act = transformer(act.to(dt), seq_mask.to(dt), single_cond.to(dt), pair_cond.to(dt), s.child("transformer")).float()
    act = nn.layer_norm(act, s, "output_norm", offset=False)
    update = decoder(act, enc, batch, s, "diffusion")

    skip = SIGMA_DATA ** 2 / (nl ** 2 + SIGMA_DATA ** 2)
    out = nl * SIGMA_DATA / torch.sqrt(nl ** 2 + SIGMA_DATA ** 2)
    return (skip * positions_noisy + out * update) * atom_mask[..., None]


def _random_rotation(gen: torch.Generator, device) -> torch.Tensor:
    v = torch.randn((2, 3), generator=gen, device=device)
    e0 = v[0] / torch.clamp(v[0].norm(), min=1e-10)
    v1 = v[1] - e0 * torch.dot(v[1], e0)
    e1 = v1 / torch.clamp(v1.norm(), min=1e-10)
    e2 = torch.cross(e0, e1, dim=0)
    return torch.stack([e0, e1, e2])


def random_augmentation(positions: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random rigid motion per sample: positions (B, T, D, 3)."""
    out = []
    for b in range(positions.shape[0]):
        p = positions[b]
        center = nn.mask_mean(mask[..., None], p, dim=(-2, -3), keepdim=True, eps=1e-6)
        rot = _random_rotation(gen, p.device)
        trans = torch.randn((3,), generator=gen, device=p.device)
        out.append((torch.einsum("...i,ij->...j", p - center, rot) + trans) * mask[..., None])
    return torch.stack(out)


def sample(batch, embeddings: dict, s: Scope, *, num_samples: int, steps: int, gen: torch.Generator,
           gamma_0: float = 0.8, gamma_min: float = 1.0, noise_scale: float = 1.003, step_scale: float = 1.5,
           sample_batch: int | None = None, engines=None) -> dict:
    """Euler sampling over the EDM noise schedule. Samples run in groups of `sample_batch`."""
    mask = batch.predicted_structure_info.atom_mask.float()
    device = mask.device
    levels = noise_schedule(torch.linspace(0, 1, steps + 1, device=device))
    pair_cond = pair_conditioning(batch, embeddings, s)
    positions = torch.randn((num_samples,) + tuple(mask.shape) + (3,), generator=gen, device=device) * levels[0]
    prev = levels[0].expand(num_samples).clone()
    groups = [slice(i, min(i + (sample_batch or num_samples), num_samples)) for i in range(0, num_samples, sample_batch or num_samples)]
    for lvl in levels[1:]:
        positions = random_augmentation(positions, mask, gen)
        gamma = gamma_0 if float(lvl) > gamma_min else 0.0
        t_hat = prev * (1 + gamma)
        ns = noise_scale * torch.sqrt(t_hat ** 2 - prev ** 2)
        noise = ns[:, None, None, None] * torch.randn(positions.shape, generator=gen, device=device)
        noisy = positions + noise
        denoised = torch.cat([denoise(noisy[g], t_hat[g], batch, embeddings, pair_cond, s, engines=engines) for g in groups])
        grad = (noisy - denoised) / t_hat[:, None, None, None]
        d_t = lvl - t_hat
        positions = noisy + step_scale * d_t[:, None, None, None] * grad
        prev = lvl.expand(num_samples).clone()
    return {"atom_positions": positions, "mask": mask[None].expand(num_samples, *mask.shape).clone()}
