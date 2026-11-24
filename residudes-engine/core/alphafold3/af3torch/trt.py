# SPDX-License-Identifier: Apache-2.0
"""TensorRT engines for the AlphaFold 3 port: the 48-block trunk pairformer and the
24-block diffusion token transformer, built by core/alphafold3/trt/build_engines.sh
with the tensorrt_bionemo recipe. Everything else (MSA stack, templates, atom
encoders and decoders, heads) stays in PyTorch."""
from __future__ import annotations

import pathlib

import torch


class Engines:
    def __init__(self, engines_dir: pathlib.Path, *, log=print):
        import tensorrt_bionemo  # noqa: F401  (loads the TensorRT plugins)
        from tensorrt_bionemo._trt.module_wrappers import PairformerTRT, TokenTransformerTRT
        from tensorrt_bionemo.runtime import OnDemandContextMemoryManager

        engines_dir = pathlib.Path(engines_dir)
        self.allocator = OnDemandContextMemoryManager()
        pdir = engines_dir / "pairformer_engine"
        if not (pdir / "trt" / "rank0.engine").is_file():
            raise FileNotFoundError(f"no pairformer engine under {pdir}")
        self.pairformer = PairformerTRT.load_weights(checkpoint_dir=pdir, context_memory_allocator=self.allocator)
        tdir = engines_dir / "token_transformer_engine"
        self.token = None
        self.multiplicity = 0
        if (tdir / "trt" / "rank0.engine").is_file():
            self.token = TokenTransformerTRT.load_weights(checkpoint_dir=tdir, context_memory_allocator=self.allocator)
            self.multiplicity = int(getattr(self.token.config, "multiplicity", 1) or 1)
        log(f"[af3 trt] engines from {engines_dir}: pairformer" + (f", token transformer ×{self.multiplicity}" if self.token else ""))

    def trunk(self, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor, pair_mask: torch.Tensor):
        """48 pairformer blocks: (N, 384), (N, N, 128), (N,), (N, N) → (single, pair)."""
        s, z = self.pairformer.forward_udf(single.contiguous()[None], pair.contiguous()[None],
                                           mask.float().contiguous()[None], pair_mask.float().contiguous()[None])
        return s[0], z[0]

    def token_transformer(self, act: torch.Tensor, single_cond: torch.Tensor, pair_cond: torch.Tensor, mask: torch.Tensor):
        """24 diffusion blocks for a batch of samples: act (B, N, 768), single_cond (B, N, 384), pair_cond (N, N, 128), mask (N,).

        The engine takes a, s and mask with a leading sample axis of up to `multiplicity`, and z as (1, N, N, 128)."""
        outs = []
        step = max(self.multiplicity, 1)
        z = pair_cond.contiguous()[None]
        for i in range(0, act.shape[0], step):
            a = act[i:i + step].contiguous()
            s = single_cond[i:i + step].contiguous()
            m = mask.float()[None].expand(a.shape[0], -1).contiguous()
            outs.append(self.token.forward_udf(a, s, z, m))
        return torch.cat(outs, dim=0)
