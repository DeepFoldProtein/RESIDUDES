# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
TRT Python plugin wrapping the trifast Triton kernel for triangle attention.

Registers two plugins under the 'trifast' namespace:
  - trifast::triangle_attention(q, k, v, bias, mask, scale, num_heads)
  - trifast::identity_2(s, z)

These use the add_plugin_v3() / trtp.register() API and require NO pre-built .so.
"""
from typing import Tuple

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import triton


def _trt_to_torch_dtype(trt_dtype):
    _map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    return _map[trt_dtype]


def _to_torch(*args):
    ret = []
    for a in args:
        if a is None:
            ret.append(None)
        elif isinstance(a, trtp.Tensor):
            if a.dtype == trt.bfloat16:
                a._immutable = False
                a._dtype = trt.float16
                tt = torch.as_tensor(a, device="cuda").view(dtype=torch.bfloat16)
            else:
                tt = torch.as_tensor(a, device="cuda")
            ret.append(tt)
        else:
            raise ValueError(f"Unexpected type: {type(a)}")
    return ret


# ── identity_2 ──────────────────────────────────────────────────────────────

@trtp.register("trifast::identity_2")
def _(s: trtp.TensorDesc, z: trtp.TensorDesc) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    return s.like(), z.like()


@trtp.impl("trifast::identity_2")
def _(s: trtp.Tensor, z: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int):
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        s_, z_, s_out, z_out = _to_torch(s, z, outputs[0], outputs[1])
        s_out.copy_(s_)
        z_out.copy_(z_)


# ── triangle_attention ───────────────────────────────────────────────────────

@trtp.register("trifast::triangle_attention")
def _(
    q: trtp.TensorDesc,
    k: trtp.TensorDesc,
    v: trtp.TensorDesc,
    bias: trtp.TensorDesc,
    mask: trtp.TensorDesc,
    scale: float,
    num_heads: int,
) -> trtp.TensorDesc:
    return q.like()


@trtp.impl("trifast::triangle_attention")
def _(
    q: trtp.Tensor,
    k: trtp.Tensor,
    v: trtp.Tensor,
    bias: trtp.Tensor,
    mask: trtp.Tensor,
    scale: float,
    num_heads: int,
    outputs: Tuple[trtp.Tensor],
    stream: int,
):
    import trifast as _tf
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        q_, k_, v_, bias_, mask_, out_ = _to_torch(q, k, v, bias, mask, outputs[0])
        # Bio-TRT layout: q,k,v (B, I, H, J, D); bias (B, 1, H, J, J); mask (B, I, 1, 1, J).
        # PyPI trifast expects: (b, h, n, n, d), (b, h, n, n), (b, n, n) with N = I = J.
        B, I_, H_, J_, D_ = q_.shape
        q_t = q_.permute(0, 2, 1, 3, 4).contiguous()
        k_t = k_.permute(0, 2, 1, 3, 4).contiguous()
        v_t = v_.permute(0, 2, 1, 3, 4).contiguous()
        b_t = bias_.squeeze(1).contiguous()
        # Bio-TRT mask True=attend → trifast True=mask_out (invert).
        m_t = (~mask_.squeeze(2).squeeze(2).bool()).contiguous()
        # trifast applies scale = d**-0.5 internally; compensate if user scale differs.
        internal_scale = D_ ** -0.5
        if abs(float(scale) - internal_scale) > 1e-5:
            q_t = q_t * (float(scale) / internal_scale)
        o = _tf.triangle_attention(q_t, k_t, v_t, b_t, m_t)
        # trifast out: (B, H, N, N, D) → Bio-TRT (B, I, H, J, D)
        out_.copy_(o.permute(0, 2, 1, 3, 4).contiguous().to(out_.dtype))


def _trifast_triangle_attention(q, k, v, bias, mask, scale, num_heads, out, stream, cast_dtype=None):
    """Run trifast Triton kernel on tensors in Bio-TRT layout.

    Input layout (from Bio-TRT attention.py CUEQUIV/TRIFAST path):
        q, k, v:  [B, I, H, J, D]
        bias:     [B, 1, H, J, J]
        mask:     [B, I, 1, 1, J]  (bool)

    The trifast kernel expects:
        q, k, v:  [BH, I, J, D]
        bias:     [BH, J, J]
        mask:     [B, I, J]  (bool)
    """
    from tensorrt_bionemo.triton_kernels.trifast import trifast_attention_kernel_fwd

    B, I, H, J, D = q.shape

    # Reshape: [B, I, H, J, D] -> [B*H, I, J, D]
    q = q.permute(0, 2, 1, 3, 4).reshape(B * H, I, J, D).contiguous()
    k = k.permute(0, 2, 1, 3, 4).reshape(B * H, I, J, D).contiguous()
    v = v.permute(0, 2, 1, 3, 4).reshape(B * H, I, J, D).contiguous()

    # bias: [B, 1, H, J, J] -> [B*H, J, J]
    bias = bias.squeeze(1).reshape(B * H, J, J).contiguous()

    # mask: [B, I, 1, 1, J] -> [B, I, J] (bool)
    # Bio-TRT passes mask with True=attend (cuequivariance convention),
    # but trifast kernel uses True=mask_out, so invert.
    mask = (~mask.squeeze(2).squeeze(2).bool()).contiguous()

    # Allocate output and lse
    o = torch.zeros(B * H, I, J, D, device=q.device, dtype=q.dtype)
    lse = torch.zeros(B * H, I, J, device=q.device, dtype=torch.float32)

    neg_inf = torch.finfo(q.dtype).min

    # Find closest power of 2 >= J for Triton
    CLOSEST_N = triton.next_power_of_2(J) if J > 0 else 16

    # Fixed config (no autotuning in TRT — deterministic)
    BLOCK_J = min(64, CLOSEST_N)
    BLOCK_K = 32

    def grid(meta):
        return (triton.cdiv(J, meta["BLOCK_J"]), I, B * H)

    trifast_attention_kernel_fwd[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse, lse.stride(0), lse.stride(1), lse.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bias, bias.stride(0), bias.stride(1), bias.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        sm_scale=scale,
        neg_inf=neg_inf,
        batch_size=B,
        si=I,
        seq_len=J,
        heads=H,
        DIM=D,
        BLOCK_J=BLOCK_J,
        BLOCK_K=BLOCK_K,
        CLOSEST_N=CLOSEST_N,
    )

    # Reshape back: [B*H, I, J, D] -> [B, H, I, J, D] -> [B, I, H, J, D]
    o = o.reshape(B, H, I, J, D).permute(0, 2, 1, 3, 4).contiguous()
    if cast_dtype is not None and cast_dtype != o.dtype:
        o = o.to(cast_dtype)
    out.copy_(o)


def register_trifast_plugins():
    """No-op — plugins are registered at import time via @trtp.register decorators."""
    pass
