try:
    from .cuequiv import CuEquivAttention, CuEquivAttentionMetadata
    _CUEQUIV_ATTN_AVAILABLE = True
except (ImportError, OSError):
    CuEquivAttention = None
    CuEquivAttentionMetadata = None
    _CUEQUIV_ATTN_AVAILABLE = False

from .interface import AttentionBackend, AttentionMetadata, AttentionType
from .trifast import TrifastAttention, TrifastAttentionMetadata
from .utils import create_attention, get_attention_backend
from .vanilla import (VanillaAttentionMetadata, VanillaPairwiseAttention,
                      VanillaTriangleAttention)

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "VanillaTriangleAttention",
    "VanillaPairwiseAttention",
    "VanillaAttentionMetadata",
    "CuEquivAttention",
    "CuEquivAttentionMetadata",
    "TrifastAttention",
    "TrifastAttentionMetadata",
    "AttentionType",
    "get_attention_backend",
    "create_attention",
]
