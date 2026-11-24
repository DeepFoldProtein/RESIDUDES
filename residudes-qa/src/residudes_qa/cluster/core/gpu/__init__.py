"""
GPU-accelerated distance matrix computation

Requires: PyTorch with CUDA support

Modules:
    tmscore: GPU TM-score computation
    itm: Interface TM-score for multimers
    drmsd: Distance-based RMSD computation
"""

from residudes_qa.cluster.core.gpu.tmscore import compute_tm_matrix_gpu
from residudes_qa.cluster.core.gpu.itm import compute_itm_matrix_gpu
from residudes_qa.cluster.core.gpu.drmsd import compute_drmsd_matrix_gpu
from residudes_qa.cluster.core.gpu.anchored import compute_anchored_matrix_gpu

__all__ = [
    "compute_tm_matrix_gpu",
    "compute_itm_matrix_gpu",
    "compute_drmsd_matrix_gpu",
    "compute_anchored_matrix_gpu",
]
