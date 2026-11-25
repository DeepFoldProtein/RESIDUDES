"""Monkey-patch Protenix's scipy-based weight init with PyTorch's native impl.

Protenix's `trunc_normal_init_` uses `scipy.stats.truncnorm.rvs`, which runs a
numerical inverse-CDF for every weight tensor. Building the 464M-param v2 model
calls this ~1581 times, taking ~267s per process start.

Since residudes-engine always loads weights from a checkpoint immediately after model
construction, the initial weight values are discarded — we only care about
tensor shapes. Replacing scipy with `torch.nn.init.trunc_normal_` drops model
construction time from ~4.5 min to ~20 s.

Usage: call `patch_protenix_trunc_normal()` AFTER `sys.path` and CWD are set up
for Protenix imports, but BEFORE importing any protenix.model.* module.
"""

from typing import Any


def _fast_trunc_normal_init_(
    weights: Any, scale: float = 1.0, fan: str = "fan_in"
) -> None:
    """No-op stand-in for protenix trunc_normal_init_.

    residudes-engine always loads a checkpoint immediately after model construction, so
    any initial weight values are discarded. This no-op replacement avoids the
    ~90s spent on scipy/torch truncated normal sampling during inference model
    construction.

    For training (which needs actual random init), do NOT apply this patch.
    """
    return None


def patch_protenix_trunc_normal() -> None:
    """Replace Protenix's scipy-based weight init with torch.nn.init version,
    and skip redundant weight initialization since a checkpoint will overwrite
    all values anyway.

    Must be called after Protenix is importable and before constructing any
    Protenix module. Idempotent.
    """
    import torch.nn as nn

    import protenix.model.triangular.layers as _tl
    _tl.trunc_normal_init_ = _fast_trunc_normal_init_
    _tl.lecun_normal_init_ = lambda w: _fast_trunc_normal_init_(w, scale=1.0)
    _tl.he_normal_init_ = lambda w: _fast_trunc_normal_init_(w, scale=2.0)

    # primitives.py imported trunc_normal_init_ by name — patch that binding
    try:
        import protenix.model.modules.primitives as _pr
        _pr.trunc_normal_init_ = _fast_trunc_normal_init_
    except ImportError:
        pass

    # Skip torch's default kaiming_uniform init in nn.Linear.reset_parameters.
    # Protenix wraps nn.Linear and immediately overwrites weights in _init_params,
    # and we'll overwrite again when loading the checkpoint. Skipping this saves
    # another ~90s of tensor.uniform_() calls on 464M-param v2.
    _orig_linear_reset = nn.Linear.reset_parameters
    def _noop_reset(self):
        pass
    nn.Linear.reset_parameters = _noop_reset

    # Also skip nn.LayerNorm reset (ones/zeros init, small but called often)
    _orig_ln_reset = nn.LayerNorm.reset_parameters
    def _noop_ln_reset(self):
        if self.elementwise_affine:
            # Keep shape-correct zero tensors; checkpoint loader will fill them
            pass
    nn.LayerNorm.reset_parameters = _noop_ln_reset
