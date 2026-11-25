# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: PT DiffusionModule vs JAX DiffusionModule.

Approach:
  1. Run a Protenix v2 inference up to the point of calling
     DiffusionModule.f_forward.
  2. At first call, capture all input tensors and the PT output, then
     short-circuit further calls (we only need 1 sample).
  3. Reproduce on the JAX side using the converted weights and compare.

Run with::

    python protenix/test_diffusion_module_parity.py \\
        --input_json_path examples/protenix/ubiquitin_protenix.json \\
        --load_checkpoint_dir /home/deepfold/checkpoint \\
        --model_name protenix-v2 \\
        --use_msa false --enable_efficient_fusion false
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("dm-parity")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROTENIX_ROOT = os.path.join(_CORE_ROOT, "models/protenix")

sys.path.insert(0, _CORE_ROOT)
sys.path.insert(0, _PROTENIX_ROOT)
os.chdir(_PROTENIX_ROOT)

# Capture path
CAPTURE_PATH = "/tmp/dm_inputs.pkl"

# Pre-parse our flags
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--capture_only", action="store_true")
_pre.add_argument("--no_capture", action="store_true")
_args, _remaining = _pre.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Fast init
import importlib.util
_fi_spec = importlib.util.spec_from_file_location(
    "residudes_fast_init", os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fi_mod = importlib.util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_fi_mod)
_fi_mod.patch_protenix_trunc_normal()

import torch
import torch.nn.functional as F


# ── Capture inputs to PT DiffusionModule.f_forward ──────────────────────
import protenix.model.modules.diffusion as _pt_diff

_orig_f_forward = _pt_diff.DiffusionModule.f_forward
_captured: dict | None = None
_capture_done = False


def _capture_f_forward(
    self,
    r_noisy, t_hat_noise_level, input_feature_dict,
    s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
    inplace_safe=False, chunk_size=None, use_conditioning=True,
    enable_efficient_fusion=False,
):
    global _captured, _capture_done
    if not _capture_done and not _args.no_capture:
        logger.info("Capturing DiffusionModule inputs...")
        def _cpu(t):
            return t.detach().cpu().clone() if torch.is_tensor(t) else t
        # Save everything as CPU tensors / dicts
        capture = {
            "r_noisy": _cpu(r_noisy),
            "t_hat_noise_level": _cpu(t_hat_noise_level),
            "s_inputs": _cpu(s_inputs),
            "s_trunk": _cpu(s_trunk),
            "z_trunk": _cpu(z_trunk),
            "pair_z": _cpu(pair_z),
            "p_lm": _cpu(p_lm),
            "c_l": _cpu(c_l),
        }
        # Capture relevant input_feature_dict tensors (small set)
        feat_keys = ["atom_to_token_idx", "ref_pos", "ref_charge", "ref_mask",
                     "ref_atom_name_chars", "ref_element", "d_lm", "v_lm",
                     "pad_info", "relp"]
        capture["input_feature_dict"] = {}
        for k in feat_keys:
            if k in input_feature_dict:
                v = input_feature_dict[k]
                if isinstance(v, dict):
                    capture["input_feature_dict"][k] = {kk: _cpu(vv) for kk, vv in v.items()}
                else:
                    capture["input_feature_dict"][k] = _cpu(v)
        # Run PT forward and capture output too
        out = _orig_f_forward(
            self, r_noisy, t_hat_noise_level, input_feature_dict,
            s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
        capture["pt_output"] = _cpu(out)
        capture["enable_efficient_fusion"] = enable_efficient_fusion
        capture["use_conditioning"] = use_conditioning
        with open(CAPTURE_PATH, "wb") as f:
            pickle.dump(capture, f)
        _captured = capture
        _capture_done = True
        logger.info(f"Saved capture to {CAPTURE_PATH}")
        if _args.capture_only:
            sys.exit(0)
        return out
    return _orig_f_forward(
        self, r_noisy, t_hat_noise_level, input_feature_dict,
        s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
        inplace_safe=inplace_safe, chunk_size=chunk_size,
        use_conditioning=use_conditioning,
        enable_efficient_fusion=enable_efficient_fusion,
    )


_pt_diff.DiffusionModule.f_forward = _capture_f_forward


# ── Run protenix inference (forces capture on first DM call) ────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    from runner.inference import run
    run()
