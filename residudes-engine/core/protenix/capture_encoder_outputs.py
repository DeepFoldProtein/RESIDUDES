# SPDX-License-Identifier: Apache-2.0
"""Capture encoder + decoder outputs from PT inference."""
import argparse, logging, os, pickle, sys
logger = logging.getLogger("cap-enc")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROTENIX_ROOT = os.path.join(_CORE_ROOT, "models/protenix")
sys.path.insert(0, _CORE_ROOT); sys.path.insert(0, _PROTENIX_ROOT)
os.chdir(_PROTENIX_ROOT)

import importlib.util
_fi = importlib.util.spec_from_file_location("residudes_fast_init", os.path.join(_SCRIPT_DIR, "_fast_init.py"))
_fim = importlib.util.module_from_spec(_fi); _fi.loader.exec_module(_fim); _fim.patch_protenix_trunc_normal()

import torch

import protenix.model.modules.transformer as _t
_orig_enc = _t.AtomAttentionEncoder.forward
_orig_dec = _t.AtomAttentionDecoder.forward
_done = False
CAP_PATH = "/tmp/encdec_outputs.pkl"

def _enc_hook(self, *args, **kwargs):
    global _done
    out = _orig_enc(self, *args, **kwargs)
    if not _done:
        a_token, q_skip, c_skip, p_skip = out
        # Also save all the inputs
        names = ['atom_to_token_idx', 'ref_pos', 'ref_charge', 'ref_mask', 'ref_atom_name_chars',
                 'ref_element', 'd_lm', 'v_lm', 'pad_info']
        data = {'enc_out_a_token': a_token.detach().cpu().clone(),
                'enc_out_q_skip': q_skip.detach().cpu().clone(),
                'enc_out_c_skip': c_skip.detach().cpu().clone(),
                'enc_out_p_skip': p_skip.detach().cpu().clone()}
        for i, name in enumerate(names):
            v = args[i] if i < len(args) else kwargs.get(name)
            if torch.is_tensor(v): data[f'enc_in_{name}'] = v.detach().cpu().clone()
            elif isinstance(v, dict):
                data[f'enc_in_{name}'] = {kk: vv.detach().cpu().clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        for kw_name in ['r_l', 's', 'z', 'p_lm', 'c_l']:
            if kw_name in kwargs:
                v = kwargs[kw_name]
                if torch.is_tensor(v): data[f'enc_in_{kw_name}'] = v.detach().cpu().clone()
                elif v is None: data[f'enc_in_{kw_name}'] = None
        # save partial; will overwrite with dec data later
        with open(CAP_PATH, "wb") as f: pickle.dump(data, f)
        logger.info(f"Saved encoder outputs to {CAP_PATH}")
    return out

def _dec_hook(self, *args, **kwargs):
    global _done
    out = _orig_dec(self, *args, **kwargs)
    if not _done:
        with open(CAP_PATH, "rb") as f: data = pickle.load(f)
        data['dec_out_r'] = out.detach().cpu().clone()
        data['dec_in_atom_to_token_idx'] = (args[0] if args else kwargs.get('atom_to_token_idx')).detach().cpu().clone()
        for i, n in enumerate(['a', 'q_skip', 'c_skip', 'p_skip']):
            v = args[i+1] if (i+1) < len(args) else kwargs.get(n)
            data[f'dec_in_{n}'] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        with open(CAP_PATH, "wb") as f: pickle.dump(data, f)
        logger.info(f"Saved decoder outputs (and inputs)")
        _done = True
        sys.exit(0)
    return out

_t.AtomAttentionEncoder.forward = _enc_hook
_t.AtomAttentionDecoder.forward = _dec_hook

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    from runner.inference import run
    run()
