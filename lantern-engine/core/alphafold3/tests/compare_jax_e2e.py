#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end comparison of the PyTorch port against DeepMind's JAX model on one
input: the fork featurises once, both models run on the same example and seed,
and the script reports where the two diverge (trunk embeddings, the confidence
head on identical coordinates, per-sample pLDDT / pTM). Needs a GPU and the JAX
environment (core/jaxenv/.venv).

    python compare_jax_e2e.py --json_path input.json --model_dir /path/with/af3.bin [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
AF3 = HERE.parent
FORK = AF3.parent / "models" / "alphafold3"
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(AF3))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.5")
os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

import jax  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_alphafold as ra  # noqa: E402
from alphafold3.common import folding_input  # noqa: E402
from alphafold3.constants import chemical_components  # noqa: E402
from alphafold3.data import featurisation  # noqa: E402
from alphafold3.model import confidences  # noqa: E402
from alphafold3.model.components import utils  # noqa: E402

from af3torch import model as torch_model  # noqa: E402
from af3torch.evoformer import create_target_feat_embedding  # noqa: E402
from af3torch.featurization import batch_to_device  # noqa: E402
from af3torch.heads import confidence_head  # noqa: E402
from af3torch.runner import TorchModelRunner  # noqa: E402

BUCKETS = (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)


def rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))


def masked_mean(x, mask) -> float:
    x = np.asarray(x, dtype=np.float64); mask = np.asarray(mask, dtype=bool)
    return float(x[mask].mean()) if mask.any() else float("nan")


def ptm(tm_adjusted_pae, asym_id, mask) -> float:
    """pTM of one sample from its TM-adjusted PAE, as the fork's post-processing computes it (pair mask = token mask)."""
    pm = np.outer(mask, mask).astype(bool)
    return float(confidences.predicted_tm_score(tm_adjusted_pae=np.asarray(tm_adjusted_pae), asym_id=np.asarray(asym_id),
                                                pair_mask=pm, interface=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", default=None, help="directory for the two result dicts (npz)")
    ap.add_argument("--msa-rows", type=int, default=None,
                    help="keep only the first K MSA rows that have an unmasked position; with K <= 1024 both models "
                         "see the same rows every recycle, so the trunk becomes deterministic")
    args = ap.parse_args()
    t0 = time.time()
    fold_input = next(iter(folding_input.load_fold_inputs_from_path(pathlib.Path(args.json_path))))
    seed = int(fold_input.rng_seeds[0])
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    example = featurisation.featurise_input(fold_input=fold_input, buckets=BUCKETS, ccd=ccd, verbose=False)[0]
    example = utils.remove_invalidly_typed_feats(example)
    valid_rows = int(np.asarray(example["msa_mask"]).any(-1).sum())
    if args.msa_rows:
        keep = np.flatnonzero(np.asarray(example["msa_mask"]).any(-1))[: args.msa_rows]
        m = np.zeros(np.asarray(example["msa_mask"]).shape[0], dtype=bool); m[keep] = True
        example["msa_mask"] = np.asarray(example["msa_mask"]) * m[:, None]
        example["msa"] = np.where(m[:, None], np.asarray(example["msa"]), 0)
        example["deletion_matrix"] = np.asarray(example["deletion_matrix"]) * m[:, None]
    print(f"featurised {fold_input.name}: seed {seed}, msa rows {np.asarray(example['msa']).shape} ({valid_rows} with data"
          + (f", kept {len(keep)}" if args.msa_rows else "") + f"), templates {np.asarray(example['template_aatype']).shape[0]} "
          f"({time.time() - t0:.0f} s)", flush=True)
    rng_key = jax.random.PRNGKey(seed)
    model_dir = pathlib.Path(args.model_dir)

    # ── JAX (DeepMind's model) ──────────────────────────────────────────────
    jr = ra.ModelRunner(ra.make_model_config(return_embeddings=True), jax.local_devices(backend="gpu")[0], model_dir)
    t1 = time.time()
    rj = jr.run_inference(example, rng_key)
    print(f"jax done in {time.time() - t1:.1f} s; keys: {sorted(k for k in rj if not k.startswith('__'))}", flush=True)

    # ── PyTorch port (bf16 trunk, then fp32 trunk) ──────────────────────────
    results = {}
    for name, dtype in (("torch-bf16", torch.bfloat16), ("torch-fp32", torch.float32)):
        tr = TorchModelRunner(ra.make_model_config(return_embeddings=True), None, model_dir)
        tr.cfg.trunk_dtype = dtype
        t1 = time.time()
        results[name] = tr.run_inference(example, rng_key)
        print(f"{name} done in {time.time() - t1:.1f} s", flush=True)
    params = tr.model_params

    b = batch_to_device(example, params.device)
    tok_mask = np.asarray(b.token_features.mask.cpu().numpy(), dtype=bool)
    n = int(tok_mask.sum())
    asym = np.asarray(b.token_features.asym_id.cpu().numpy())[:n]
    atom_mask = np.asarray(b.predicted_structure_info.atom_mask.cpu().numpy(), dtype=bool)      # (N_pad, 24)
    print(f"\ntokens {n} (padded {atom_mask.shape[0]})")

    def ptm_of(r, k):
        return ptm(np.asarray(r["tmscore_adjusted_pae_global"][k])[:n, :n], asym, tok_mask[:n])

    # 1. trunk embeddings
    print("\n== trunk embeddings vs jax (relative error on the real tokens) ==")
    for name, rt in results.items():
        print(f"  {name:11s} single {rel(rj['single_embeddings'][:n], rt['single_embeddings'][:n]):.2e}   "
              f"pair {rel(rj['pair_embeddings'][:n, :n], rt['pair_embeddings'][:n, :n]):.2e}")

    # 2. per-sample confidence from each model's own samples
    print("\n== each model's own samples: masked mean pLDDT per sample, pTM ==")
    for name, r in [("jax", rj)] + list(results.items()):
        pl = [masked_mean(r["predicted_lddt"][k][:atom_mask.shape[0]], atom_mask) for k in range(r["predicted_lddt"].shape[0])]
        pt = [ptm_of(r, k) for k in range(r["predicted_lddt"].shape[0])]
        print(f"  {name:11s} pLDDT {' '.join(f'{v:5.1f}' for v in pl)}   pTM {' '.join(f'{v:.3f}' for v in pt)}")

    # 3. the port's confidence head on the JAX samples' coordinates, with JAX's and with the port's embeddings
    print("\n== port confidence head on the JAX sample coordinates ==")
    root = params.root().child("diffuser")
    dev = params.device
    with torch.inference_mode():
        for emb_name, emb_src in (("jax embeddings", rj), ("torch-bf16 embeddings", results["torch-bf16"]),
                                  ("torch-fp32 embeddings", results["torch-fp32"])):
            for dt_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                target_feat = create_target_feat_embedding(b, root, dtype)
                emb = {"pair": torch.as_tensor(np.asarray(emb_src["pair_embeddings"]), device=dev),
                       "single": torch.as_tensor(np.asarray(emb_src["single_embeddings"]), device=dev),
                       "target_feat": target_feat}
                rows = []
                for k in range(rj["predicted_lddt"].shape[0]):
                    pos = torch.as_tensor(np.asarray(rj["diffusion_samples"]["atom_positions"][k]), device=dev, dtype=torch.float32)
                    c = confidence_head(pos, emb, b.token_features.mask, b.pseudo_beta_info.token_atoms_to_pseudo_beta,
                                        b.token_features.asym_id, root.child("confidence_head"), dtype=dtype)
                    pl = masked_mean(c["predicted_lddt"].float().cpu().numpy(), atom_mask)
                    pl_j = masked_mean(rj["predicted_lddt"][k], atom_mask)
                    pt = ptm(c["tmscore_adjusted_pae_global"].float().cpu().numpy()[:n, :n], asym, tok_mask[:n])
                    rows.append((pl, pl_j, pt, ptm_of(rj, k)))
                print(f"  {emb_name:22s} head {dt_name}: " + "  ".join(f"[{pl:5.1f} vs jax {plj:5.1f}; pTM {pt:.3f} vs {ptj:.3f}]" for pl, plj, pt, ptj in rows))

    if args.out:
        out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "jax.npz", single=rj["single_embeddings"][:n], pair=rj["pair_embeddings"][:n, :n],
                            plddt=rj["predicted_lddt"], pos=rj["diffusion_samples"]["atom_positions"])
        for name, rt in results.items():
            np.savez_compressed(out / f"{name}.npz", single=rt["single_embeddings"][:n], pair=rt["pair_embeddings"][:n, :n],
                                plddt=rt["predicted_lddt"], pos=rt["diffusion_samples"]["atom_positions"])
        print(f"\nsaved to {out}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
