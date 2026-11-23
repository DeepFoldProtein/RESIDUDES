# Output schema (`lantern-engine/sample@1`)

Every runner writes the same per-seed tree. This is the contract downstream
analysis relies on; it is produced inside the vendored runners (Protenix
`runner/dumper.py`, OF3 `core/runners/writer.py`, Boltz
`scripts/boltz_inference.py`, and the AF3 fork's `post_processing.py`) and
checked by `lantern-engine verify`. The layout was fixed during CASP17 and has not
changed since.

## Layout

```
<seed dir>/                       (= --out of `predict --seed N`, or <out>/seed_<N>/ from `submit`)
├── inference.log                 full stdout/stderr of the run
├── slurm.out, slurm.err          when submitted through Slurm
├── timing.json                   { "runtime_s": 17.3 }  forward pass only (no model/engine load)
├── distogram_logits.npz          int8 pre-softmax trunk distogram (N, N, 64); sample-independent
├── inputs_runtime.json           MSA depth / templates actually used (pt2, of3; af3)
├── .lantern-engine/                     input.json as staged, run.json manifest
└── sample_<i>/                   i = 0 … n_diffusion_samples-1, sample_0 = top ranked
    ├── model.cif                 AF3-canonical 18-column atom_site, B_iso = pLDDT (0–100), _ma_qa_metric rows
    ├── summary.json              per-sample metrics (below)
    ├── meta.json                 writer, model, job_id, seed, sample_idx, native_sample_idx
    └── arrays/
        ├── plddt_atom.npy        float32 (N_atom,)  0–100   [bt2: per-token]
        ├── plddt_token.npy       float32 (N,)       0–100
        ├── pae_logits.npz        int8 pre-softmax (N, N, 64)   [bt2_jax, ef2: pae.npy in Å instead]
        ├── pde_logits.npz        int8 pre-softmax (N, N, 64)   [bt2_jax, ef2: pde.npy in Å instead]
        └── contact_probs.npy     float32 (N, N), P(d < 8 Å)
```

`N` = token count (unique `(chain, resnum)` in the CIF).

## `summary.json`

```jsonc
{
  "model_name": "pt2",          // af3 | pt2 | of3 | bt2 | bt2_jax | ef2
  "seed": 1,
  "sample_idx": 0,              // sample_0 is top ranked (writers pre-sort)
  "n_tokens": 36, "n_atoms": 282,
  "plddt": 89.47,               // mean of plddt_atom, 0–100
  "ptm": 0.76, "iptm": null,    // iptm null for monomers
  "gpde": 0.39,                 // AF3 SI Eq.16, Σ pde·cp / Σ cp, same implementation in all writers
  "has_clash": false,           // absent for bt2
  "ranking_score": 0.15,        // model-native composite, not comparable across models
  "chain_ptm": {"A": 0.76},
  "chain_iptm": null,                        // per-chain mean off-diagonal iPTM for multimers
  "chain_pair_iptm": {"A": {"A": 0.76}},     // symmetric dict-of-dict, diagonal = chain self-pTM
  "fraction_disordered": 0.07,               // af3 only
  "inputs": { "msa": {...}, "templates": {...} }   // provenance snapshot
}
```

Chain keys are letters (`A`, `B`, …) for every model; Boltz's numeric ids are
remapped at write time.

### `ranking_score` is model-specific

| model | formula |
|---|---|
| af3, of3 | `0.8·iptm + 0.2·ptm + 0.5·disorder − 100·has_clash` (monomer: ptm) |
| pt2 | `0.8·iptm + 0.2·ptm − 100·has_clash` |
| bt2 | `(4·plddt + (iptm or ptm)) / 5`, pLDDT dominated |

Use it only to pick the top sample *within* a model. Cross-model ranking
uses the individual metrics.

## Confidence arrays

**pLDDT** is 0–100 everywhere. af3/pt2/of3 are atom-native and derive the
token array by averaging; bt2 is token-native and broadcasts to atoms.

**PAE / PDE** are stored as pre-softmax logits quantised to int8
(`int8_softmax`: subtract the per-position max, clip, linear-quantise; ≈ 1
byte per bin, softmax recovery error ≲ 1e-4). The bin grid is identical for
all models: 64 bins × 0.5 Å on [0, 32] Å, centres 0.25 … 31.75 Å.

```python
from lantern_engine.schema import logits as codec        # vendored logit_codec
L = codec().load_logits(f"{sample}/arrays/pae_logits.npz")["pae"]   # float32 (N, N, 64)
p = codec().softmax(L, axis=-1)
pae_A = (p * codec().bin_centers(n_bins=64, end=32.0)).sum(-1)      # (N, N) Å
```

**Distogram** (seed level) uses model-specific bin grids:

| model | boundaries | centres | step |
|---|---|---|---|
| af3, pt2, of3 | `linspace(2.3125, 21.6875, 63)` | 2.16 → 21.84 Å | 0.3125 Å |
| bt2 | `linspace(2.0, 22.0, 63)` | 1.84 → 22.16 Å | 0.3226 Å |

Bin 0 and bin 63 are open-ended; centres are half-step extrapolated. bt2's
diagonal is unsupervised (peaks near 4.9 Å); ignore self-pairs.

**contact_probs** is the native contact head for af3/pt2/of3; bt2 derives it
from the distogram (19 bins below 8 Å, same count as the others).

## Known per-model differences

| item | detail |
|---|---|
| terminal OXT | af3, pt2 emit it (+1 atom per chain end); of3, bt2 do not |
| `plddt_atom.npy` for bt2 | per-token, not per-atom (Boltz is residue-level) |
| `has_clash`, disorder | not available for bt2 |
| `fraction_disordered` | af3 only |
| `bt2_jax`, `ef2` writers | `pae.npy` / `pde.npy` in Å instead of logits; `model_name` carries the suffix |
| `ef2` | no PDE head (`gpde` null), no templates, AF3-style composite ranking score |

## Versioning

This layout is `lantern-engine/sample@1`. `lantern-engine verify` accepts both array
forms above. A future writer port (Phase 1) will stamp
`meta.json["schema"] = "lantern-engine/sample@1"` and add `"backend"` to
`summary.json` so `model_name` can go back to the plain tag.
