# AlphaFold 3 in residudes-engine

AlphaFold 3 is the fourth model (`--model af3`). It reads the same `input.json`
as the others, writes the same output layout, and runs on three backends:

| backend | what runs | where |
|---|---|---|
| `jax` | DeepMind's Haiku model, as published (3.0.2), with the CASP17 fork's unified writer, cross-chain templates and template weight | `core/models/alphafold3` |
| `torch` | a PyTorch port of the whole network that loads the published parameters by their Haiku names | `core/alphafold3/af3torch` |
| `trt` | the port with the trunk pairformer (48 blocks) and the diffusion token transformer (24 blocks) on TensorRT engines; everything else in PyTorch | `core/alphafold3/trt` |

`--backend auto` takes `trt` when engines for the GPU exist and `jax` otherwise.

## Environment and weights

All three backends run in the JAX 0.11 environment `core/jaxenv/.venv`
(`bash core/jaxenv/build.sh`; Python 3.12, JAX 0.11, torch 2.11, TensorRT 10.16,
the alphafold3 package with its C++ extension and CCD tables). The site file
points at it with `venv_jax` and routes the model to it:

```toml
venv_jax = "{residudes_engine_root}/core/jaxenv/.venv"
af3_weights = "/data/af3"            # directory holding af3.bin or af3.bin.zst
[jax]
venv = { pt2 = "venv", of3 = "venv", bt2 = "venv", af3 = "venv_jax" }
```

The parameters are licensed to each user by Google DeepMind and are not
downloaded by residudes-engine: request access through the
[alphafold3 repository](https://github.com/google-deepmind/alphafold3), then
put `af3.bin` in `<checkpoints>/af3/` or point `af3_weights` at it.
`residudes-engine setup checkpoints` reports whether they are present.

## The PyTorch port

`core/alphafold3/af3torch` mirrors the Haiku module tree: evoformer (target
features, relative encoding, bonds, templates, the MSA stack, 48 pairformer
blocks), the diffusion head (conditioning, atom cross-attention encoder and
decoder, the 24-block transformer, the Euler sampler), confidence and distogram
heads. Parameters are addressed by their Haiku scope and name, so the published
file loads without renaming; stacked layers are indexed on their leading axis.
Featurisation, the seed loop and the unified writer are the fork's own code,
reached through a drop-in model runner (`core/alphafold3/inference.py`).

Every block was checked against the Haiku module on the same parameters and
inputs (`core/alphafold3/tests/compare_jax.py`, CPU, float32):

| block | relative error |
|---|---|
| transitions, triangle multiplications, outer product mean, MSA attention | 1e-6 to 1e-7 |
| pair grid attention (both orientations) | 1e-6 |
| evoformer block (MSA stack) | 3e-6 (MSA), 2e-5 (pair) |
| trunk pairformer block, confidence pairformer block, template block | 1e-5 or below |
| diffusion transformer, 24 blocks | 7e-5 |
| atom cross-attention transformer | 1e-6 |

Random numbers (MSA shuffling, diffusion noise and augmentation) come from a
torch generator seeded with the run's seed, so samples are statistically but
not bit-for-bit equivalent to JAX's. The end-to-end check
(`core/alphafold3/tests/compare_jax_e2e.py`) runs DeepMind's model and the port
on the same featurised input in one process: with the MSA fixed to the same
1024 rows the trunk embeddings agree to bf16 rounding (relative error 4e-3 on
the single, 1e-2 on the pair representation after 11 recycles) and the
confidence head gives the same pLDDT and pTM on identical coordinates (within
0.1 and 0.001); with the full MSA, where each recycle draws its own random
1024 rows, the two models' samples fall in the same pLDDT range.

The port runs the trunk and the heads in bf16 with float32 layer norms, the
diffusion transformer in bf16 and the atom encoder and decoder in float32, the
same split as the JAX model. On an RTX PRO 6000 a 1627-token complex takes
433 s on the port and 203 s on JAX with a warm compile cache; the TensorRT
engines bring the port to 264 s (benchmark page for the full matrix).

## TensorRT engines

```sh
residudes-engine engines build --model af3 --max-seqlen 2048          # one GPU, 20 to 60 minutes
```

The recipe (`core/alphafold3/trt/build_engines.sh`) writes the parameters in
OpenFold 3's names, then runs the OpenFold 3 engine converters and builder
unchanged: AlphaFold 3's pairformer and token transformer have the same shapes,
so only the naming differs. Two details are specific to AlphaFold 3 and are set
in the engine configuration: LayerNorm epsilon 1e-5, and the ending-node
triangle attention takes its pair bias from the grid before transposing it
(`tri_attn_end_bias_before_transpose`, an option added to the vendored
`tensorrt_bionemo`). The engine's own torch reference model is checked against
the port with `core/alphafold3/tests/compare_trt_reference.py`.

## Options

| option | effect |
|---|---|
| `--num-recycles`, `--diffusion-samples` | as for the other models (defaults 10 and 5) |
| `--template-weight W` | scales the template embedder's contribution (the fork's `AF3_TPL_WEIGHT`) |
| `RESIDUDES_ENGINE_AF3_TORCH_DTYPE=fp32` | run the trunk in float32 instead of bfloat16 (torch backend) |
| `RESIDUDES_ENGINE_AF3_SAMPLE_BATCH=N` | denoise N diffusion samples at a time (memory) |
| `[jax] af3_attention`, `af3_mem_fraction` | JAX backend: tokamax attention implementation (`triton`), XLA memory fraction (0.95) |

## Limits

- Fold-CP (the fork's context-parallel trunk over 4 GPUs) is not exposed yet.
- The torch and TensorRT backends do not implement the fork's embedding injection and recycle-continuation flags.
