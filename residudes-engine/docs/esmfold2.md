# ESMFold2 in residudes-engine

ESMFold2 (`ef2`, EvolutionaryScale / Biohub, on the ESMC-6B language model) is the fifth
predictor and the one that behaves differently from the other four:

| | the other four | ESMFold2 |
|---|---|---|
| backends | PyTorch, TensorRT, JAX | **PyTorch only** — no engine to build, no JAX port |
| templates | taken, and weighted | **none**: no template embedder |
| MSA | paired and unpaired | unpaired only, or no MSA at all |
| environment | `core/.venv` (or `core/jaxenv/.venv`) | its own python 3.12 conda env in the esm checkout |
| weights | a file under `<checkpoints>` | a HuggingFace cache (`biohub/ESMFold2`, multi-GB) |

It produced 36,975 of the 726,097 structures in
[residudes-decoy](../../docs/residudes-decoy.md), almost all of them single-sequence folds.

## Environment

The runner python lives **with the esm checkout**, because it imports `esm.*`.
residudes-engine supplies the input and reads the output; it does not vendor the model.

The checkout is EvolutionaryScale's [`esm`](https://github.com/evolutionaryscale/esm) with the
CASP17 runner scripts under `inferences/esmfold2/` (`build_input.py`, `run_inference.py`,
`write_unified.py`) and `setup_env.sh`, which builds the conda env and fetches the weights:

```sh
esm/inferences/esmfold2/setup_env.sh      # conda env at esm/.conda + the HF weights
```

Then point the site at it:

```toml
[site]
esm = "/data/esm"                           # the checkout; its .conda/bin/python runs ef2
# esm_python = "/data/esm/.conda/bin/python"  # if the interpreter is elsewhere
# esmfold2_weights = "/data/esmfold2_hf"      # HF cache, if not <checkpoints>/esmfold2_hf
```

`[site] esm` unset, or no `run_inference.py` under it, is a plain error that says this.

## Running it

```sh
residudes-engine predict --model ef2 --input inputs/A3/single_seq/input.json --out out/ef2/seed_1 --seed 1
```

`--backend` is `torch` (the only one; `auto` resolves to it). `--diffusion-samples` defaults to
5, as for the other models, and `--num-recycles` maps to ESMFold2's "loops", whose own default
of 3 stands unless you pass it. One seed per run: `<out>` is itself the seed directory and the
samples land in it, the bt2 and af3 convention.

## Template-free inputs

ESMFold2 has no template embedder, so an input carrying `templates` (per chain) or
`multimerTemplates` is refused **while planning**, before a GPU is taken:

```
ESMFold2 has no template embedder, but …/af3_msa/input.json carries templates for chain(s) A, B, C.
Build a template-free input variant (single_seq, or <variant>_msa_only, which keeps the unpaired MSA).
```

The CASP17 pipeline built two such variants per target, and the runner folds either without a flag —
whether an MSA is used follows from what the input carries:

| variant | what it is |
|---|---|
| `single_seq` | the sequence alone |
| `<variant>_msa_only` | the unpaired MSA of that variant, templates dropped |

## Limits

- No templates, so the recipes that carry CASP17's biggest gains (a multimer template from
  MODELLER, a pinned homolog) do not apply to it.
- No TensorRT or JAX path, so it is not in the backend benchmark
  ([`benchmark.md`](benchmark.md) covers the four models that have all three).
- The ESMC-6B backbone is large: a high-memory card is preferred, though small targets fit a 24 GB
  RTX 4090.
