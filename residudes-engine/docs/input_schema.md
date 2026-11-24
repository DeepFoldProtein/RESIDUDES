# Input

residudes-engine accepts two kinds of input and passes both to the model entry
points unchanged apart from seed injection.

## 1. AF3-dialect `input.json` (recommended: one file drives every model)

The AlphaFold 3 input format with MSAs and templates already attached,
converted at runtime to each model's native input by
`core/scripts/af3_input_adapter.py`. residudes-engine recognises it by
`"dialect": "alphafold3"`.

```jsonc
{
  "name": "T1430_A1",
  "dialect": "alphafold3",
  "version": 1,
  "modelSeeds": [1],
  "sequences": [
    {
      "protein": {
        "id": "A",                                   // or ["A", "B"] for a homomer
        "sequence": "FNRVKIGENVEVGPHVVVKNHVRIGRNVKVGPHVVV",
        "unpairedMsaPath": "/abs/path/unpaired_A.a3m",
        "pairedMsaPath":   "/abs/path/paired_A.a3m",
        "templates": [
          {
            "mmcifPath": "/abs/path/7xyz_B.cif",     // single-chain mmCIF
            "queryIndices":    [2, 3, 4, ...],
            "templateIndices": [115, 116, 117, ...]
          }
        ]
      }
    }
    // "rna": {...}, "dna": {...}, "ligand": {...} are supported by the adapter
  ]
}
```

What happens per model (all inside the adapter):

| model | native input produced | notes |
|---|---|---|
| pt2 | Protenix list-of-jobs JSON | templates re-expressed as hmmsearch-style a3m from the cif + indices; `modelSeeds` → job seeds |
| of3 | OF3 query JSON (`queries`, `seeds`) | MSAs staged as `colabfold_main.a3m` / `colabfold_paired.a3m` per target (OF3 filters MSA files by stem); needs a pairing mode (`--pairing-mode`, `$OF3_PAIRING_MODE`, or a `pairing_mode.txt` sidecar; default `precomputed`) |
| bt2 | Boltz YAML | templates by cif path; contacts/constraints/glycan bonds carried over; seed via `--seed` |

Paths inside the JSON should be absolute. residudes-engine copies the file to
`<out>/.residudes-engine/input.json`, and when `--seed N` is given rewrites
`modelSeeds` to `[N]` (and `seeds` for OF3).

Where these files come from: `residudes-engine prepare` (or `msa` + `inputs`) builds
them from a FASTA: MSAs from the DeepFold-PLM MSA API, the ColabFold/MMseqs2
API or a local MMseqs2, templates from the search hits or pinned / self /
multi-chain structures, plus ligands, glycans and contact constraints. See
[`inputs.md`](inputs.md). Any other pipeline that emits a3m files and
single-chain template mmCIFs works too.

## 2. Native inputs (pass-through)

Anything that is not AF3 dialect is handed to the model as-is:

| model | accepted | example |
|---|---|---|
| pt2 | Protenix JSON (dict or list of jobs) | `core/examples/protenix/ubiquitin_protenix.json` |
| of3 | OF3 query JSON | `core/examples/openfold3/ubiquitin.json` |
| bt2 | Boltz YAML | `core/examples/boltz2/ubiquitin.yaml` |

Native inputs are useful for MSA-free smoke tests and for anything the AF3
dialect cannot express. `--seed` applies to bt2 (CLI flag); for pt2/of3
native inputs put the seed in the file.

## Seeds and output placement

| invocation | staged seeds | output |
|---|---|---|
| `predict --seed 3` | `[3]` | `engine_out/<name>_<model>/seed_3/` (or `--out DIR`) |
| `predict` (no `--seed`) | as in the file | one seed → `<out>/`; several → `<out>/seed_<N>/` |
| `submit --seeds 1,5-7` | one job per seed | `<out>/seed_1/`, `<out>/seed_5/`, … |

## Size limits

TensorRT engines have a build-time `max_seq_len` (tokens): 1024 on
RTX 4090 and in the default `engines build`, 3072–5120 on RTX PRO 6000 and
B200 (see [`sites.md`](sites.md)). Above it the TensorRT path errors out;
use `--backend jax` or `torch`, or build a larger engine.
