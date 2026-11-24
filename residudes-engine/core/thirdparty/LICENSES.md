# Third-party code under `core/`

| path | origin | license |
|---|---|---|
| `thirdparty/tensorrt_bionemo/` | NVIDIA BioNeMo NIM TensorRT engine pipeline, with local patches (FP32 triangle-bias cast for the cuEquivariance plugin, Protenix/Boltz-2 model configs, prebuilt `IdentitySZ` plugin `.so`) | Apache-2.0 (SPDX headers in each file) |
| `thirdparty/tensorrt_llm_lite/` | subset of NVIDIA TensorRT-LLM's network builder API | Apache-2.0 |
| `thirdparty/ipsae/` | DunbrackLab IPSAE (`ipsae_original.py` verbatim, `ipsae.py` refactored into a callable API) | MIT (see file header) |
| `models/protenix/` | ByteDance Protenix, with local patches (unified output writer, template weight, MSA/template featurizer fixes, metadiffusion) | Apache-2.0 — `models/protenix/LICENSE` |
| `models/openfold-3/` | AQ Laboratory OpenFold 3, with local patches (unified output writer, template embedder weight, MSA I/O) | Apache-2.0 — `models/openfold-3/LICENSE` |
| Boltz-2 | installed from PyPI (`boltz`), patched at runtime by `scripts/boltz_inference*.py` | MIT |

The Apache-2.0 text is included at `models/protenix/LICENSE`. Model weights
are downloaded from their publishers by `residudes-engine setup checkpoints` and are
subject to each publisher's terms.
