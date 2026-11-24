# residudes-engine runtime (`core/`)

The inference runtime behind the `residudes-engine` CLI: model code for Protenix v2,
OpenFold 3 and Boltz-2 (`models/`, `protenix/`, `openfold3/`, `boltz/`), the
TensorRT and JAX acceleration backends (`*/trt/`, `residudes_engine_jax/`), the
AF3-dialect input adapter and unified output helpers (`scripts/`), and
third-party TensorRT libraries (`thirdparty/`, see `thirdparty/LICENSES.md`).

Users do not call anything here directly. `residudes-engine setup venv` builds the
runtime environment at `core/.venv` from `pyproject.toml` + `uv.lock`, and
`residudes-engine predict` runs the entry points with the right environment. Internal
module and environment-variable names in this tree are historical and are
mapped by the CLI; see `../docs/architecture.md`.
