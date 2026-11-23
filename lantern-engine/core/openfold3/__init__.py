# Bio-TRT OpenFold3 adapter — this package lives alongside the upstream
# openfold3 source at models/openfold-3/openfold3/. To let both coexist
# under the same `openfold3.*` namespace we extend __path__ to include the
# upstream tree so that submodule lookups (e.g. `openfold3.entry_points`,
# `openfold3.core`, ...) are found there while Bio-TRT-specific extras
# (`openfold3.compat`, `openfold3.inference`, `openfold3.trt`) remain in
# this directory.
from pathlib import Path as _Path

__path__.append(
    str(_Path(__file__).resolve().parent.parent / "models" / "openfold-3" / "openfold3")
)
