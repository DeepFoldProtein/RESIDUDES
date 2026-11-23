# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""AlphaFold 3 parameters for the PyTorch port.

The published weights are a flat list of (scope, name, array) records whose
scopes are Haiku module paths; layer stacks store one array with a leading
layer axis. We keep that exact naming: a `Scope` is a path prefix plus the
stack indices already chosen, and every module asks it for `weights` / `bias`
/ `scale` / `offset` by the Haiku name. Arrays are cast to the activation
dtype on first use and cached per dtype, which is what Haiku's parameter
getter does under `bfloat16_context`.
"""
from __future__ import annotations

import pathlib
from typing import Iterable

import io
import re
import struct

import numpy as np
import torch

WEIGHT_PATTERNS = (  # the file sets alphafold3.model.params.select_model_files knows; uncompressed first
    (r"(?P<model_name>.*)\.[0-9]+\.bin$", False), (r"(?P<model_name>.*)\.bin$", False),
    (r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$", True), (r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$", True),
    (r"(?P<model_name>.*)\.bin\.zst$", True),
)


def select_model_files(model_dir: pathlib.Path) -> tuple[list[pathlib.Path], bool]:
    files = [f for f in model_dir.iterdir() if f.is_file()]
    for pattern, compressed in WEIGHT_PATTERNS:
        rx = re.compile(pattern)
        groups: dict[str, list[pathlib.Path]] = {}
        for f in files:
            m = rx.fullmatch(f.name)
            if m:
                groups.setdefault(m.group("model_name"), []).append(f)
        if groups:
            if len(groups) > 1:
                raise RuntimeError(f"several parameter sets in {model_dir}: {sorted(groups)}")
            return sorted(next(iter(groups.values()))), compressed
    raise FileNotFoundError(f"no AlphaFold 3 parameters (af3.bin or af3.bin.zst) under {model_dir}")


def _read_record(stream) -> tuple[str, str, np.ndarray] | None:
    """One (scope, name, array) record of the published format (alphafold3.model.params)."""
    header = stream.read(5 * 4)
    if not header:
        return None
    scope_len, name_len, dtype_len, shape_len, arr_len = struct.unpack("<5i", header)
    payload = stream.read(scope_len + name_len + dtype_len + shape_len * 4)
    fmt = f"<{scope_len}s{name_len}s{dtype_len}s{shape_len}i"
    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    data = stream.read(arr_len)
    dt = dtype.decode()
    if dt == "bfloat16":
        arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        arr = arr.view(np.dtype([("bf16", np.uint16)])).view("V2").reshape(shape)   # opaque 2-byte, see _to_tensor
    else:
        arr = np.frombuffer(data, dtype=np.dtype(dt)).reshape(shape)
    return scope.decode(), name.decode(), arr


def load_records(model_dir: pathlib.Path) -> dict[tuple[str, str], np.ndarray]:
    files, compressed = select_model_files(pathlib.Path(model_dir))
    raw = b"".join(f.read_bytes() for f in files)
    if compressed:
        import zstandard
        raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    stream = io.BytesIO(raw)
    out: dict[tuple[str, str], np.ndarray] = {}
    while (rec := _read_record(stream)) is not None:
        scope, name, arr = rec
        out[(scope, name)] = arr
    if not out:
        raise FileNotFoundError(f"no AlphaFold 3 parameters under {model_dir}")
    return out


def _to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    if arr.dtype.kind == "V" or str(arr.dtype) == "bfloat16":      # bfloat16 bits → torch bf16
        bits = np.frombuffer(arr.tobytes(), dtype=np.uint16).reshape(arr.shape).copy()
        return torch.from_numpy(bits).view(torch.bfloat16).to(device)
    if arr.dtype == np.uint8:
        return torch.from_numpy(arr.copy()).to(device)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


class Params:
    """All parameters on one device, addressed by (scope, name)."""

    def __init__(self, model_dir: pathlib.Path, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        raw = load_records(pathlib.Path(model_dir))
        self.identifier: bytes = raw.pop(("__meta__", "__identifier__")).tobytes()
        self._base: dict[tuple[str, str], torch.Tensor] = {k: _to_tensor(v, self.device) for k, v in raw.items()}
        self._cast: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
        self.used: set[tuple[str, str]] = set()

    def keys(self) -> Iterable[tuple[str, str]]:
        return self._base.keys()

    def has(self, scope: str, name: str) -> bool:
        return (scope, name) in self._base

    def get(self, scope: str, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        key = (scope, name)
        if key not in self._base:
            raise KeyError(f"AlphaFold 3 parameter missing: {scope} / {name}")
        self.used.add(key)
        t = self._base[key]
        if dtype is None or t.dtype == dtype:
            return t
        ck = (scope, name, dtype)
        if ck not in self._cast:
            self._cast[ck] = t.to(dtype)
        return self._cast[ck]

    def unused(self) -> list[tuple[str, str]]:
        return sorted(k for k in self._base if k not in self.used)

    def root(self) -> "Scope":
        return Scope(self, "", ())


class Scope:
    """A Haiku-style module path with the layer-stack indices chosen so far."""

    __slots__ = ("params", "prefix", "idx")

    def __init__(self, params: Params, prefix: str, idx: tuple[int, ...]):
        self.params, self.prefix, self.idx = params, prefix, idx

    # path building
    def child(self, name: str) -> "Scope":
        """Sub-module scope (adds a `/`)."""
        return Scope(self.params, f"{self.prefix}/{name}" if self.prefix else name, self.idx)

    def stack(self, kind: str, i: int) -> "Scope":
        """Enter a layer stack: `kind` is the Haiku scope element
        (`__layer_stack_no_per_layer`, `__layer_stack_no_per_layer_1`,
        `__layer_stack_with_per_layer`) and `i` the layer picked from the leading axis."""
        return Scope(self.params, f"{self.prefix}/{kind}", self.idx + (i,))

    def indexed(self, i: int) -> "Scope":
        """Pick another leading index without changing the path (nested stacks)."""
        return Scope(self.params, self.prefix, self.idx + (i,))

    # parameter access
    def _join(self, name: str) -> str:
        if not name:
            return self.prefix
        return f"{self.prefix}/{name}" if self.prefix else name

    def has(self, name: str, kind: str = "weights") -> bool:
        return self.params.has(self._join(name), kind)

    def get(self, name: str, kind: str = "weights", dtype: torch.dtype | None = None) -> torch.Tensor:
        """`name` is the sub-module (`""` for a parameter that sits directly in this scope,
        such as OuterProductMean's output_w); `kind` the array name."""
        scope = self._join(name)
        t = self.params.get(scope, kind, dtype)
        for i in self.idx:
            t = t[i]
        return t

    def __repr__(self) -> str:
        return f"Scope({self.prefix!r}, idx={self.idx})"
