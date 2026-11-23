"""Single source of truth for tags: models, backends, GPUs, engine layouts.

Tags are the only names used across CLI, env vars, output metadata and
engine paths. Long names live here and in docs only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ("trt", "jax", "torch")

# nvidia-smi compute_cap → GPU tag (also the first path component under <engines>/)
GPUS: dict[str, str] = {
    "8.9": "rtx4090",
    "12.0": "rtx6000ada",   # RTX PRO 6000 Blackwell — tag kept for on-disk engine compatibility
    "10.0": "b200",
    "9.0": "h100",
    "8.0": "a100",
}


KERNELS = ("vanilla", "cuequiv")   # TensorRT triangle-attention kernel, fixed at engine build time


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    family: str                 # code unit: protenix | openfold3 | boltz
    name: str                   # human-readable
    checkpoint: str             # relative to <site.checkpoints>
    engine_prefix: str          # on-disk engine dir = <engines>/<gpu>/<prefix>_<kernel>_<maxN>
    engines: dict[str, str]     # gpu tag → known engine dir (relative to <site.engines>)
    weights: dict = field(default_factory=dict)   # how `lantern-engine setup checkpoints` obtains the checkpoint
    extra: dict = field(default_factory=dict)
    backends: tuple[str, ...] = BACKENDS          # which of trt / jax / torch this model runs on
    fallback: str = "torch"                       # what `--backend auto` uses when no engines exist

    def engine_dir(self, engines_root: Path, gpu_tag: str) -> Path | None:
        rel = self.engines.get(gpu_tag)
        return engines_root / rel if rel else None

    def engine_dir_for_build(self, engines_root: Path, gpu_tag: str, kernel: str, max_seqlen: int) -> Path:
        return engines_root / gpu_tag / f"{self.engine_prefix}_{kernel}_{max_seqlen}"


# Engine directory names follow the established on-disk / HuggingFace
# convention (protenix_v2_vanilla_5120 …). Data is not renamed; the registry resolves it.
MODELS: dict[str, ModelSpec] = {
    "pt2": ModelSpec(
        tag="pt2", family="protenix", name="Protenix v2 (ByteDance)",
        checkpoint="protenix-v2.pt", engine_prefix="protenix_v2",
        engines={
            "rtx4090": "rtx4090/protenix_v2_vanilla",
            "rtx6000ada": "rtx6000ada/protenix_v2_cuequiv_4096",
            "b200": "b200/protenix_v2_vanilla_5120",
        },
        # Publisher URL first (denied from some networks), then a public mirror whose
        # file is byte-identical to the publisher's (same size and sha256; both checked).
        weights={"method": "url",
                 "urls": [
                     "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt",
                     "https://huggingface.co/fredzzp/protenix-v2/resolve/main/checkpoint/protenix_base_default_v2.0.0.pt",
                 ],
                 "size": 1859785497,
                 "sha256": "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"},
        extra={"model_name": "protenix-v2"},
    ),
    "of3": ModelSpec(
        tag="of3", family="openfold3", name="OpenFold 3 preview-2 (AQ Lab)",
        checkpoint="of3-p2-155k.pt", engine_prefix="openfold3_p2",
        engines={
            "rtx4090": "rtx4090/openfold3_p2_vanilla",
            "rtx6000ada": "rtx6000ada/openfold3_p2_cuequiv_3072",
            "b200": "b200/openfold3_p2_vanilla_4096",
        },
        # public S3 via the model's own downloader (runs inside the runtime venv)
        weights={"method": "openfold3", "name": "openfold3-p2-155k", "size": 2287928196,
                 "sha256": "af09eac4f29cef856633af07558cb143226fe95ebbef2c20921769d4a5f4bee4"},
    ),
    "bt2": ModelSpec(
        tag="bt2", family="boltz", name="Boltz-2",
        checkpoint="boltz2/boltz2_conf.ckpt", engine_prefix="boltz2",
        engines={
            "rtx4090": "rtx4090/boltz2_vanilla_896",
            "rtx6000ada": "rtx6000ada/boltz2_cuequiv_5120",
            "b200": "b200/boltz2_vanilla_5120",
        },
        # boltz's own downloader fetches boltz2_conf.ckpt + mols.tar into <checkpoints>/boltz2
        weights={"method": "boltz2", "size": 2286561469,
                 "sha256": "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"},
    ),
    "af3": ModelSpec(
        tag="af3", family="alphafold3", name="AlphaFold 3 (Google DeepMind)",
        checkpoint="af3/af3.bin", engine_prefix="alphafold3", engines={},
        backends=("trt", "jax", "torch"), fallback="jax",   # TensorRT (core/alphafold3/trt), native JAX, the PyTorch port
        # The parameters are licensed to each user by Google DeepMind; lantern-engine downloads nothing.
        weights={"method": "manual",
                 "note": "request access at https://github.com/google-deepmind/alphafold3 (WEIGHTS_TERMS_OF_USE.md); "
                         "put af3.bin or af3.bin.zst in <checkpoints>/af3/ or set [site] af3_weights"},
    ),
}


def engine_candidates(engines_root: Path, gpu_tag: str, spec: ModelSpec) -> list[Path]:
    """Every usable engine dir for this model and GPU: the known (CASP17) dir plus
    local `engines build` outputs `<engines>/<gpu>/<prefix>_<kernel>_<N>`."""
    def ok(d: Path) -> bool:
        return (d / "pairformer_engine" / "trt" / "rank0.engine").is_file()

    cands: list[Path] = []
    known = spec.engine_dir(engines_root, gpu_tag)
    if known is not None and ok(known):
        cands.append(known)
    for d in sorted((engines_root / gpu_tag).glob(f"{spec.engine_prefix}_*")):
        if d.is_dir() and ok(d) and d.resolve() not in {c.resolve() for c in cands}:
            cands.append(d)
    return cands


def find_engine_dir(engines_root: Path, gpu_tag: str, spec: ModelSpec,
                    n_tokens: int | None = None) -> Path | None:
    """Pick the engine set for an input.

    With `n_tokens` known: the engine with the *smallest* max_seq_len that still
    covers the input (best fit). TensorRT picks kernels for the optimisation
    profile it was built with, so a 2048-token engine runs a 1315-token Boltz-2
    pairformer 1.37× faster than the 5120-token one (docs/benchmark_details.md). When no
    engine covers the input, or n_tokens is unknown, the largest wins."""
    cands = engine_candidates(engines_root, gpu_tag, spec)
    if not cands:
        return None
    sized = [(engine_max_seq_len(d) or 0, d) for d in cands]
    if n_tokens is not None:
        fitting = [(m, d) for m, d in sized if m >= n_tokens]
        if fitting:
            return min(fitting, key=lambda md: md[0])[1]
    return max(sized, key=lambda md: md[0])[1]

# Registered for completeness (results on disk use these tags); not runnable via lantern-engine v0.
EXTERNAL_MODELS = {
    "ef2": "ESMFold2 (inferences/esmfold2/)",
}


def get_model(tag: str) -> ModelSpec:
    try:
        return MODELS[tag]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise SystemExit(f"unknown model tag {tag!r} (known: {known})") from None


def gpu_tag_for(compute_cap: str) -> str | None:
    return GPUS.get(compute_cap.strip())


def engine_max_seq_len(engine_dir: Path) -> int | None:
    """Build-time upper bound read from the pairformer engine config."""
    cfg = engine_dir / "pairformer_engine" / "trt" / "config.json"
    try:
        d = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return None
    pc = d.get("pretrained_config", d)
    v = pc.get("max_seq_len")
    return int(v) if v is not None else None
