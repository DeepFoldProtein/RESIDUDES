"""Generate & push README.md for each HF repo.

Keeps existing B200 row (merge with existing README table) for repos that already
have one; creates fresh README for the two new boltz2 repos.
"""
import io
import json
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


# --- static metadata per repo -----------------------------------------------

REPOS = {
    "DeepFoldProtein/protenix-v2-trt": {
        "title": "DeepFoldProtein/protenix-v2-trt",
        "model": "Protenix v2",
        "backend": "VANILLA",
        "tags": ["protein-structure", "tensorrt", "protenix"],
        "sources": [
            ("protenix_v2.vanilla.rtx4090",    1024),
            ("protenix_v2.vanilla.rtx6000ada", 1024),
        ],
    },
    "DeepFoldProtein/protenix-v2-cuequiv-trt": {
        "title": "DeepFoldProtein/protenix-v2-cuequiv-trt",
        "model": "Protenix v2",
        "backend": "CUEQUIV",
        "tags": ["protein-structure", "tensorrt", "protenix", "cuequivariance"],
        "sources": [
            ("protenix_v2.cuequiv.rtx6000ada", 1920),
        ],
    },
    "DeepFoldProtein/openfold3-trt": {
        "title": "DeepFoldProtein/openfold3-trt",
        "model": "OpenFold3 preview2",
        "backend": "VANILLA",
        "tags": ["protein-structure", "tensorrt", "openfold3"],
        "sources": [
            ("openfold3_p2.vanilla.rtx4090",    1024),
            ("openfold3_p2.vanilla.rtx6000ada", 1024),
        ],
    },
    "DeepFoldProtein/openfold3-cuequiv-trt": {
        "title": "DeepFoldProtein/openfold3-cuequiv-trt",
        "model": "OpenFold3 preview2",
        "backend": "CUEQUIV",
        "tags": ["protein-structure", "tensorrt", "openfold3", "cuequivariance"],
        "sources": [
            ("openfold3_p2.cuequiv.rtx6000ada", 2560),
        ],
    },
    "DeepFoldProtein/boltz2-trt": {
        "title": "DeepFoldProtein/boltz2-trt",
        "model": "Boltz-2",
        "backend": "VANILLA",
        "tags": ["protein-structure", "tensorrt", "boltz2"],
        "sources": [
            ("boltz2.vanilla.rtx4090",    896),
            ("boltz2.vanilla.rtx6000ada", 1024),
        ],
    },
    "DeepFoldProtein/boltz2-cuequiv-trt": {
        "title": "DeepFoldProtein/boltz2-cuequiv-trt",
        "model": "Boltz-2",
        "backend": "CUEQUIV",
        "tags": ["protein-structure", "tensorrt", "boltz2", "cuequivariance"],
        "sources": [
            ("boltz2.cuequiv.rtx6000ada", 3072),
        ],
    },
}


# Matches e.g. `| B200_SM10.0_TRT10.15/ | ... |` rows inside the
# "## Available Engines" table.
TABLE_ROW_RE = re.compile(r"^\|\s*`?(\w[\w\.]+_SM[\d\.]+_TRT[\d\.]+)/?`?\s*\|", re.MULTILINE)


def _contents_block(has_confidence: bool) -> str:
    if has_confidence:
        return """```
<gpu_tag>/
├── pairformer_engine/trt/              (rank0.engine, config.json)
├── token_transformer_engine/trt/       (rank0.engine, config.json)
├── confidence_pairformer_engine/trt/   (rank0.engine, config.json)
├── pairformer_safetensors/trt/         (config.json only)
├── token_transformer_safetensors/trt/  (config.json only)
└── confidence_pairformer_safetensors/trt/  (config.json only)
```"""
    return """```
<gpu_tag>/
├── pairformer_engine/trt/              (rank0.engine, config.json)
├── token_transformer_engine/trt/       (rank0.engine, config.json)
├── pairformer_safetensors/trt/         (config.json only)
└── token_transformer_safetensors/trt/  (config.json only)
```"""


def _table_row(meta_key, meta_map, max_seqlen):
    m = meta_map[meta_key]
    return (
        f"| `{m['repo_folder']}/` | {m['gpu']} | SM {m['sm']} | {m['trt']} "
        f"| {m['cuda']} | {max_seqlen} | {m['built']} |"
    )


def build_readme(repo_id, cfg, meta_map, has_confidence, existing_header_rows):
    header = [
        "---",
        "license: apache-2.0",
        "tags:",
    ]
    for t in cfg["tags"]:
        header.append(f"  - {t}")
    header.append("---")
    header.append("")
    header.append(f"# {cfg['title']}")
    header.append("")
    header.append(
        f"TensorRT BF16 engines for **{cfg['model']}**, built with "
        f"[residudes-engine](https://github.com/DeepFoldProtein/RESIDUDES)."
    )
    header.append("")
    header.append(f"Triangle attention backend: **{cfg['backend']}**")
    header.append("")
    header.append(
        "> **TRT engines are GPU-architecture and TRT-version specific.** "
        "Download the directory matching your GPU/TRT."
    )
    header.append("")
    header.append("## Available Engines")
    header.append("")
    header.append(
        "| Directory | GPU | SM | TRT Version | CUDA | Max Seqlen | Built |"
    )
    header.append(
        "|-----------|-----|----|-------------|------|-----------:|-------|"
    )
    # Existing rows (e.g. B200) first, if any.
    for row in existing_header_rows:
        header.append(row)
    # New rows.
    for key, max_seq in cfg["sources"]:
        header.append(_table_row(key, meta_map, max_seq))
    header.append("")
    header.append("## Contents (per GPU directory)")
    header.append("")
    header.append(_contents_block(has_confidence))
    header.append("")
    header.append("## Usage")
    header.append("")
    any_folder = meta_map[cfg["sources"][0][0]]["repo_folder"]
    header.append("```bash")
    header.append("pip install huggingface-hub")
    header.append(
        f"huggingface-cli download {repo_id} --local-dir engines "
        f'--include "{any_folder}/**"'
    )
    header.append("```")
    header.append("")
    header.append(
        "See the [residudes-engine](https://github.com/DeepFoldProtein/RESIDUDES) "
        "inference scripts for how to consume `engines/<gpu_tag>/`."
    )
    header.append("")
    return "\n".join(header)


def get_existing_header_rows(repo_id, api):
    """Return the pre-existing B200 (etc.) table rows to preserve."""
    try:
        p = hf_hub_download(repo_id, "README.md", token=api.token)
    except Exception:
        return []
    text = Path(p).read_text()
    rows = []
    for m in TABLE_ROW_RE.finditer(text):
        tag = m.group(1)
        # Only keep B200 rows (preserve existing external content)
        if tag.startswith("B200_"):
            full_line = text[m.start():text.find("\n", m.start())]
            rows.append(full_line)
    return rows


def main():
    meta_map = json.loads(Path("/tmp/engine_meta.json").read_text())
    api = HfApi(token=os.environ["HF_TOKEN"])
    for repo_id, cfg in REPOS.items():
        has_confidence = cfg["model"] == "Boltz-2"
        existing = get_existing_header_rows(repo_id, api)
        readme = build_readme(repo_id, cfg, meta_map, has_confidence, existing)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
            tf.write(readme)
            tf_path = tf.name
        api.upload_file(
            path_or_fileobj=tf_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update README with RTX 4090 / RTX PRO 6000 rows",
        )
        print(f"updated README: {repo_id} (existing_rows={len(existing)}, added={len(cfg['sources'])})")


if __name__ == "__main__":
    main()
