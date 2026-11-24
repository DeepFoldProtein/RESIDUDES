"""Upload TRT engines to DeepFoldProtein HF repos.

Skips rank0.safetensors (huge, follows existing repo convention of config-only).
Idempotent: re-running will overwrite any existing files of the same path.
"""
import os
import sys
import time
from huggingface_hub import HfApi

ENGINES_ROOT = "/gpfs/deepfold/casp/casp17/engines"

# (src_gpu, src_subdir, repo_id, dest_folder)
UPLOADS = [
    ("rtx4090",    "protenix_v2_vanilla",  "DeepFoldProtein/protenix-v2-trt",          "RTX4090_SM8.9_TRT10.15"),
    ("rtx4090",    "openfold3_p2_vanilla", "DeepFoldProtein/openfold3-trt",            "RTX4090_SM8.9_TRT10.15"),
    ("rtx4090",    "boltz2_vanilla",       "DeepFoldProtein/boltz2-trt",               "RTX4090_SM8.9_TRT10.15"),
    ("rtx6000ada", "protenix_v2_vanilla",  "DeepFoldProtein/protenix-v2-trt",          "RTXPRO6000_SM12.0_TRT10.15"),
    ("rtx6000ada", "openfold3_p2_vanilla", "DeepFoldProtein/openfold3-trt",            "RTXPRO6000_SM12.0_TRT10.15"),
    ("rtx6000ada", "boltz2_vanilla",       "DeepFoldProtein/boltz2-trt",               "RTXPRO6000_SM12.0_TRT10.15"),
    ("rtx6000ada", "protenix_v2_cuequiv",  "DeepFoldProtein/protenix-v2-cuequiv-trt",  "RTXPRO6000_SM12.0_TRT10.16"),
    ("rtx6000ada", "openfold3_p2_cuequiv", "DeepFoldProtein/openfold3-cuequiv-trt",    "RTXPRO6000_SM12.0_TRT10.16"),
    ("rtx6000ada", "boltz2_cuequiv",       "DeepFoldProtein/boltz2-cuequiv-trt",       "RTXPRO6000_SM12.0_TRT10.16"),
]

# Include engine .engine files and all config.json (covers safetensors/trt/config.json too).
# Exclude the giant rank0.safetensors intermediates.
ALLOW_PATTERNS = [
    "*_engine/trt/rank0.engine",
    "*/trt/config.json",
]


def main(only=None):
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    t_all0 = time.time()
    for i, (gpu, sub, repo, dest) in enumerate(UPLOADS, 1):
        if only is not None and only != i:
            continue
        src = os.path.join(ENGINES_ROOT, gpu, sub)
        if not os.path.isdir(src):
            print(f"[{i}/9] SKIP missing: {src}", flush=True)
            continue
        print(f"[{i}/9] {src}  ->  {repo}/{dest}", flush=True)
        t0 = time.time()
        api.upload_folder(
            folder_path=src,
            repo_id=repo,
            repo_type="model",
            path_in_repo=dest,
            allow_patterns=ALLOW_PATTERNS,
            commit_message=f"Add {dest} engines from {gpu}/{sub}",
        )
        print(f"[{i}/9] done in {time.time()-t0:.1f}s", flush=True)
    print(f"TOTAL elapsed: {time.time()-t_all0:.1f}s", flush=True)


if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(only=only)
