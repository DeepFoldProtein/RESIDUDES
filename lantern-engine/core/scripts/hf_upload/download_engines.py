"""Mirror all HF engine repos into /gpfs/deepfold/casp/casp17/engines/<repo-name>/.

Structure (per repo):
    engines/<repo-name>/
        B200_SM10.0_TRT10.15/...                  (if present)
        RTX4090_SM8.9_TRT10.15/...
        RTXPRO6000_SM12.0_TRT10.15/...            (vanilla repos)
        RTXPRO6000_SM12.0_TRT10.16/...            (cuequiv repos)
        README.md
"""
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = [
    "DeepFoldProtein/protenix-v2-trt",
    "DeepFoldProtein/protenix-v2-cuequiv-trt",
    "DeepFoldProtein/openfold3-trt",
    "DeepFoldProtein/openfold3-cuequiv-trt",
    "DeepFoldProtein/boltz2-trt",
    "DeepFoldProtein/boltz2-cuequiv-trt",
]

ENGINES_ROOT = Path("/gpfs/deepfold/casp/casp17/engines")


def main():
    token = os.environ["HF_TOKEN"]
    t_all0 = time.time()
    for i, repo in enumerate(REPOS, 1):
        short = repo.split("/", 1)[1]
        dest = ENGINES_ROOT / short
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(REPOS)}] {repo}  ->  {dest}", flush=True)
        t0 = time.time()
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=str(dest),
            token=token,
            max_workers=8,
        )
        print(f"[{i}/{len(REPOS)}] done in {time.time()-t0:.1f}s", flush=True)
    print(f"TOTAL elapsed: {time.time()-t_all0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
