#!/usr/bin/env bash
# Build and push the five large images with Docker, from a machine that has a daemon and the disk.
#
# GitHub's hosted runners cannot do this: with the preinstalled toolchains deleted (41 GB free),
# with docker and buildkit moved to /mnt (41 GB), and with the wheel cache disabled, the TensorRT
# and JAX builds still ended in "no space left on device". CI therefore builds the three small
# images and this script publishes the rest.
#
#   containers/publish-large.sh                       # build and push all five
#   containers/publish-large.sh residudes-ef2         # or name the ones you want
#   PUSH=0 containers/publish-large.sh                # build only, push nothing
#
# Requirements: docker, ~60 GB free on the docker data root, and a login that may write packages:
#   echo "$GHCR_TOKEN" | docker login ghcr.io -u <you> --password-stdin
# where the token is a PAT with write:packages. A `gh` login is not enough - its token carries
# repo and workflow scopes, not packages.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
REGISTRY=${REGISTRY:-ghcr.io/deepfoldprotein}
TAG=${TAG:-latest}
PUSH=${PUSH:-1}
ALL="residudes-qa-cu130 residudes-ef2 residudes-engine-trt residudes-engine-jax residudes-relax"
IMAGES=${*:-$ALL}

context_for() {            # each recipe's build context, which is not always its own directory
    case "$1" in
        residudes-engine-*) echo "$REPO/residudes-engine/core" ;;   # it copies the vendored model sources
        residudes-qa-cu130) echo "$REPO/.ctx-qa-cu130" ;;           # it needs residudes-qa's lockfile
        *) echo "$REPO/containers/$1" ;;
    esac
}

prepare() {
    case "$1" in
        residudes-engine-*)
            cp "$REPO/containers/$1/Dockerfile" "$REPO/containers/$1/install.sh" "$REPO/residudes-engine/core/" ;;
        residudes-qa-cu130)
            rm -rf "$REPO/.ctx-qa-cu130"; mkdir -p "$REPO/.ctx-qa-cu130"
            cp "$REPO/containers/$1"/* "$REPO/.ctx-qa-cu130/"
            cp "$REPO/residudes-qa/pyproject.toml" "$REPO/residudes-qa/uv.lock" "$REPO/.ctx-qa-cu130/" ;;
    esac
}

check() {                  # the same assertions CI makes for the small images
    case "$1" in
        residudes-qa-cu130)  docker run --rm --entrypoint python "$1:$TAG" -c 'import torch; print("torch", torch.__version__)' ;;
        residudes-ef2)       docker run --rm --entrypoint /opt/venv/bin/python "$1:$TAG" -c 'import torch; from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print("ESMFold2Model ok")' ;;
        residudes-engine-trt) docker run --rm --entrypoint python "$1:$TAG" -c 'import torch, tensorrt; print("tensorrt", tensorrt.__version__)' ;;
        residudes-engine-jax) docker run --rm --entrypoint /opt/venv/bin/python "$1:$TAG" -c 'import jax; from alphafold3.cpp import cif_dict; print("jax", jax.__version__)' ;;
        residudes-relax)     docker run --rm --entrypoint sh "$1:$TAG" -c 'gmx --version | grep -m1 "GROMACS version"' ;;
    esac
}

for name in $IMAGES; do
    echo "▶ $name"
    prepare "$name"
    docker build -t "$name:$TAG" -f "$REPO/containers/$name/Dockerfile" "$(context_for "$name")"
    check "$name"
    if [ "$PUSH" = 1 ]; then
        # AlphaFold 3's vendored code is CC BY-NC-SA 4.0: this image is non-commercial, and
        # publishing it is a decision, not a default.
        if [ "$name" = residudes-engine-jax ] && [ "${PUBLISH_JAX:-0}" != 1 ]; then
            echo "  not pushed: it contains CC BY-NC-SA AlphaFold 3 code (PUBLISH_JAX=1 to push it anyway)"
            continue
        fi
        docker tag "$name:$TAG" "$REGISTRY/$name:$TAG"
        docker push "$REGISTRY/$name:$TAG"
        echo "  pushed $REGISTRY/$name:$TAG"
    fi
    docker image rm -f "$name:$TAG" >/dev/null 2>&1 || true     # 10 GB each; do not keep five around
done
rm -rf "$REPO/.ctx-qa-cu130"
