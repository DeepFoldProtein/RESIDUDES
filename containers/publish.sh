#!/usr/bin/env bash
# Build and push the images with Docker, from a machine that has a daemon and the disk.
#
# This is how RESIDUDES publishes. A GHCR package belongs to the repository that first pushed it,
# and granting a second repository's Actions write access to a package is a web-UI setting with no
# REST or GraphQL equivalent, so the published images come from here rather than from CI. The two
# engine images could not be built on a hosted runner anyway: with the preinstalled toolchains
# deleted (41 GB free), docker and buildkit moved to /mnt, and the wheel cache disabled, they
# still ended in "no space left on device". CI builds and checks every image; this publishes.
#
#   containers/publish.sh                             # build and push the five large images
#   containers/publish.sh residudes-ef2               # or name the ones you want
#   containers/publish.sh residudes residudes-qa residudes-msa        # the small ones too
#   PUSH=0 containers/publish.sh                      # build only, push nothing
#   PRUNE=0 containers/publish.sh                     # keep the build cache afterwards
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
        residudes-qa|residudes-qa-cu130) echo "$REPO/.ctx-$1" ;;     # they need residudes-qa's lockfile
        *) echo "$REPO/containers/$1" ;;
    esac
}

prepare() {
    case "$1" in
        residudes-engine-*)
            cp "$REPO/containers/$1/Dockerfile" "$REPO/containers/$1/install.sh" "$REPO/residudes-engine/core/" ;;
        residudes-qa|residudes-qa-cu130)
            rm -rf "$REPO/.ctx-$1"; mkdir -p "$REPO/.ctx-$1"
            cp "$REPO/containers/$1"/* "$REPO/.ctx-$1/"
            cp "$REPO/residudes-qa/pyproject.toml" "$REPO/residudes-qa/uv.lock" "$REPO/.ctx-$1/" ;;
    esac
}

check() {                  # the same assertions CI makes for the small images
    case "$1" in
        residudes-qa-cu130)  docker run --rm --entrypoint python "$1:$TAG" -c 'import torch; print("torch", torch.__version__)' ;;
        residudes-ef2)       docker run --rm --entrypoint /opt/venv/bin/python "$1:$TAG" -c 'import torch; from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print("ESMFold2Model ok")' ;;
        residudes-engine-trt) docker run --rm --entrypoint python "$1:$TAG" -c 'import torch, tensorrt; print("tensorrt", tensorrt.__version__)' ;;
        residudes-engine-jax) docker run --rm --entrypoint /opt/venv/bin/python "$1:$TAG" -c 'import jax; from alphafold3.cpp import cif_dict; print("jax", jax.__version__)' ;;
        residudes-relax)     docker run --rm --entrypoint sh "$1:$TAG" -c 'gmx --version | grep -m1 "GROMACS version"' ;;
        residudes-qa)        docker run --rm --entrypoint python "$1:$TAG" -c 'import igraph, gemmi, xgboost; print("qa imports ok")' ;;
        residudes-msa)       docker run --rm --entrypoint sh "$1:$TAG" -c 'for b in jackhmmer kalign mmseqs; do "$b" -h >/dev/null 2>&1 || "$b" --help >/dev/null 2>&1 || exit 1; done; echo "msa binaries ok"' ;;
        # the agent image's entrypoint is the checkout, which is mounted rather than baked in
        residudes)           docker run --rm -v "$REPO/residudes:/opt/residudes" "$1:$TAG" --version ;;
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
    # 10-24 GB each; do not keep five around. Both tags: the registry one alone keeps the image.
    docker image rm -f "$name:$TAG" "$REGISTRY/$name:$TAG" >/dev/null 2>&1 || true
done
rm -rf "$REPO"/.ctx-residudes-qa*
# The builder stages stay behind as build cache: 110 GB after the two engine images, on a host whose
# root disk is shared. The cache is the daemon's, so this also drops other builds' unused layers.
if [ "${PRUNE:-1}" = 1 ]; then
    docker builder prune -af >/dev/null && echo "build cache pruned (PRUNE=0 keeps it)"
fi
