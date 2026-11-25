# residudes-qa with the GPU scorers: the cu130 extra, torch 2.11 — the build CASP17 scored with.
#
# On NVIDIA's CUDA runtime image, because that extra expects libnvrtc and libcudart from the
# system; on python:3.12-slim `import torch` fails outright. Run it with --nv (apptainer) or
# --gpus all (docker): those give the driver, this image gives the runtime.
#
# The CPU image (containers/residudes-qa) is the one to use when there is no card: it is 725 MB
# against this one's several GB, and every test passes in it.
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock /opt/qa/
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.
LABEL org.opencontainers.image.title="RESIDUDES QA (CUDA 13)" \
      org.opencontainers.image.description="The QA stack with torch 2.11+cu130 for the GPU scorers and clustering. Run with --gpus all or apptainer --nv." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-qa/src
ENTRYPOINT ["/opt/venv/bin/python", "-m", "residudes_qa"]
CMD ["doctor"]
