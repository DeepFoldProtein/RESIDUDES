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

WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-qa/src
ENTRYPOINT ["/opt/venv/bin/python", "-m", "residudes_qa"]
CMD ["doctor"]
