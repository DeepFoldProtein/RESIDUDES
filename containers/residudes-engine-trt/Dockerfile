# residudes-engine's prediction runtime: Protenix v2, OpenFold 3, Boltz-2; PyTorch and TensorRT.
#
# Mounted at run time: the checkout (so one image serves any branch), the checkpoints, and the
# TensorRT engines — those are built per GPU and are not portable.
#
#   docker run --gpus all -v $PWD/residudes-engine:/opt/residudes-engine \
#       -v /data/checkpoints:/ckpt -v /data/engines:/engines residudes-engine-trt \
#       predict --model pt2 --input input.json --out out/
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04 AS builder

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /opt/core/
COPY models/openfold-3 /opt/core/models/openfold-3
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh system
RUN sh /tmp/install.sh sync
RUN sh /tmp/install.sh split 4          # site-packages into four buckets of similar size


# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.

# One layer per bucket: a 9 GB layer is what fails to pull, four 2-3 GB ones do not.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh system
COPY --from=builder /split/rest/venv /opt/venv
COPY --from=builder /split/1/ /
COPY --from=builder /split/2/ /
COPY --from=builder /split/3/ /
COPY --from=builder /split/4/ /
RUN sh /tmp/install.sh check && rm /tmp/install.sh

LABEL org.opencontainers.image.title="RESIDUDES engine (PyTorch + TensorRT)" \
      org.opencontainers.image.description="Protenix v2, OpenFold 3 and Boltz-2 on PyTorch or TensorRT 10.16. Checkpoints and the per-GPU TensorRT engines are bind-mounted." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"
WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-engine/src \
    RESIDUDES_ENGINE_CORE=/opt/residudes-engine/core
ENTRYPOINT ["/opt/residudes-engine/bin/residudes-engine"]
CMD ["doctor"]
