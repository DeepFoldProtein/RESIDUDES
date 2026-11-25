# AlphaFold 3 and the JAX backends of Protenix, OpenFold 3 and Boltz-2.
#
# Mounted at run time: the checkout, the checkpoints, and af3.bin — which Google DeepMind
# licenses to each user and which therefore cannot be inside any image.
#
# Build from residudes-engine/core/ as the context, because the vendored model sources are
# copied in: docker build -f containers/residudes-engine-jax/Dockerfile residudes-engine/core
#
#   docker run --gpus all -v $PWD/residudes-engine:/opt/residudes-engine \
#       -v /data/af3:/af3 residudes-engine-jax predict --model af3 --backend jax --input input.json
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04 AS builder

COPY --from=uv /uv /usr/local/bin/uv
COPY jaxenv/pyproject.toml jaxenv/uv.lock /opt/jaxenv/
COPY models/alphafold3 /opt/models/alphafold3
COPY models/openfold-3 /opt/models/openfold-3
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh system
RUN sh /tmp/install.sh sync
RUN sh /tmp/install.sh split 4          # site-packages into four buckets of similar size

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.

# One layer per bucket: an 11.5 GB layer is what fails to pull, four 3 GB ones do not.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh system
COPY --from=builder /split/rest/venv /opt/venv
COPY --from=builder /split/1/ /
COPY --from=builder /split/2/ /
COPY --from=builder /split/3/ /
COPY --from=builder /split/4/ /
RUN sh /tmp/install.sh check && rm /tmp/install.sh

LABEL org.opencontainers.image.title="RESIDUDES engine (JAX + AlphaFold 3)" \
      org.opencontainers.image.description="AlphaFold 3 with its C++ extension already compiled, and the JAX backends. Contains vendored AlphaFold 3 code under CC BY-NC-SA 4.0: non-commercial. af3.bin is licensed per user and is bind-mounted." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"
WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-engine/src \
    RESIDUDES_ENGINE_CORE=/opt/residudes-engine/core
ENTRYPOINT ["/opt/residudes-engine/bin/residudes-engine"]
CMD ["doctor"]
