# residudes-qa on the CPU: clustering, the interface features and the plots.
#
# The scorers that need a GPU use residudes-qa-cu130 instead (same recipe, the cu130 extra on top
# of an NVIDIA base). Data files (data/files) and the pool are mounted, not baked in.
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv

# only the two files that pin the environment; the package itself is mounted at run time
COPY pyproject.toml uv.lock /opt/qa/
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.
LABEL org.opencontainers.image.title="RESIDUDES QA (CPU)" \
      org.opencontainers.image.description="Quality assessment of a prediction pool on CPU: interface scoring, clustering and the confidence plots. No torch; see residudes-qa-cu130 for the GPU scorers." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-qa/src
ENTRYPOINT ["/opt/venv/bin/python", "-m", "residudes_qa"]
CMD ["doctor"]
