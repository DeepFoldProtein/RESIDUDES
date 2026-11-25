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

WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-qa/src
ENTRYPOINT ["/opt/venv/bin/python", "-m", "residudes_qa"]
CMD ["doctor"]
