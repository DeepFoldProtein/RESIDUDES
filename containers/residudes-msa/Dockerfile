# MSA search: HMMER, MMseqs2 and kalign, with no databases inside.
#
# The databases are 630 GB and licensed by their own sources, so they are bind-mounted:
#   docker run -v /gpfs/database/af3:/db residudes-msa msa --fasta q.fasta --source mmseqs2 --db /db
#
# AlphaFold 3's own data pipeline (--source af3) needs the AF3 code and its JAX environment and
# lives in the af3 image. See containers/README.md.
FROM python:3.12-slim

COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.
LABEL org.opencontainers.image.title="RESIDUDES MSA tools" \
      org.opencontainers.image.description="HMMER, MMseqs2 and kalign for MSA search. 67 MB; the sequence databases are bind-mounted, not included." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /work
ENV PYTHONPATH=/opt/residudes-engine/src
ENTRYPOINT ["/opt/residudes-engine/bin/residudes-engine"]
CMD ["doctor"]
