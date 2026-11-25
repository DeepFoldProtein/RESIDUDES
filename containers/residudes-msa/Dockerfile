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

WORKDIR /work
ENV PYTHONPATH=/opt/residudes-engine/src
ENTRYPOINT ["/opt/residudes-engine/bin/residudes-engine"]
CMD ["doctor"]
