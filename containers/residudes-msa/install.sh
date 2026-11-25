#!/bin/sh
# The MSA search tools. No databases: they are 630 GB and get bind-mounted (see README).
#
# What is here covers `residudes-engine msa --source colabfold|mmseqs2|plmmsa` and the agent's own
# family crawl (hmmsearch / hmmscan). `--source af3` runs AlphaFold 3's data pipeline, which needs
# the AF3 code and its JAX environment, so it belongs to the af3 image, not this one.
set -eu

MMSEQS=${MMSEQS:-20-ae60e}          # soedinglab/MMseqs2 release tag
ARCH=${ARCH:-avx2}                  # avx2 | sse41 | arm64 — the static build to take

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl \
    hmmer \
    kalign
rm -rf /var/lib/apt/lists/*

curl -fsSL -o /tmp/mmseqs.tar.gz \
    "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS}/mmseqs-linux-${ARCH}.tar.gz"
tar -xzf /tmp/mmseqs.tar.gz -C /opt
rm /tmp/mmseqs.tar.gz
ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs

# every binary the engine and the agent call by name, proved to run
for b in jackhmmer nhmmer hmmsearch hmmbuild hmmalign hmmscan kalign; do
    "$b" -h >/dev/null 2>&1 || "$b" --help >/dev/null 2>&1 || { echo "missing or broken: $b" >&2; exit 1; }
done
mmseqs version
jackhmmer -h | head -2
