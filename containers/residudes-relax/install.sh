#!/bin/sh
# The relaxation tools: GROMACS and AmberTools.
#
# `residudes relax` drives gromacs/relax/minimization.sh, which runs both orders of the two
# engines under position restraints (gromacs-rosetta and rosetta-gromacs). That script calls
# gmx, and tleap/antechamber/parmchk2/reduce from AmberTools for anything with a ligand.
#
# Two things are mounted rather than installed:
#   - the pipeline itself, which lives in the campaign workspace, not in this repository
#   - Rosetta, which you license and build yourself (academic licences are free but are yours)
#
# Versions are pinned to what CASP17 ran: GROMACS 2024.1 on the cluster - conda-forge has no
# 2024.1, so 2024.2 is the nearest build - and AmberTools 24.8, the exact conda package the
# campaign's env was created from. A CPU build: this is restrained energy minimisation, and
# measured throughput on this hardware never favoured the GPU path for it.
set -eu

GROMACS_VERSION=${GROMACS_VERSION:-2024.2}
AMBERTOOLS_VERSION=${AMBERTOOLS_VERSION:-24.8}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl bzip2 procps
rm -rf /var/lib/apt/lists/*

curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/2.0.5" | tar -xvj -C /usr/local bin/micromamba

export MAMBA_ROOT_PREFIX=/opt/conda
/usr/local/bin/micromamba create -y -p /opt/relax -c conda-forge \
    "gromacs=${GROMACS_VERSION}" "ambertools=${AMBERTOOLS_VERSION}"
/usr/local/bin/micromamba clean -y --all
rm -rf /opt/conda/pkgs

# every binary the pipeline calls, run rather than assumed present
PATH=/opt/relax/bin:$PATH
export AMBERHOME=/opt/relax
gmx --version | grep -E 'GROMACS version|Precision'
for t in tleap antechamber parmchk2 reduce; do
    command -v "$t" >/dev/null || { echo "missing: $t" >&2; exit 1; }
done
echo "tleap  : $(tleap -h 2>&1 | head -1)"
echo "antechamber: $(antechamber -h 2>&1 | grep -m1 -i 'usage\|antechamber' || echo present)"
echo "reduce : $(reduce -version 2>&1 | head -1)"
printf 'quit\n' | tleap -f - >/dev/null 2>&1 && echo "tleap runs" || echo "tleap ran with warnings"
