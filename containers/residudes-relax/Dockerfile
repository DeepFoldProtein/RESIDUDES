# The relaxation tools: GROMACS 2024.2 and AmberTools 24.8, pinned to what CASP17 ran.
#
# Mounted, not installed: the relax pipeline (it lives in the campaign workspace) and Rosetta
# (you license and build that yourself).
#
#   docker run -v /path/to/gromacs/relax:/opt/pipeline -v /path/to/rosetta:/opt/rosetta \
#       -v $PWD:/work residudes-relax /opt/pipeline/minimization.sh model.pdb --bb-fc 1000000
FROM debian:13-slim

COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

WORKDIR /work
ENV PATH=/opt/relax/bin:$PATH \
    AMBERHOME=/opt/relax \
    GMXLIB=/opt/relax/share/gromacs/top
ENTRYPOINT []
CMD ["gmx", "--version"]
