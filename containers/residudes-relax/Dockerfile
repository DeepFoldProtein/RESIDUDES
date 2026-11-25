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

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.
LABEL org.opencontainers.image.title="RESIDUDES relaxation tools" \
      org.opencontainers.image.description="GROMACS 2024.2 and AmberTools 24.8 for restrained relaxation. Rosetta and the relax pipeline are bind-mounted." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /work
ENV PATH=/opt/relax/bin:$PATH \
    AMBERHOME=/opt/relax \
    GMXLIB=/opt/relax/share/gromacs/top
ENTRYPOINT []
CMD ["gmx", "--version"]
