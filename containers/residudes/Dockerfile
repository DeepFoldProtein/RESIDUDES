# The campaign agent: dossier, analysis, report, slides, gates, slate, packaging.
#
# Node comes from the official node image rather than Debian's (the Claude Code CLI needs >= 22,
# trixie ships 20) and rather than a setup script piped into a shell.
#
# Not here: sbatch and pueue (the host's schedulers — a container is what the job runs inside),
# and the predictors and relaxation, which have images of their own. See containers/README.md.
FROM node:22-slim AS node
FROM python:3.12-slim

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

# the checkout is mounted, not copied: one image serves whatever branch is checked out
WORKDIR /work
ENV PYTHONPATH=/opt/residudes/src
ENTRYPOINT ["/opt/residudes/bin/residudes"]
CMD ["doctor"]
