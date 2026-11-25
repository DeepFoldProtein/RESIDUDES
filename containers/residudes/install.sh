#!/bin/sh
# Everything the agent image needs beyond its base layers. Run by both the Dockerfile and the
# Apptainer definition, so the two cannot drift apart. Node comes from the official node image
# (copied in by the caller), not from a script piped into a shell.
set -eu

PY=${PY:-python3}
TYPST=${TYPST:-0.13.1}

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    librsvg2-bin \
    tmux \
    xz-utils
rm -rf /var/lib/apt/lists/*

# the agent's own dependencies, at the versions residudes/pyproject.toml pins
"$PY" -m pip install --no-cache-dir --root-user-action=ignore \
    "claude-agent-sdk==0.2.187" "gemmi==0.6.5"

# `residudes agent step` drives the Claude Code CLI through the SDK
# No agent CLI here. Claude Code, Codex and the others authenticate as a person and are licensed
# to one, so baking one into a shared image fixes a choice that is not the image's to make and
# ships an account's tool to whoever pulls it. `residudes setup` picks from what the machine has
# and says how to install one otherwise; mount it, or install it in the container at run time.

# molterm renders structures in the terminal, and its MCP server gives the agent tools for it:
# fetch, screenshot, select, measure, align, dssp. A conversation about a target is mostly about
# shapes, and describing one in prose when it could be shown is a poor trade. The npm install
# pulls the prebuilt molterm binary for this platform.
npm install -g --no-fund --no-audit molterm-mcp

# molterm-mcp keeps the viewer in its own vendor/bin, which the MCP server knows about and a
# shell does not. The skill uses both - the MCP tools in a conversation, the command line in a
# script - so the binary goes on PATH too.
ln -sf "$(npm root -g)/molterm-mcp/vendor/bin/molterm" /usr/local/bin/molterm

npm cache clean --force

# `residudes slides` compiles the report with typst (a .tar.xz release, hence xz-utils)
curl -fsSL "https://github.com/typst/typst/releases/download/v${TYPST}/typst-x86_64-unknown-linux-musl.tar.xz" \
    | tar -xJ -C /tmp
mv /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst
rm -rf /tmp/typst-x86_64-unknown-linux-musl

node --version
molterm --version 2>/dev/null | head -1 || echo "molterm: binary not fetched for this platform"
typst --version
"$PY" -c "import gemmi, claude_agent_sdk; print('gemmi', gemmi.__version__)"
