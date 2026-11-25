# ESMFold2: its own python 3.12, torch and the Biohub transformers fork.
#
# Two things are mounted, not baked in: the runner scripts (a private fork's
# inferences/esmfold2/) at /opt/esm, and the weights, which are a HuggingFace cache of
# biohub/ESMFold2 — multi-GB and not ours to redistribute.
#
#   docker run --gpus all -v /path/to/esm:/opt/esm -v /data/esmfold2_hf:/hf \
#       -e HF_HOME=/hf residudes-ef2 /opt/esm/inferences/esmfold2/run_inference.py in.json out/
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

# What the package page shows for this image. Without these every package inherits the
# repository's description and the eight pages read identically.
LABEL org.opencontainers.image.title="RESIDUDES ESMFold2" \
      org.opencontainers.image.description="ESMFold2: python 3.12, torch 2.12 and the Biohub transformers fork. The runner scripts and the HuggingFace weight cache are bind-mounted." \
      org.opencontainers.image.source="https://github.com/DeepFoldProtein/RESIDUDES" \
      org.opencontainers.image.documentation="https://github.com/DeepFoldProtein/RESIDUDES/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /work
ENV HF_HOME=/hf
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
