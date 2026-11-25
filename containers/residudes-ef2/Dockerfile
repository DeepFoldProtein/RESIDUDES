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

WORKDIR /work
ENV HF_HOME=/hf
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
