# residudes-engine's prediction runtime: Protenix v2, OpenFold 3, Boltz-2; PyTorch and TensorRT.
#
# Mounted at run time: the checkout (so one image serves any branch), the checkpoints, and the
# TensorRT engines — those are built per GPU and are not portable.
#
#   docker run --gpus all -v $PWD/residudes-engine:/opt/residudes-engine \
#       -v /data/checkpoints:/ckpt -v /data/engines:/engines residudes-engine-trt \
#       predict --model pt2 --input input.json --out out/
FROM ghcr.io/astral-sh/uv:0.11.7 AS uv
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /opt/core/
COPY install.sh /tmp/install.sh
RUN sh /tmp/install.sh && rm /tmp/install.sh

WORKDIR /work
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/residudes-engine/src \
    RESIDUDES_ENGINE_CORE=/opt/residudes-engine/core
ENTRYPOINT ["/opt/residudes-engine/bin/residudes-engine"]
CMD ["doctor"]
