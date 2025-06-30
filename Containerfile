FROM registry.access.redhat.com/ubi9/podman:latest

RUN dnf --assumeyes update \
        && dnf --assumeyes install \
        cmake \
        gcc-c++ \
        git \
        python3-devel \
        python3-numpy \
        python3-pyyaml \
        python3-setuptools \
        && dnf clean all
WORKDIR /src
COPY warmcache.py /src/warmcache.py

ENV MODEL_NAME="facebook/opt-125m"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
RUN /root/.local/bin/uv pip install --system -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly

ENTRYPOINT ["/bin/sh", "-c", "exec python3 /src/warmcache.py --json --model $MODEL_NAME"]

