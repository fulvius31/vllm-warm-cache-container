FROM  docker.io/vllm/vllm-openai:v0.9.2
        
WORKDIR /src

COPY warmcache.py /src/warmcache.py

ENV MODEL_NAME="facebook/opt-125m"
ENV VLLM_USE_V1=1 

ENTRYPOINT ["/bin/sh", "-c", "exec python3 /src/warmcache.py --json --model $MODEL_NAME"]

