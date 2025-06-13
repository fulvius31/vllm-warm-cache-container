FROM  docker.io/vllm/vllm-openai:v0.9.1
        
WORKDIR /src

COPY warmcache.py /src/warmcache.py

ENV MODEL_NAME="facebook/opt-125m"

ENTRYPOINT ["/bin/sh", "-c", "exec python3 /src/warmcache.py --json --model $MODEL_NAME"]

