vllm serve \
  --omni black-forest-labs/FLUX.1-dev \
  --stage-config $1 \
  --port 59085 \
  --dtype float16
