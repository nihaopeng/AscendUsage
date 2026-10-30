#!/bin/sh
set -eu

# Run on physical NPU 6 and 7. They are exposed to vLLM as logical devices
# 0 and 1, so tensor parallelism must match the two visible devices.
export ASCEND_RT_VISIBLE_DEVICES=6,7

# Ascend runtime settings recommended for multi-NPU vLLM serving.
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

MODEL_PATH=/mnt/nvme1p1/pengyt/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VLLM_BIN="$SCRIPT_DIR/.venv/bin/vllm"

if [ ! -x "$VLLM_BIN" ]; then
    echo "vLLM executable not found: $VLLM_BIN" >&2
    exit 1
fi

if [ ! -r "$MODEL_PATH/config.json" ]; then
    echo "Model config not found: $MODEL_PATH/config.json" >&2
    exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 59079 \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --seed 1024 \
    --served-model-name qwen3.8 \
    --max-num-seqs 2 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.96 \
    --safetensors-load-strategy eager \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config '{"enable_cpu_binding":true}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
    # --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
