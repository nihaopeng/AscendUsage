#!/bin/sh
set -eu

# 单张物理 NPU 6
export ASCEND_RT_VISIBLE_DEVICES=7

export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

MODEL_PATH=/home/pengyt/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-14B/snapshots/f2ad5164aade432d6d56c24bb71589184d5d613d

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
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name qwen2.5-coder-14b \
    --max-num-seqs 4 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --safetensors-load-strategy eager \
    --generation-config vllm \
    --additional-config '{"enable_cpu_binding":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
