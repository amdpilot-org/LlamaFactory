#!/bin/bash
set -e

MODEL_PATH="/data/yikzhang/Qwen3.5-397B-A17B"
OUTPUT_DIR="$HOME/LlamaFactory/saves"
LLAMAFACTORY_DIR="$HOME/LlamaFactory"

mkdir -p "$OUTPUT_DIR"

exec docker run \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=64g \
  --ipc=host \
  --network=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e WANDB_PROJECT=amdpilot-sft \
  -e WANDB_ENTITY=amdpilot \
  -e WANDB_RUN_GROUP=qwen35-397b-lora-v5-1 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "$LLAMAFACTORY_DIR":/workspace/LlamaFactory \
  -v "$MODEL_PATH":/workspace/models \
  -v "$OUTPUT_DIR":/workspace/output \
  -w /workspace/LlamaFactory \
  llamafactory-mi355x:latest \
  bash -c '
    pip install -q "transformers>=5.2.0,<=5.2.0" "tokenizers>=0.22" 2>&1 | tail -3
    torchrun --nproc_per_node=8 --master_port=29500 \
      launch_compat.py \
      examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5_1.yaml
  '
