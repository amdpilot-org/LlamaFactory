# Qwen3.5-397B LoRA SFT Training — Complete Agent Guide

> **What is this?** A self-contained guide for an agent on a new MI355X node to
> run LoRA SFT fine-tuning of Qwen3.5-397B-A17B using the AMDPilot dataset (v5
> and v5.1). This document contains every command, every config file inline, and
> every decision rationale. No prior context is needed.

---

## Table of Contents

1. [Background](#1-background)
2. [Environment & Prerequisites](#2-environment--prerequisites)
3. [Step-by-Step Execution](#3-step-by-step-execution)
4. [All Config Files (Inline)](#4-all-config-files-inline)
5. [Troubleshooting](#5-troubleshooting)
6. [Autoresearch Experiments (Optional)](#6-autoresearch-experiments-optional)
7. [Hard Constraints & Lessons Learned](#7-hard-constraints--lessons-learned)

---

## 1. Background

### What are we doing?

Fine-tuning **Qwen3.5-397B-A17B** (a Mixture-of-Experts model with 397B total
params, 17B active per token) using **LoRA** (Low-Rank Adaptation) on AMD GPU
debugging and kernel engineering task data.

### Why?

The AMDPilot project trains LLMs to solve real AMD GPU software issues —
debugging ROCm kernel crashes, optimizing MoE inference, fixing attention
backends, etc. Each training example is a multi-turn conversation where an agent
diagnoses and fixes a real GPU issue.

### Previous training (v4 baseline)

| Metric | Value |
|--------|-------|
| Dataset | v4: 270 train / 10 eval |
| Context window | 32768 tokens (32k) |
| LoRA rank / alpha | 32 / 64 |
| Epochs | 10 |
| Final train_loss | 0.199 |
| Final eval_loss | 0.055 |
| Runtime | ~8 hours on 8x MI355X |
| Hardware | AMD Instinct MI355X, 8 GPUs |

### What's new for v5/v5.1?

- **v5 dataset**: 42 train / 2 eval examples (curated from Arist pipeline)
- **v5.1 dataset**: 89 train / 3 eval examples (v5 superset + gpumode contributions)
- **64k context window** (up from 32k) — many examples were being truncated at 32k
- **Liger kernel** enabled for ~30% training speedup
- **neat_packing** enabled — fixes packing behavior with mRoPE models like Qwen3.5
- Two separate training runs producing two separate LoRA adapters

### Dataset analysis

| Dataset | Train | Eval | Examples >32k tokens (est.) | Examples >64k tokens (est.) |
|---------|-------|------|-------|-------|
| v5 | 42 | 2 | 8 (19%) | 2 (5%) |
| v5.1 | 89 | 3 | 29 (33%) | 2 (2%) |

The model natively supports 256k tokens (`max_position_embeddings: 262144`), so
64k requires no RoPE scaling.

### Datasets on Hugging Face

Consolidated at: https://huggingface.co/datasets/JinnP/amdpilot-lora-sft-dataset

```python
from datasets import load_dataset
ds = load_dataset("JinnP/amdpilot-lora-sft-dataset", "v5")      # 42 train
ds = load_dataset("JinnP/amdpilot-lora-sft-dataset", "v5_1")    # 89 train
```

Available configs: `v4`, `v5`, `v5_full`, `v5_combined`, `v5_chunks`, `v5_1`,
`v5_1_full`, `v5_1_combined`, `v5_1_chunks`.

---

## 2. Environment & Prerequisites

### Hardware

- **GPU**: 8x AMD Instinct MI355X (256GB HBM each, 2TB total)
- **ROCm**: Must be installed and working (`rocm-smi` should list 8 GPUs)
- **Docker**: Must be installed with ROCm device support

### Paths on this node

| What | Path |
|------|------|
| Base model weights | `/data/yikzhang/Qwen3.5-397B-A17B` |
| Repo clone target | `~/LlamaFactory` |
| Training output | `~/LlamaFactory/saves` |

### Credentials

Set these environment variables before running any training or upload commands:

```bash
export WANDB_API_KEY="<obtain-from-team>"
export HF_TOKEN="<obtain-from-team>"
```

These are read by the launch scripts and passed into the Docker container. **Do
not hardcode them into any files that get committed to git.**

---

## 3. Step-by-Step Execution

### Step 1: Verify GPU Access

```bash
rocm-smi
```

**Expected**: 8 GPUs listed. If this fails, ROCm drivers are not properly
installed. Do not proceed until this works.

### Step 2: Verify Model Weights

```bash
ls /data/yikzhang/Qwen3.5-397B-A17B/config.json
```

**Expected**: The file exists. If not, the path may be a HuggingFace cache layout:

```bash
# If it's a HF cache, find the actual model directory:
find /data/yikzhang/Qwen3.5-397B-A17B -name "config.json" -type f 2>/dev/null
```

If `config.json` is at a deeper path (e.g. `snapshots/<hash>/config.json`), you
must update `MODEL_PATH` in both launch scripts to point to the directory
containing `config.json`.

### Step 3: Clone the Repository

```bash
cd ~
git clone https://github.com/amdpilot-org/LlamaFactory.git
cd LlamaFactory
git checkout feat/qwen35-lora-sft-amdpilot
```

**Verify**:

```bash
# These should all exist:
ls Dockerfile.mi355x
ls run_train_mi355x_v5.sh run_train_mi355x_v5_1.sh
ls examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5.yaml
ls examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5_1.yaml
ls data/amdpilot_v5_train.jsonl data/amdpilot_v5_1_train.jsonl
```

### Step 4: (If needed) Adjust MODEL_PATH

If Step 2 revealed the model is inside a subdirectory, edit both launch scripts:

```bash
# In run_train_mi355x_v5.sh AND run_train_mi355x_v5_1.sh, change:
MODEL_PATH="/data/yikzhang/Qwen3.5-397B-A17B"
# to the actual path containing config.json, e.g.:
MODEL_PATH="/data/yikzhang/Qwen3.5-397B-A17B/snapshots/<hash>"
```

### Step 5: Build Docker Image

```bash
cd ~/LlamaFactory
docker build -f Dockerfile.mi355x -t llamafactory-mi355x:latest .
```

**Expected**: Build completes in 5-10 minutes.

**Verify**:

```bash
docker run --rm llamafactory-mi355x:latest python -c "import llamafactory; print('OK')"
```

Should print `OK`.

### Step 6: Verify Model is Accessible from Container

```bash
docker run --rm \
  -v /data/yikzhang/Qwen3.5-397B-A17B:/workspace/models \
  llamafactory-mi355x:latest \
  ls /workspace/models/config.json
```

**Expected**: `/workspace/models/config.json` listed. If not, go back to Step 4.

### Step 7: Run v5 Training

```bash
cd ~/LlamaFactory
bash run_train_mi355x_v5.sh
```

**What this does**:
1. Launches a Docker container with all 8 MI355X GPUs
2. Installs `transformers>=5.2.0` (required for Qwen3.5, not in base image)
3. Runs `torchrun` with 8 processes, each on one GPU
4. Uses `launch_compat.py` which applies Python 3.10 compatibility patches
5. Trains using the v5 YAML config with DeepSpeed ZeRO-3

**Config summary (v5)**:

| Parameter | Value | Why |
|-----------|-------|-----|
| Dataset | `amdpilot_v5` (42 train, 2 eval) | Curated Arist pipeline data |
| Context window | 65536 (64k) | Captures ~95% of examples without truncation |
| LoRA rank / alpha | 32 / 64 | Same as proven v4 config |
| LoRA target | all (13 module types) | Includes MoE expert gates |
| Epochs | 20 | More epochs because fewer examples (42 vs v4's 270) |
| Batch size | 1 per device, grad_accum=2 | Memory-constrained at 64k |
| Learning rate | 2e-5, cosine | Same as v4 |
| Precision | bf16 | Required for stability |
| Packing | neat_packing=true | Efficient variable-length training |
| Liger kernel | enabled | ~30% speedup |
| DeepSpeed | ZeRO-3 | Required to fit 397B model |
| Eval | every 10 steps | |
| Checkpoints | every 20 steps | |
| wandb project | `amdpilot-sft` | |
| wandb run group | `qwen35-397b-lora-v5` | |
| wandb run name | `qwen35-397b-lora-sft-amdpilot-v5` | |

**Monitor**: Go to https://wandb.ai/amdpilot/amdpilot-sft and look for the run.
Key metrics: `train/loss`, `eval/loss`.

**Expected runtime**: ~4-8 hours.

**Output**: `~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5/`

### Step 8: Upload v5 Adapter to Hugging Face

After training completes:

```bash
pip install huggingface-hub
huggingface-cli login --token "$HF_TOKEN"

huggingface-cli upload \
  JinnP/qwen35-397b-lora-sft-amdpilot-v5 \
  ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5/ \
  . \
  --repo-type model
```

**Verify**: https://huggingface.co/JinnP/qwen35-397b-lora-sft-amdpilot-v5

The adapter should contain: `adapter_config.json`, `adapter_model.safetensors`,
tokenizer files, `train_results.json`, `eval_results.json`.

### Step 9: Run v5.1 Training

```bash
cd ~/LlamaFactory
bash run_train_mi355x_v5_1.sh
```

**Config differences from v5**:

| Parameter | v5 | v5.1 |
|-----------|-----|------|
| Dataset | `amdpilot_v5` (42 train) | `amdpilot_v5_1` (89 train) |
| Epochs | 20 | 10 |
| wandb run group | `qwen35-397b-lora-v5` | `qwen35-397b-lora-v5-1` |
| wandb run name | `...-v5` | `...-v5-1` |
| Output dir | `...-v5` | `...-v5-1` |

Everything else (context window, LoRA config, learning rate, etc.) is identical.

**Expected runtime**: ~4-8 hours.

### Step 10: Upload v5.1 Adapter to Hugging Face

```bash
huggingface-cli upload \
  JinnP/qwen35-397b-lora-sft-amdpilot-v5-1 \
  ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5-1/ \
  . \
  --repo-type model
```

**Verify**: https://huggingface.co/JinnP/qwen35-397b-lora-sft-amdpilot-v5-1

### Step 11: Record Final Results

```bash
echo "=== v5 results ==="
cat ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5/train_results.json
cat ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5/eval_results.json

echo "=== v5.1 results ==="
cat ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5-1/train_results.json
cat ~/LlamaFactory/saves/qwen35-397b-lora-sft-amdpilot-v5-1/eval_results.json
```

**Compare against v4 baseline**: train_loss=0.199, eval_loss=0.055.

---

## 4. All Config Files (Inline)

These files are already in the repo after cloning. They are included here for
reference so you can understand and debug without needing to open files.

### 4.1 Dockerfile.mi355x

```dockerfile
FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315

WORKDIR /workspace/LlamaFactory
COPY . /workspace/LlamaFactory/

RUN sed -i 's/requires-python = ">=3.11.0"/requires-python = ">=3.10.0"/' pyproject.toml && \
    pip install -e ".[metrics]" && \
    pip install deepspeed wandb && \
    git checkout pyproject.toml 2>/dev/null || true

ENV HF_HOME=/workspace/hf_cache
ENV WANDB_DIR=/workspace/output
```

**Note**: The base image has Python 3.10, but LlamaFactory requires 3.11. The
Dockerfile patches `pyproject.toml` to relax this, and `py310_compat.py` adds
the missing stdlib features at runtime.

### 4.2 run_train_mi355x_v5.sh

```bash
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
  -e WANDB_RUN_GROUP=qwen35-397b-lora-v5 \
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
      examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5.yaml
  '
```

**v5.1 variant** (`run_train_mi355x_v5_1.sh`): Identical except:
- `WANDB_RUN_GROUP=qwen35-397b-lora-v5-1`
- YAML path: `examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5_1.yaml`

### 4.3 Training YAML: v5

```yaml
### model
model_name_or_path: /workspace/models
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_target: all
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: amdpilot_v5
template: qwen3_5_nothink
cutoff_len: 65536
preprocessing_num_workers: 16
dataloader_num_workers: 4
packing: true
neat_packing: true

### output
output_dir: /workspace/output/qwen35-397b-lora-sft-amdpilot-v5
logging_steps: 1
save_steps: 20
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb
run_name: qwen35-397b-lora-sft-amdpilot-v5

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 2.0e-5
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
enable_liger_kernel: true

### eval
eval_dataset: amdpilot_v5_eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 10
```

### 4.4 Training YAML: v5.1

Same as v5 except:

```yaml
dataset: amdpilot_v5_1
output_dir: /workspace/output/qwen35-397b-lora-sft-amdpilot-v5-1
run_name: qwen35-397b-lora-sft-amdpilot-v5-1
num_train_epochs: 10.0
eval_dataset: amdpilot_v5_1_eval
```

### 4.5 DeepSpeed ZeRO-3 Config

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 4.6 launch_compat.py

```python
#!/usr/bin/env python3
"""Launcher wrapper that applies Python 3.10 compat shim before calling LLaMA-Factory."""

exec(open("/workspace/LlamaFactory/py310_compat.py").read())

from llamafactory.train.tuner import run_exp

run_exp()
```

### 4.7 py310_compat.py

```python
"""Python 3.10 compatibility shim for LLaMA-Factory (requires 3.11+).

Patches typing.Self, typing.NotRequired, and enum.StrEnum into stdlib
so LLaMA-Factory code works on Python 3.10 with typing_extensions.
"""

import sys

if sys.version_info < (3, 11):
    import typing
    import enum
    from typing_extensions import Self, NotRequired

    if not hasattr(typing, "Self"):
        typing.Self = Self
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = NotRequired
    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            pass
        enum.StrEnum = StrEnum
```

### 4.8 Docker Volume Mapping

```
Host                                        Container                     Purpose
─────────────────────────────────────────── ───────────────────────────── ────────────────────
~/LlamaFactory                              /workspace/LlamaFactory       Code + datasets
/data/yikzhang/Qwen3.5-397B-A17B           /workspace/models             Base model weights
~/LlamaFactory/saves                        /workspace/output             Checkpoints + wandb
```

The YAML configs reference `/workspace/models` and `/workspace/output` — these
are container paths, not host paths.

---

## 5. Troubleshooting

### OOM (Out of Memory)

If training crashes with HIP/CUDA OOM, try these in order:

1. **Increase gradient accumulation** (edit YAML):
   ```yaml
   gradient_accumulation_steps: 4  # was 2
   ```

2. **Reduce context window** (edit YAML):
   ```yaml
   cutoff_len: 49152  # 48k, was 65536
   ```

3. **Last resort — match v4**:
   ```yaml
   cutoff_len: 32768  # 32k, same as proven v4
   gradient_accumulation_steps: 1
   ```

### Slow Training (>5 min per step)

Check these are set in the YAML:
- `enable_liger_kernel: true`
- `neat_packing: true`
- `dataloader_num_workers: 4`

### wandb Not Logging

```bash
# Check env is passed through to the container:
docker inspect <container_id> | grep WANDB_API_KEY
```

If empty, `WANDB_API_KEY` was not set in your shell before launching.

### Docker Build Fails

If `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315` is not available:

```bash
docker pull rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315
```

If this fails, the image may not be in the public registry. Check if it's
available locally or in a private registry on the cluster.

### Model Path Issues

If you see `FileNotFoundError` for `config.json` or tokenizer files during
training startup, the model path mapping is wrong. Run:

```bash
docker run --rm \
  -v /data/yikzhang/Qwen3.5-397B-A17B:/workspace/models \
  llamafactory-mi355x:latest \
  find /workspace/models -name "config.json" -type f
```

Update `MODEL_PATH` in the launch scripts to the parent directory of the found
`config.json`.

### DeepSpeed Timeout

If you see `NCCL timeout` or `DDP timeout` errors, the current
`ddp_timeout: 180000000` (50 hours) should prevent this. If it still occurs,
it likely indicates a GPU communication issue — check `rocm-smi` for
unhealthy GPUs.

---

## 6. Autoresearch Experiments (Optional)

After the baseline v5 and v5.1 runs complete, run these experiments to optimize.
**Each experiment changes exactly ONE variable from the baseline.**

### Experiment Matrix

| # | Variable | Baseline | Experiment | Expected Impact |
|---|----------|----------|------------|-----------------|
| 1 | Dataset view | v5_1 (=full) | v5_1_chunks | Shorter sequences, may train faster |
| 2 | Context window | 65536 | 32768 | Less memory, faster, but truncates 33% of v5.1 |
| 3 | neat_packing | true | false | Test if neat_packing matters for mRoPE models |
| 4 | LoRA rank | 32 | 64 | More capacity, more memory |
| 5 | LoRA rank | 32 | 16 | Less capacity, less memory, faster |
| 6 | Learning rate | 2e-5 | 1e-5 | More conservative, may reduce overfitting |
| 7 | Learning rate | 2e-5 | 5e-5 | More aggressive, may improve with small data |

### How to Run an Experiment

```bash
# 1. Copy baseline config
cp examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5_1.yaml \
   examples/train_lora/qwen35_397b_lora_sft_amdpilot_v5_1_exp_NAME.yaml

# 2. Edit the ONE variable being tested

# 3. Change output_dir and run_name to include experiment name, e.g.:
#    output_dir: /workspace/output/qwen35-397b-lora-sft-amdpilot-v5-1-exp-NAME
#    run_name: qwen35-397b-lora-sft-amdpilot-v5-1-exp-NAME

# 4. Copy and edit the launch script
cp run_train_mi355x_v5_1.sh run_train_mi355x_v5_1_exp_NAME.sh
# Change WANDB_RUN_GROUP and YAML path

# 5. Run
bash run_train_mi355x_v5_1_exp_NAME.sh

# 6. Compare eval_loss from eval_results.json against baseline
```

### Decision Rule

- **eval_loss improved** and training completed: **KEEP**
- **eval_loss same or worse**: **DISCARD** (note the result for future reference)
- **OOM or crash**: Reduce memory (see troubleshooting) and retry once

---

## 7. Hard Constraints & Lessons Learned

These are non-negotiable rules learned from previous training runs:

| # | Constraint | Reason |
|---|-----------|--------|
| 1 | Template must be `qwen3_5_nothink` | `qwen3_5` enables thinking/reasoning mode which is wrong for tool-use SFT |
| 2 | `lora_target: all` | Must target all 13 module types including MoE gates (`shared_expert_gate`, `gate_proj`) |
| 3 | `packing: true` | Essential for efficiency — examples range from 2k to 350k chars |
| 4 | DeepSpeed ZeRO-3 required | 397B model cannot fit on 8x MI355X without parameter sharding |
| 5 | `gradient_checkpointing: true` | Required for memory at any context length |
| 6 | `transformers>=5.2.0` installed at runtime | Not baked into Docker image — the ROCm base image has an older version |
| 7 | `ddp_timeout: 180000000` | Prevents timeout during slow DS ZeRO-3 all-gather on large model |
| 8 | `save_only_model: true` | Saves only adapter weights, not optimizer states (reduces disk 10x) |
| 9 | `bf16: true` | fp16 is unstable with this model; bf16 is required |
| 10 | Never commit API keys to git | GitHub push protection will reject the push |

### LoRA Target Modules (13 total)

When `lora_target: all` is set, LlamaFactory applies LoRA to these modules:

```
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj,
in_proj_a, in_proj_b, in_proj_z, in_proj_qkv,
out_proj, shared_expert_gate
```

### Model Architecture

- **Type**: `qwen3_5_moe` (Mixture of Experts)
- **Total params**: 397B
- **Active params per token**: 17B
- **Max position embeddings**: 262,144 (256k)
- **RoPE**: mrope (multimodal rotary), `rope_theta: 10,000,000`, interleaved
- **Partial rotary factor**: 0.25
