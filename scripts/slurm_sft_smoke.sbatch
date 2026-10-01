#!/usr/bin/env bash
#SBATCH --job-name=sft-smoke
#SBATCH --partition=general
#SBATCH --qos=protected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -ne 1 || "$1" != /* || ! -d "$1" ]]; then
  echo "Usage: sbatch --output=RUN_DIR/slurm-%j.log scripts/slurm_sft_smoke.sbatch RUN_DIR" >&2
  echo "RUN_DIR must be an existing absolute directory." >&2
  exit 2
fi

: "${SLURM_JOB_ID:?Submit this script with sbatch.}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
RUN_DIR="$1"

# Slurm runs a spooled copy of this file; its location is not the repository.
cd -- "$SLURM_SUBMIT_DIR"
if [[ ! -f scripts/run_sft_smoke.sh || ! -f pyproject.toml ]]; then
  echo "Submit from the llm-post-training-lab repository root." >&2
  exit 2
fi

# Batch shells do not necessarily load the interactive ~/.bashrc settings.
export HF_HOME="/net/scratch/$USER/hf-cache"
export TMPDIR="/net/scratch/$USER/tmp"
export UV_CACHE_DIR="/net/scratch/$USER/uv-cache"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$TMPDIR" "$UV_CACHE_DIR"

DATASET_PATH="/net/scratch/$USER/datasets/gsm8k/train"
if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Prepared dataset not found: $DATASET_PATH" >&2
  echo 'On a compute node, run: make data DATA_ARGS="--output-dir /net/scratch/$USER/datasets/gsm8k"' >&2
  exit 2
fi

printf 'Job ID: %s\nNode: %s\nRepository: %s\nCommit: %s\nRun directory: %s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$(git rev-parse --short HEAD)" "$RUN_DIR"
date -u '+Started: %Y-%m-%dT%H:%M:%SZ'
nvidia-smi

# Use the same reference model for training and the generation comparison.
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
bash scripts/run_sft_smoke.sh \
  --model-id "$MODEL_ID" \
  --device cuda \
  --precision bf16 \
  --dataset-path "$DATASET_PATH" \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-train-examples 128 \
  --max-steps 20 \
  --max-length 512 \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  2>&1 | tee "$RUN_DIR/train.log"

# Reuse the allocated GPU to check that the saved checkpoint can generate.
uv run python scripts/compare_sft_generation.py \
  --base-model-id "$MODEL_ID" \
  --sft-checkpoint "$RUN_DIR/checkpoint" \
  --max-new-tokens 64 \
  2>&1 | tee "$RUN_DIR/compare.log"

date -u '+Completed: %Y-%m-%dT%H:%M:%SZ'
