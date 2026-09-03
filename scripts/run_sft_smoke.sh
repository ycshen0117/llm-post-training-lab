#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

exec uv run python scripts/train_sft_tiny.py \
  --config configs/sft_smoke.toml \
  "$@"
