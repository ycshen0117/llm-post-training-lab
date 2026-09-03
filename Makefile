.DEFAULT_GOAL := check

SFT_ARGS ?=
SFT_CHECKPOINT ?= checkpoints/sft-smoke
COMPARE_ARGS ?=

.PHONY: data test lint format check sft-smoke compare

data:
	uv run python scripts/prepare_data.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint test

sft-smoke:
	./scripts/run_sft_smoke.sh $(SFT_ARGS)

compare:
	uv run python scripts/compare_sft_generation.py \
		--sft-checkpoint "$(SFT_CHECKPOINT)" \
		$(COMPARE_ARGS)
