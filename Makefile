.PHONY: data test lint format

data:
	uv run python scripts/prepare_data.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
