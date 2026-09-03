# LLM Post-Training Lab

A learning-oriented research engineering project for building an end-to-end LLM post-training workflow.

Core ML pipeline:

**SFT → GRPO → Evaluation**

## Project Goals

This project combines two learning tracks:

- LLM post-training with SFT, GRPO, and evaluation
- Research engineering with reproducible local and remote workflows

The project introduces tools when they solve a concrete need.

## Local Setup

Create the project environment and install dependencies:

```bash
uv sync
```

Prepare the GSM8K training data:

```bash
make data
```

Run lint and tests:

```bash
make
```

The default Make target is `check`, so the previous command is equivalent to:

```bash
make check
```

## Local SFT Smoke Test

Run the default SFT smoke configuration:

```bash
make sft-smoke
```

The training configuration is stored in:

```text
configs/sft_smoke.toml
```

The default checkpoint is written to:

```text
checkpoints/sft-smoke
```

The resolved configuration for a successful run is recorded in:

```text
checkpoints/sft-smoke/run_config.json
```

Command-line arguments override values from the TOML file:

```bash
make sft-smoke \
  SFT_ARGS="--max-steps 2 --checkpoint-dir checkpoints/sft-trial"
```

Configuration priority is:

```text
built-in defaults < TOML configuration < command-line arguments
```

## Compare Base and SFT Generations

Compare the base model with the default SFT checkpoint:

```bash
make compare
```

Compare another checkpoint and override generation settings:

```bash
make compare \
  SFT_CHECKPOINT=checkpoints/sft-trial \
  COMPARE_ARGS="--max-new-tokens 64"
```

## Development Commands

```bash
make data
make check
make test
make lint
make format
make sft-smoke
make compare
```

Datasets, checkpoints, caches, and generated outputs are kept out of Git. Source code, tests, configuration files, and workflow definitions are tracked.
