# AGENTS.md

## Project Purpose

This repository is a learning-oriented LLM post-training capstone project.

Core ML pipeline:

**SFT → GRPO → Evaluation**

The project also serves as a hands-on environment for learning modern AI research engineering tools.

## Development Principles

- Develop primarily on local macOS.
- Use Git/GitHub to synchronize code.
- Run small smoke tests locally before expensive training.
- Run full training on remote GPU servers.
- Use `uv` for Python environments and dependencies.
- Use shell scripts and Makefile for repeatable workflows.
- Use Ruff, pytest, and pre-commit for code quality.
- Keep datasets, checkpoints, caches, and large outputs out of Git.

## Engineering Philosophy

Introduce tools only when they solve a real project need.

Examples:

- Hydra when experiment configuration becomes complex.
- GPU monitoring when remote training begins.
- W&B when multiple experiments need tracking.
- Hugging Face Hub when model artifacts need management.
- Slurm when a real HPC cluster is used.
- Distributed training / DeepSpeed only when required.

Avoid unnecessary infrastructure and premature complexity.

## Teaching Style

When introducing a new tool or workflow:

1. Explain the concept in beginner-friendly terms.
2. Explain why the project needs it.
3. Start with the simplest useful implementation.
4. Explain important commands and files.
5. Add complexity incrementally.

## Current Workflow

```text id="30g8mn"
Local development
      ↓
Test / lint
      ↓
GitHub
      ↓
Remote GPU
      ↓
SFT
      ↓
GRPO
      ↓
Evaluation
```
