# LLM Post-Training Lab Curriculum

Core ML pipeline:

**SFT → GRPO → Evaluation**

Engineering principle:

> Introduce tools when the project first creates a real need for them.

---

## Milestone 1 — Project Foundation

- [x] Initialize Git repository
- [x] Create `README.md`, `AGENTS.md`, and curriculum
- [x] Initialize project with `uv`
- [x] Understand `pyproject.toml`
- [x] Create `src/` and `tests/`
- [x] Configure `.gitignore`
- [x] Add Ruff
- [x] Add pytest
- [ ] Add pre-commit
- [ ] Create first clean commit

---

## Milestone 2 — Data Pipeline

- [ ] Select a small instruction dataset
- [ ] Load data with Hugging Face Datasets
- [ ] Inspect JSON / JSONL with `jq`, `rg`, `head`, `wc`
- [ ] Implement preprocessing
- [ ] Format data for SFT
- [ ] Add preprocessing tests
- [ ] Create a reproducible data-preparation command

---

## Milestone 3 — Local SFT

- [ ] Select a small causal language model
- [ ] Load model and tokenizer
- [ ] Implement SFT training
- [ ] Run a tiny local smoke test
- [ ] Save checkpoints
- [ ] Add a simple generation test

---

## Milestone 4 — Automation and Configuration

- [ ] Add shell training scripts
- [ ] Add Makefile targets
- [ ] Separate experiment configuration from code
- [ ] Introduce Hydra when configuration becomes complex
- [ ] Support command-line overrides

Example workflow:

```bash id="r9a1mz"
make data
make smoke
make sft
make eval
```

---

## Milestone 5 — Remote GPU Workflow

- [ ] Connect to a GPU server with SSH
- [ ] Synchronize code through Git
- [ ] Reproduce the environment remotely
- [ ] Run training inside tmux
- [ ] Manage datasets and checkpoints on persistent storage
- [ ] Learn GPU/Linux diagnostics:
  - `nvidia-smi`
  - `ps`
  - `top` / `htop`
  - `free -h`
  - `df -h`
  - `du -sh`
  - `tail -f`

---

## Milestone 6 — Experiment Tracking

Introduce W&B once multiple experiments need to be compared.

- [ ] Log training metrics
- [ ] Log configs and seeds
- [ ] Track checkpoints
- [ ] Compare SFT runs
- [ ] Record Git commit information

---

## Milestone 7 — Docker

- [ ] Write a Dockerfile
- [ ] Build the training image
- [ ] Run smoke tests in Docker
- [ ] Mount datasets and checkpoints
- [ ] Run GPU containers
- [ ] Keep persistent data outside containers

---

## Milestone 8 — Hugging Face Hub

Introduce the Hub once the project produces useful model artifacts.

- [ ] Use the `hf` CLI
- [ ] Understand model cache
- [ ] Upload an SFT checkpoint
- [ ] Create a model card
- [ ] Understand revisions and Git LFS

---

## Milestone 9 — GRPO

- [ ] Choose a task with automatic rewards
- [ ] Implement reward functions
- [ ] Test reward functions
- [ ] Generate multiple responses per prompt
- [ ] Understand group-relative rewards
- [ ] Run a GRPO smoke test
- [ ] Train from the SFT checkpoint
- [ ] Track GRPO experiments with W&B

If needed here:

- distributed training
- Accelerate
- DeepSpeed

---

## Milestone 10 — Evaluation

Compare:

**Base vs SFT vs GRPO**

- [ ] Define evaluation tasks
- [ ] Define metrics
- [ ] Evaluate all three models
- [ ] Save structured results
- [ ] Analyze results with Python and `jq`
- [ ] Inspect failure cases
- [ ] Produce a final comparison

---

## Milestone 11 — CI

Introduce GitHub Actions once the repo becomes complex enough to break accidentally.

- [ ] Run Ruff automatically
- [ ] Run pytest automatically
- [ ] Add lightweight smoke tests if practical

---

## Milestone 12 — HPC / Slurm

Introduce only when a real Slurm cluster is available.

- [ ] Write Slurm job scripts
- [ ] Request GPU / CPU / memory
- [ ] Submit and monitor jobs
- [ ] Run SFT and GRPO
- [ ] Use job arrays when useful

---

## Milestone 13 — Agentic Coding Workflow

- [ ] Use Codex / Claude Code to inspect the repo
- [ ] Give scoped implementation tasks
- [ ] Require tests and lint
- [ ] Review Git diffs before accepting changes
- [ ] Maintain project rules in `AGENTS.md`

---

## Final Workflow

```text id="k6ef41"
Local development
      ↓
Tests / Ruff
      ↓
GitHub
      ↓
Remote GPU / HPC
      ↓
SFT
      ↓
GRPO
      ↓
Evaluation
      ↓
W&B / Hugging Face artifacts
```
