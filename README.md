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

## DSI GPU Smoke Test

Prepare data on a compute node and save it to scratch:

```bash
make data DATA_ARGS="--output-dir /net/scratch/$USER/datasets/gsm8k"
```

The training split is saved under the output directory's `train` subdirectory.
Hugging Face's download cache is controlled separately by `HF_HOME`.
Without `DATA_ARGS`, `make data` retains the local default at
`data/processed/gsm8k/train`.

After obtaining a Slurm GPU allocation, run a five-step smoke test:

```bash
RUN_DIR="/net/scratch/$USER/runs/sft-gpu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail

make sft-smoke \
  SFT_ARGS="--device cuda --precision bf16 --dataset-path /net/scratch/$USER/datasets/gsm8k/train --batch-size 1 --num-train-examples 32 --max-steps 5 --max-length 512 --checkpoint-dir $RUN_DIR/checkpoint" \
  2>&1 | tee "$RUN_DIR/train.log"
```

Explicit CUDA requests fail if CUDA is unavailable. BF16 requires a supported
CUDA GPU. Model parameters stay in FP32 while autocast controls the precision
of forward computations. Use `--precision fp32` if BF16 is unsupported.

Scratch stores reproducible data and active experiment outputs. Copy important
results to persistent storage for long-term retention.

## Gradient Accumulation and Training Metrics

`--batch-size` controls the microbatch processed by each forward/backward pass.
`--gradient-accumulation-steps` controls how many microbatches contribute to one
optimizer update. On a single GPU, the nominal effective batch size is their
product. Parameters are updated only after the accumulation window finishes.

For example, batch size 1 with accumulation 4 processes four examples separately
before updating parameters. Loss contributions are weighted by the number of
supervised next-token targets so variable-length answers receive the same token
weighting as a combined batch. A final incomplete window is also updated using
its actual token count.

`--max-steps` counts optimizer updates. The script still makes only one pass
through the selected dataset and reports if it will end before the requested
number of updates. With 128 examples, batch size 1, accumulation 4, and 20 updates,
training consumes 80 examples in 80 microbatches.

After obtaining a GPU allocation:

```bash
RUN_DIR="/net/scratch/$USER/runs/sft-accum-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail

make sft-smoke \
  SFT_ARGS="--device cuda --precision bf16 --dataset-path /net/scratch/$USER/datasets/gsm8k/train --batch-size 1 --gradient-accumulation-steps 4 --num-train-examples 128 --max-steps 20 --max-length 512 --checkpoint-dir $RUN_DIR/checkpoint" \
  2>&1 | tee "$RUN_DIR/train.log"
```

Each successful checkpoint includes `train_metrics.json` with actual update,
microbatch, example, and supervised-token counts; loop duration; throughput;
and peak CUDA memory. Timing includes data collation and logging inside the loop,
but excludes model loading and checkpoint saving. CUDA synchronization brackets
the timed loop. Peak allocated and reserved memory measure this process's
PyTorch allocator, not total GPU usage; non-CUDA memory values are `null`.

## DSI Slurm Batch Smoke Test

Submit from the repository root on a login node. The job requests one GPU,
four CPU cores, 16 GiB of host RAM (not GPU memory), and 30 minutes. It uses
DSI's `protected` QoS for this short smoke test: one job at a time, no scheduler
preemption, and a higher fairshare cost than `general`. See the
[DSI batch job guide](https://cluster-policy.ds.uchicago.edu/using-the-cluster/batch-jobs/).

The prepared dataset must already exist at
`/net/scratch/$USER/datasets/gsm8k/train`. The script explicitly sets the cache
and temporary directories because batch shells may not read `~/.bashrc`.
It trains with batch size 1 and accumulation 4 for 20 optimizer updates, then
runs the existing base/SFT generation comparison on the allocated GPU.

```bash
cd ~/Developer/llm-post-training-lab
RUN_DIR="/net/scratch/$USER/runs/sft-batch-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"

sbatch --output="$RUN_DIR/slurm-%j.log" \
  scripts/slurm_sft_smoke.sbatch "$RUN_DIR"
```

Create the output directory before submitting: Slurm opens its log before the
script runs. `%j` is replaced by the job ID. Pass the output path on the command
line because shell variables are not expanded inside `#SBATCH` directives.
The script uses `SLURM_SUBMIT_DIR` to find the repository; Slurm executes a
spooled script copy whose own directory is not the checkout.

Submission prints `Submitted batch job JOB_ID`; this confirms submission, not
completion. Replace `12345` below with that ID:

```bash
JOB_ID=12345
squeue -j "$JOB_ID"
tail -n 50 "$RUN_DIR/slurm-$JOB_ID.log"
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed
```

`PD` means pending and `R` means running. The log may not exist while pending.
After the job leaves the queue, check `sacct`: success is `COMPLETED` with exit
code `0:0`. Training or comparison failures propagate through `tee` because
the script enables `pipefail`. The job can continue after the submitting SSH
connection closes. To cancel it, use `scancel "$JOB_ID"`.

Outputs are `slurm-JOB_ID.log`, `train.log`, `compare.log`, and `checkpoint/`
under the run directory. Inspect `checkpoint/train_metrics.json` for 20
optimizer updates, 80 microbatches, and 80 examples. The two fixed comparison
prompts check checkpoint loading and generation; they are not a quality score.
Slurm captures this batch script, but not a snapshot of the repository: leave
the checkout unchanged until the job finishes.

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
