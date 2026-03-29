# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Megatron-AxoNN** integrates AxoNN's 2D/3D tensor parallelism into NVIDIA's Megatron-LM framework for training large language models (1B–1T+ parameters) on supercomputers like NERSC Perlmutter. The key addition over stock Megatron-LM is AxoNN's depth/row/column tensor parallelism axes.

## Running Training

Training runs via SLURM on Perlmutter. The primary entry point is `pretrain_gpt.py`.

**Single job submission:**
```bash
sbatch examples/template_perlmutter.sh
```

**Using the communication model to find optimal parallelism configs:**
```bash
cd examples/
python launch_per_comm_model.py --gpus 64 --batch-size 512 --model 5B [--run]
# Prints top-k parallelism configs ranked by predicted communication time
# Add --run to actually submit the generated sbatch scripts
```

**Key environment variables (Perlmutter-specific):**
```bash
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0   # reverse order on Perlmutter
```

**Run command pattern:**
```bash
srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 \
  shifter ./examples/get_rank_from_slurm.sh python -u pretrain_gpt.py $ARGS
```

## Key Parallelism Arguments

AxoNN adds three new parallelism axes on top of Megatron's pipeline parallelism:

| Argument | Variable | Meaning |
|---|---|---|
| `--column-tensor-model-parallel-size` | `COLUMN_TENSOR_PARR` | Column-wise intra-layer parallelism (Gc) |
| `--row-tensor-model-parallel-size` | `ROW_TENSOR_PARR` | Row-wise intra-layer parallelism (Gr) |
| `--depth-tensor-model-parallel-size` | `DEPTH_TENSOR_PARR` | Depth (weight-sharding) parallelism (Gd) |
| `--pipeline-model-parallel-size` | `PIPE_PARR` | Pipeline stages |

Total GPUs = `COLUMN × ROW × DEPTH × PIPELINE × DATA_PARALLEL`

**Communication overlap flags** (enable for performance):
```
--overlap-axonn-comm
--overlap-axonn-reduce-scatter
--overlap-axonn-all-gather
--num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather $CACHE_LAYERS
--layer-caching-level $CACHE_LEVEL   # 0=none, 1=fwd, 2=fwd+bwd
```

## Testing and Formatting

```bash
# Unit tests (requires 8 GPUs)
torchrun --nproc_per_node=8 -m pytest --cov=megatron/core tests/unit_tests

# Format check
black megatron/core --check --verbose --diff
isort megatron/core --check

# Auto-format
bash tools/autoformat.sh
```

Code style: Black, 100-char line length, isort with black profile (configured in `pyproject.toml`).

## Data Preprocessing

```bash
# GPT data
python tools/preprocess_data.py \
    --input <raw_data.json> \
    --output-prefix $DATA_DIR/BookCorpusDataset \
    --vocab-file $DATA_DIR/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file $DATA_DIR/gpt2-merges.txt \
    --workers 4
```

Expected data files: `$SCRATCH/gpt_data/` with `gpt2-vocab.json`, `gpt2-merges.txt`, and a `*_text_document` dataset.

## Architecture Overview

```
pretrain_gpt.py               # Entry point: sets up distributed init, calls pretrain()
  ├── patch_torch_dist_init   # Patches torch dist init to use TCPStore (faster on large clusters)
  ├── megatron/training.py    # Main training loop (forward, backward, optimizer step)
  ├── megatron/initialize.py  # Megatron + AxoNN initialization, process groups
  ├── megatron/arguments.py   # All CLI arguments including AxoNN-specific ones
  └── megatron/model/
        └── transformer.py    # Core transformer (attention, MLP, fused kernels)
```

**AxoNN integration points in `pretrain_gpt.py`:**
- `drop(tokens, skip_channels=True)` — shards input batch across depth-parallel group
- `optimize_communication(...)` context manager — overlaps compute and communication
- `ForwardAllReduce.apply(loss, group)` — all-reduces loss across depth-parallel group
- `ax.comm_handle.depth_intra_layer_parallel_group` — AxoNN's depth process group

**Process group hierarchy** (outermost → innermost): data parallel → pipeline → depth → row → column

## Communication Model (`examples/comm_model.py`)

`get_configs_for_transformer()` analytically predicts communication time for a given model and GPU count across all valid `(Gc, Gr, Gd, Gdata)` factorizations. Uses empirically measured bandwidth values for Perlmutter and Frontier. Used by `launch_per_comm_model.py` to generate and optionally submit optimally-configured SLURM jobs.

## Profiling

**nsys:**
Set `NSYS_PROFILE=True` in the launch script. Adds `--profile-step-start 5 --profile-step-end 10` args.

**PyTorch profiler:**
Set `TORCH_PROFILE=True`. Requires `--path-for-traces` and `--profile-ranks`.

**AxoNN internal timers** — printed at end of `pretrain_gpt.py`:
```python
times, events = ax_intra_layer.timers.get_times()
```
