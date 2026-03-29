# Replacing `torch.distributed` with `torchcomms.distwrap` in Megatron-AxoNN

## What is distwrap?

`torchcomms.distwrap` is a drop-in replacement API for `torch.distributed`.
It routes collective operations through torchcomms (backed by ncclx) instead of
native torch.distributed when `use_torchcomms=True` is passed to `init_process_group`.

- Source: `torchcomms-sparse/comms/torchcomms/distwrap/`
- Full docs: `torchcomms-sparse/comms/torchcomms/distwrap/README.md`

---

## Change 1: Import swap (all ~48 files)

Anywhere `torch.distributed` is used as `dist`, replace the import.

**Before:**
```python
import torch.distributed as dist
```
**After:**
```python
from torchcomms import distwrap as dist
```

For files that use `torch.distributed.xxx` directly (via `import torch`), either
add the import above alongside `import torch`, or replace `torch.distributed.`
calls with `dist.`.

Files that need this (non-test):
- `megatron/initialize.py` (uses `torch.distributed.init_process_group` at line 198)
- `megatron/core/parallel_state.py` (uses `torch.distributed.new_group` at lines 178, 212, 223, 242, 264, 270, 286)
- `megatron/training.py`, `megatron/utils.py`, `megatron/timers.py`, `megatron/checkpointing.py`
- `megatron/core/tensor_parallel/mappings.py`, `layers.py`, `data.py`, `cross_entropy.py`, `utils.py`
- `megatron/core/pipeline_parallel/p2p_communication.py`
- `megatron/model/distributed.py`, `module.py`
- `megatron/optimizer/distrib_optimizer.py`, `optimizer.py`, `clip_grads.py`
- (and others — run `grep -rl "torch\.distributed" megatron/ --include="*.py"` for full list)

One-liner to check all affected files:
```bash
grep -rl "torch\.distributed" megatron/ --include="*.py" | grep -v __pycache__ | grep -v test
```

---

## Change 2: Enable torchcomms at init (`megatron/initialize.py:198`)

**Before:**
```python
torch.distributed.init_process_group(
    backend=args.distributed_backend,
    world_size=args.world_size,
    rank=args.rank,
    timeout=timedelta(minutes=args.distributed_timeout_minutes),
)
```
**After:**
```python
dist.init_process_group(
    backend=args.distributed_backend,
    world_size=args.world_size,
    rank=args.rank,
    timeout=timedelta(minutes=args.distributed_timeout_minutes),
    use_torchcomms=True,
)
```

The `use_torchcomms=True` flag enables torchcomms routing. Without it, distwrap
is a pure passthrough to torch.distributed (no behavior change).

See: `torchcomms-sparse/comms/torchcomms/distwrap/new_comm.py`

---

## Change 3: `new_group` → `split_group` (`megatron/core/parallel_state.py`)

**This is the only non-mechanical change.**

When `use_torchcomms=True`, `dist.new_group()` raises `AssertionError`.
You must use `dist.split_group(split_ranks=<all_partitions>)` instead.

`split_group` takes the **full partition** (list of all rank-lists) rather than
just the ranks for your group. Every rank in the world must call it with the
same `split_ranks` argument.

See: `torchcomms-sparse/comms/torchcomms/distwrap/new_comm.py` — `split_group()`
See: `torchcomms-sparse/comms/torchcomms/distwrap/README.md` lines 107–115

### Pattern

**Before (parallel_state.py pattern):**
```python
all_groups = []
for ...:
    ranks = [...]
    all_groups.append(list(ranks))
    group = torch.distributed.new_group(ranks)   # called once per partition
    if rank in ranks:
        _MY_GROUP = group
```

**After:**
```python
all_groups = []
for ...:
    ranks = [...]
    all_groups.append(list(ranks))
# Build all_groups first, then split once
groups = dist.split_group(split_ranks=all_groups)
for ranks, group in zip(all_groups, groups):
    if rank in ranks:
        _MY_GROUP = group
```

> Note: `split_group` returns a list of ProcessGroups, one per partition.
> The calling rank gets a valid group object for *all* partitions, but only
> the one containing it is meaningful for collectives.

### Specific locations in `megatron/core/parallel_state.py`

| Line | Group | Notes |
|------|-------|-------|
| 178 | `_DATA_PARALLEL_GROUP` | `all_data_parallel_group_ranks` already built above — pass it directly |
| 212 | `_MODEL_PARALLEL_GROUP` | ranks built in loop from `all_data_parallel_group_ranks` — collect first |
| 223 | `_TENSOR_MODEL_PARALLEL_GROUP` | simple range-based partition |
| 242 | `_PIPELINE_MODEL_PARALLEL_GROUP` | range-based partition |
| 264 | `_EMBEDDING_GROUP` | subset of pipeline ranks — collect all embedding_ranks lists first |
| 270 | `_POSITION_EMBEDDING_GROUP` | subset of pipeline ranks — collect all position_embedding_ranks lists first |
| 286 | `_AMAX_REDUCTION_GROUP` | FP8 only, simple range partition |

### `two_stage.py:147` — skip or leave as-is

`megatron/core/dist_checkpointing/strategies/two_stage.py:147` creates a gloo
group for checkpointing. Leave this as `torch.distributed.new_group(..., backend='gloo')`
(direct call bypassing distwrap) since it's checkpointing-only and torchcomms
doesn't need to manage it.

---

## Limitations to be aware of

- **`new_group()` is not supported** when torchcomms is enabled — use `split_group()`.
- **Wildcard `recv(src=None)`** is not supported — must pass explicit `src`.
- **Direct `torch.distributed` calls are blocked** once torchcomms is enabled —
  all collectives must go through `distwrap`. This means if any library you use
  internally calls `torch.distributed.all_reduce` etc., it will raise
  `NotImplementedError`.

---

## Verification

After migration, a quick sanity check:
```python
from torchcomms import distwrap as dist
print(dist.is_initialized())   # should work like torch.distributed
print(dist.get_rank())
print(dist.get_world_size())
```
