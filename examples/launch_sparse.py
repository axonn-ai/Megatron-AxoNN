"""
Launch gradient-sparsity experiments on Perlmutter.

Two parallelism modes (AxoNN depth = FSDP-equivalent weight sharding):

  fsdp     Gc=1, Gr=1, Gd=4*N   Pure depth across all GPUs.
                                  All collectives cross nodes (Slingshot).

  fsdp_tp  Gc=2, Gr=2, Gd=N     2x2 TP within each node (NVLink),
                                  depth parallelism across nodes (Slingshot).

Generates one SLURM job per (mode, sparsity) combination.
All jobs for a given run share the same --seed for fair comparison.

Usage:

  # Sanity check: 4 nodes, 5B model, baseline vs 99% sparsity, both modes
  python launch_sparse.py --nodes 4 --model 5B --batch-size 64 \\
      --mode fsdp fsdp_tp --sparsity 0.0 0.99 --train-iters 100

  # Submit
  python launch_sparse.py --nodes 4 --model 5B --batch-size 64 \\
      --mode fsdp fsdp_tp --sparsity 0.0 0.99 --train-iters 100 --run

  # Larger sweep
  python launch_sparse.py --nodes 16 --model 20B --batch-size 512 \\
      --mode fsdp fsdp_tp --sparsity 0.0 0.5 0.9 0.99 --run
"""

import os
import argparse
import subprocess

GPUS_PER_NODE = 4

parser = argparse.ArgumentParser()
parser.add_argument("--run", action="store_true",
                    help="Submit generated scripts with sbatch")
parser.add_argument("--nodes", type=int, required=True,
                    help="Number of nodes (4 GPUs each). Supported: 4, 8, 16, 32+")
parser.add_argument("--model", type=str, required=True,
                    choices=["5B", "10B", "20B", "40B",
                             "60B", "80B", "160B", "320B", "640B"])
parser.add_argument("--batch-size", type=int, required=True,
                    help="Global batch size in samples")
parser.add_argument("--mode", type=str, nargs="+",
                    choices=["fsdp", "fsdp_tp"], default=["fsdp"],
                    help="Parallelism mode(s) to generate jobs for")
parser.add_argument("--sparsity", type=float, nargs="+", default=[0.0],
                    help="Gradient sparsity values to sweep")
parser.add_argument("--grad-sample-pct", type=float, default=100.0,
                    help="Pct of gradient elements sampled for threshold estimation "
                         "(100=exact, lower=faster approximate)")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--grad-acc", type=int, default=1,
                    help="Gradient accumulation steps")
parser.add_argument("--train-iters", type=int, default=100)
parser.add_argument("--seed", type=int, default=1234,
                    help="Random seed — identical across all jobs for fair comparison")
parser.add_argument("--time", type=int, default=30,
                    help="SLURM wall-clock limit in minutes")
args = parser.parse_args()

megatron_home = "/pscratch/sd/s/ssingh37/Megatron-LM"
log_folder    = os.path.join(megatron_home, "logs", "sparse")

# ---------------------------------------------------------------------------
# Model architecture  (matches model-table.md)
# ---------------------------------------------------------------------------
model_configs = {
    "5B":   dict(nlayers=24,  nhidden=4096,  nheads=32,  min_tp=4),
    "10B":  dict(nlayers=32,  nhidden=5120,  nheads=40,  min_tp=8),
    "20B":  dict(nlayers=32,  nhidden=7168,  nheads=56,  min_tp=16),
    "40B":  dict(nlayers=38,  nhidden=9216,  nheads=72,  min_tp=32),
    "60B":  dict(nlayers=56,  nhidden=9216,  nheads=72,  min_tp=64),
    "80B":  dict(nlayers=42,  nhidden=12288, nheads=96,  min_tp=64),
    "160B": dict(nlayers=84,  nhidden=12288, nheads=96,  min_tp=128),
    "320B": dict(nlayers=96,  nhidden=16384, nheads=128, min_tp=256),
    "640B": dict(nlayers=192, nhidden=16384, nheads=128, min_tp=512),
}
cfg     = model_configs[args.model]
nlayers = cfg["nlayers"]
nhidden = cfg["nhidden"]
nheads  = cfg["nheads"]

N    = args.nodes
GPUS = N * GPUS_PER_NODE
gbs  = args.batch_size
sq   = args.seq_len

# ---------------------------------------------------------------------------
# Parallelism configs per mode
# ---------------------------------------------------------------------------
# Both modes use dp=1: all GPUs go to tensor parallelism.
# mbs must be divisible by dtp because AxoNN's drop() shards the batch
# across the depth group (each depth rank gets mbs/dtp samples).

def get_parallel_config(mode, N):
    if mode == "fsdp":
        # All 4*N GPUs form one depth group — pure FSDP across the whole job.
        return dict(ctp=1, rtp=1, dtp=N * GPUS_PER_NODE)
    elif mode == "fsdp_tp":
        # 2x2 intra-node TP (Gc=2, Gr=2) uses all 4 GPUs per node via NVLink.
        # N-way depth across nodes uses Slingshot.
        assert GPUS_PER_NODE == 4
        return dict(ctp=2, rtp=2, dtp=N)

# ---------------------------------------------------------------------------
# Batch size validation
# ---------------------------------------------------------------------------
def check_batch(gbs, grad_acc, dtp, mode):
    assert gbs % grad_acc == 0, \
        f"batch_size {gbs} not divisible by grad_acc {grad_acc}"
    mbs = gbs // grad_acc
    assert mbs % dtp == 0, (
        f"[{mode}] micro_batch_size={mbs} not divisible by dtp={dtp}. "
        f"Minimum batch size for this config: {dtp * grad_acc}."
    )
    return mbs

# ---------------------------------------------------------------------------
# Generate jobs
# ---------------------------------------------------------------------------
with open("template_sparse.sh") as f:
    template = f.read()

for mode in args.mode:
    pc  = get_parallel_config(mode, N)
    ctp = pc["ctp"]
    rtp = pc["rtp"]
    dtp = pc["dtp"]
    mbs = check_batch(gbs, args.grad_acc, dtp, mode)

    folder = os.path.join(log_folder, args.model, f"N{N}_{mode}")
    os.makedirs(folder, exist_ok=True)

    for sparsity in args.sparsity:
        sp_tag   = f"sp{sparsity:.2f}".replace(".", "p")
        exp_name = f"{args.model}_N{N}_{mode}_GBS{gbs}_MBS{mbs}_{sp_tag}"

        script = template.format(
            nodes=N,
            time=args.time,
            output=os.path.join(folder, f"{exp_name}.out"),
            megatron_home=megatron_home,
            nlayers=nlayers,
            nhidden=nhidden,
            nheads=nheads,
            ctp=ctp,
            rtp=rtp,
            dtp=dtp,
            gbs=gbs,
            mbs=mbs,
            sq=sq,
            grad_sparsity=sparsity,
            grad_sample_pct=args.grad_sample_pct,
            train_iters=args.train_iters,
            seed=args.seed,
        )

        script_file = os.path.join(folder, f"{exp_name}.sh")
        with open(script_file, "w") as f:
            f.write(script)

        job_name = f"sparse_{args.model}_{mode}_{sp_tag}"
        cmd = f"sbatch -t {args.time} --dependency=singleton -J {job_name} {script_file}"
        print(cmd)
        if args.run:
            subprocess.run(cmd.split())
