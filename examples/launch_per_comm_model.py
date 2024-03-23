import os
from comm_model import get_configs_for_transformer
import argparse
from prime import closest_prime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', action="store_true")
parser.add_argument('--gpus', type=int, help="number of GPUs")
parser.add_argument('--batch-size', type=int, help="batch size (in #samples)")
parser.add_argument('--time', type=int, help="launch wall clock time (in mins)", default=20)
parser.add_argument('--seq-len', type=int, default=2048)
parser.add_argument('--cache-all', action='store_true', default=False)
parser.add_argument('--grad-acc', type=int, help="gradient acc degree", default=1)
parser.add_argument('--model', type=str, choices=["5B", "10B", "20B", "40B", "50B", "60B", "80B", "100B", "175B", "160B"])
parser.add_argument('--manual', action='store_true', default=False, help="run with manual configuration")
parser.add_argument('--config', type=int, nargs='+', help="if --manual, pass Gr,Gc,Gd as tuple here")

args = parser.parse_args()

#megatron_home = "/pscratch/sd/s/ssingh37/Megatron-LM/"
megatron_home = "/pscratch/sd/s/ssingh37/Megatron-LM"
trace_folder = "/pscratch/sd/s/ssingh37/gordon_bell_traces/"
model=args.model

## arch

# 175B
if model == "175B":
    nlayers=96
    nhidden=12288
    nheads=96
    min_tp=128

elif model == "80B":
    nlayers=42
    nhidden=12288
    nheads=96
    min_tp=64

elif model == "100B":
    nlayers=53
    nhidden=12288
    nheads=96
    min_tp=64

elif model == "160B":
    nlayers=84
    nhidden=12288
    nheads=96
    min_tp=128

# 40B
elif model == "60B":
    nlayers=56
    nhidden=9216
    nheads=72
    min_tp=64
# 40B
elif model == "40B":
    nlayers=38
    nhidden=9216
    nheads=72
    min_tp=32

# 40B
elif model == "50B":
    nlayers=48
    nhidden=9216
    nheads=72
    min_tp=32
# 20B
elif model == "20B":
    nlayers=32
    nhidden=7168
    nheads=56
    min_tp=16
#10B
elif model == "10B":
    nlayers=32
    nhidden=5120
    nheads=40
    min_tp=8
#5B
elif model == "5B":
    nlayers=24
    nhidden=4096
    nheads=32
    min_tp=4

else:
    raise NotImplementedError

## gbs and sq
gbs=args.batch_size
sq=args.seq_len
GPUS=args.gpus
topk=20
grad_acc = args.grad_acc

if not args.manual:
    top_k_configs = get_configs_for_transformer(
            global_batch_size_in_samples=gbs,
            sequence_length=sq,
            num_layers=nlayers,
            hidden_size=nhidden, 
            GPUs=GPUS,
            minimum_degree_of_tensor_parallelism=min_tp,
            model_version="v2",
            topk=topk,
            no_dp=False,
            machine="perlmutter",
            grad_acc=grad_acc
    )
    print(top_k_configs)
    ## perf hparams
    ctps=top_k_configs["Gc"]
    rtps=top_k_configs["Gr"]
    dtps=top_k_configs["Gd"]
else:
    config = tuple(args.config)
    rtps = [config[0]]
    ctps = [config[1]] 
    dtps = [config[2]]
GPUS_PER_NODE=4
ncache=0
lcache=0
if args.cache_all:
    lcache=2
    ncache=nlayers


folder = f"{megatron_home}/logs/per_comm_model/{model}/"

if not os.path.exists(folder):
    os.makedirs(folder)

with open("template_perlmutter.sh") as f:
    template = f.read()

def sanity_checks(gpu):
    mp = ctp * rtp * dtp
    assert gpu % mp == 0
    dp = gpu // mp
    assert gbs % dp == 0
    bs_per_dp = gbs // dp
    assert bs_per_dp % mbs == 0
    assert mbs % dtp == 0

gpu=GPUS

def get_profile_ranks(gpu, fact=100):
    p = closest_prime(gpu // fact)
    ranks = np.arange(p, gpu, p)
    arg = ""
    for rank in ranks:
        arg += f"{rank} "
    return arg

for ctp,rtp,dtp in zip(ctps, rtps, dtps):
    assert  gpu % GPUS_PER_NODE == 0
    nodes = gpu // GPUS_PER_NODE

    dp = gpu // (rtp*ctp*dtp)
    bs_per_dp = gbs // dp
    mbs = bs_per_dp // grad_acc
    
    try:
        sanity_checks(gpu)
    except AssertionError:
        continue
    exp_name = f"GPUS_{gpu}_BS_{gbs}_MBS_{mbs}_sq_{sq}_{rtp}x{ctp}x{dtp}"
    output_file = os.path.join(folder, f"{exp_name}.out")
    trace_subfolder = os.path.join(os.path.join(os.path.join(trace_folder, model), f"GPUS_{gpu}"), 
                                   f"{exp_name}")
    script = template.format(
                nodes=nodes,
                nlayers=nlayers,
                nhidden=nhidden,
                nheads=nheads,
                gbs=gbs,
                mbs=mbs,
                sq=sq,
                ctp=ctp,
                dtp=dtp,
                rtp=rtp,
                ncache=ncache,
                lcache=lcache,
                output=output_file,
                megatron_home=megatron_home,
                trace_path=trace_subfolder,
                profile_ranks=get_profile_ranks(gpu))
    script_file = os.path.join(folder, f"{exp_name}.sh")
    with open(script_file, "w") as f:
        f.write(script)
    print(f"sbatch -t {args.time} --reservation axonn_gb1 --dependency=singleton -J AxoNN {script_file}")
    if args.run:
        import subprocess
        subprocess.run(["sbatch", "-t" , f"{args.time}", "--reservation", "axonn_gb1" ,f"{script_file}"])
