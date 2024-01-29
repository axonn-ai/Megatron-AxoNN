# How to setup on frontier

## Installing all dependencies
# Note that this is a python virtual environment based setup
# You might need to change this a bit for conda 
```
cd /lustre/orion/scratch/$(whoami)/csc569/
bash install_everything_on_frontier.sh
```

This should work, let Siddharth know if it doesn't

## Training TinyLLaMA
First checkout the tiny-llama branch of megatron-axonn
Then open `examples/run_axonn_amd_tinyllama.sh`, and change the following

```
# These are the two things you need to change as per your setup
# 1. Make LD_LIBRARY_PATH point to wherever your plugin is installed
# this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/path/to/aws-ofi-rccl/build/lib"
# 2. Make PYTHONPATH point to your local clone of litgpt
export PYTHONPATH="$PYTHONPATH:/path/to/lit-gpt-dev"
```

Now you are ready to train.
To launch on 16 nodes (128 GPUs) for 2 hours
```
## checkout the tiny-llama branch
sbatch -N 128 -t 02:00:00 examples/run_axonn_amd_tinyllama.sh
``` 

