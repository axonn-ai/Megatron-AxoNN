# How to setup on frontier

## Installing all dependencies
```
cd /lustre/orion/scratch/$(whoami)/csc569/
bash install_everything_on_frontier.sh
```

This should work, let Siddharth know if it doesn't


## Training TinyLLaMA
To launch on 16 nodes (128 GPUs) for 2 hours
```
## checkout the tiny-llama branch
sbatch -N 128 -t 02:00:00 examples/run_axonn_amd_tinyllama.sh
``` 

