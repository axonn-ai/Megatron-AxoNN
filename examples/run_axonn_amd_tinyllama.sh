#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569
#SBATCH -C nvme

# tiny_llama = [
#     dict(
#         name="tiny-llama-1.1b{}",
#         hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
#         block_size=2048, #### CHECKED (Seq Length)
#         vocab_size=32000, #### NEED TO ADD VOCAB FILES
#         padding_multiple=64, ### NOT RELEVANT
#         n_layer=22, ### CHECKED (NUM_LAYERS)
#         n_head=32, ### CHECKED (NUM_HEADS)	
#         n_embd=2048, ### CHECKED (HIDDEN_SIZE)
#         rotary_percentage=1.0, ### TODO: CHECK IF -- USING ROTARY (I think the default is 1: https://github.com/search?q=repo%3Aaxonn-ai%2FMegatron-AxoNN+rotary_percent&type=code)
#         parallel_residual=False, ### TODO: CHECK (https://github.com/azshue/lit-gpt-dev/blob/99fb9363646bfacb686f72f58274392e6036ad6c/lit_gpt/model.py#L157 and apply_residual_connection_post_layernorm are the same)
#         bias=False, ### TODO: CHECK "disable-bias-linear" I think. This is the bias for the linear layers
#         _norm_class="RMSNorm",  ### CHECKED "--normalization RMSNorm"
#         norm_eps=1e-5, ### TODO: UNLCLEAR WHERE THIS IS -- I think this is fine (https://github.com/search?q=repo%3Aaxonn-ai%2FMegatron-AxoNN%20norm_eps&type=code)
#         _mlp_class="LLaMAMLP", ### CHECKED "From Line 112, # --swiglu makes ParallelMLP equivalent to LLAMAMLP"
#         intermediate_size=5632, ### CHECKED "FFN_HIDDEN_SIZE"
#         n_query_groups=4, #### CHECKED: NUM_QUERY_GROUPS
#     )
# ]
### WE want global batch size of 4M so 4000000/2048
#### We are gonna copy Olma's BS of 4M
# global_batch_size = 2048 #NEEL: UPDATED IN BASH SCRIPT
# learning_rate = 4e-4 #NEEL: Checked "--lr 4.0e-4"
#### THIS COULD BE SET ACCORDING TO HOW MANY GPUs we want to use
# micro_batch_size = 8
# max_tokens = int(1e12)  #NEEL: UPDATED IN BASH SCRIPT
# warmup_steps = 2000 # We are gonna use tinyllama warmup steps
#### BELOW ARE IRRELVANT ####
# log_step_interval = 1
# eval_iters = 100
# save_step_interval = 1000
# eval_step_interval = 1000
#### ABOVE ARE IRRELVANT ####

# weight_decay = 1e-1  ### Neel: CHECKED "weight-decay 1e-1"
# beta1 = 0.9 ### Neel: CHECKED
# beta2 = 0.95 ### Neel: CHECKED
# grad_clip = 1.0 ### Neel: CHECKED 
# decay_lr = True <--- This is irrevalant
# min_lr = 4e-5 ### Neel: CHECKED 

## calculating the number of nodes and GPUs
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 

userid=$(whoami)
# These are the two things you need to change as per your setup
# 1. Make LD_LIBRARY_PATH point to wherever your plugin is installed
# this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/ccs/home/$userid/aws-ofi-rccl/build/lib"
# 2. Make PYTHONPATH point to your local clone of litgpt
export PYTHONPATH="$PYTHONPATH:/lustre/orion/scratch/$userid/csc547/lit-gpt-dev"


# This blob is setting up my python venv, ignore for conda builds
echo "moving environment to burst buffer"
## load venv onto burst buffer
srun -N $NNODES --ntasks-per-node=1 prepare_venv.sh
## delete old symbolic link
rm -rf ~/axonn_venv
## create new symbolic link
ln -s /mnt/bb/ssingh37/axonn_venv ~/axonn_venv
module load PrgEnv-cray
module load cray-python/3.9.13.1
. /ccs/home/$userid/axonn_venv/bin/activate


module load amd-mixed/5.6.0 #this should match with the rocm version your pytorch uses
module load libfabric

export MPICH_GPU_SUPPORT_ENABLED=0

## some RCCL env variables
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

# train_data_dir and val_data_dir are set to this as of now
DATADIR="/lustre/orion/csc569/proj-shared/language_datasets/"
DATASET="spj_star_combined_full_tinyllama_tokd"
DATAPATH="$DATADIR/$DATASET"

TOKENIZER_DIR="/lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/llama-tokenizer"
TOKENIZER_MODEL="${TOKENIZER_DIR}/tokenizer.model"

# we will save and load model checkpoints here
# if these are non-empty training will restart from the latest checkpoint here
# else training will start from scratch
CHECKPOINT_PATH="/lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/checkpoints/dataloader_correction"

# tiny-llama1.1B architecture shapes
# https://github.com/azshue/lit-gpt-dev/blob/tiny-llama/lit_gpt/config.py
NUM_LAYERS=22
NUM_HEADS=32
HIDDEN_SIZE=2048	
FFN_HIDDEN_SIZE=5632
NUM_QUERY_GROUPS=4

# batch size, seq length, and iterations
GLOBAL_BATCH_SIZE=2048 ## Neel: 2048x2048 = 4M per batch
SEQUENCE_LENGTH=2048
TOKENS_IN_BILLIONS=1000 ### Neel: Changed 1T #####
TRAIN_ITERS=$(( TOKENS_IN_BILLIONS * 1000000000 / GLOBAL_BATCH_SIZE / SEQUENCE_LENGTH  + 100 )) 
echo "Number of training iterations : ${TRAIN_ITERS}"

## AxoNN parallelism args
## These do not affect the science
ROW_TENSOR_PARR=1
COLUMN_TENSOR_PARR=1
DEPTH_TENSOR_PARR=1
PIPE_PARR=1
CACHE_LAYERS=0
OVERLAP=True


GRAD_ACC=2
GRADIENT_CHECKPOINT=False

## DERIVED ARGUMENTS (ignore)
MP=$(( ROW_TENSOR_PARR * COLUMN_TENSOR_PARR * DEPTH_TENSOR_PARR ))
DP=$(( GPUS / MP ))
MICRO_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / DP / GRAD_ACC ))

# The following args enable LLaMA
# --swiglu makes ParallelMLP equivalent to LLAMAMLP
# --group-query-attention - enables group query attention
# --num-query-groups - number of query groups for group query attention
# --normalization RMSNorm - switch from layernorm to RMSNorm (someone confirm?)
# --use-rotary-position-embeddings - use RoPE embeddings instead of learned position embeddings
# --untie-embeddings-and-output-weights - untie embedding and last layer weights
# --disable-bias-linear - disables bias in all nn.linear layers

# The following args disable features not compatible with AMD
# --no-gradient-accumulation-fusion 
# --use-amd 

GPT_ARGS="
    --row-tensor-model-parallel-size ${ROW_TENSOR_PARR} \
    --column-tensor-model-parallel-size ${COLUMN_TENSOR_PARR} \
    --depth-tensor-model-parallel-size ${DEPTH_TENSOR_PARR} \
    --pipeline-model-parallel-size ${PIPE_PARR} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 4.0e-4 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --min-lr 4.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-amd \
    --use-flash-attn \
    --swiglu \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --use-apex-adam \
    --seed 78965 \
    --attention-dropout 0 \
    --hidden-dropout 0
"

if [[ $GRADIENT_CHECKPOINT == "True" ]]
then
    GPT_ARGS="${GPT_ARGS} --recompute-granularity full \
    			  --recompute-method uniform \
    			  --recompute-num-layers 1"
fi

## AxoNN specific args for communication optimizations
# these do not affect the ML science
if [[ $OVERLAP == "True" ]]
then
	GPT_ARGS="${GPT_ARGS} \
		--overlap-axonn-comm \
		--overlap-axonn-reduce-scatter \
		--overlap-axonn-all-gather\
		--num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather ${CACHE_LAYERS}"
fi

# --lit-gpt-data-path - is pointing to your dataset
# currently both train and val splits are taken fron --data-path
# the --custom-dataloader argument bypasses megatron's dataloaders
# --num-workers 0 - disables multiprocesses dataloading 
# which can hang jobs at scale

DATA_ARGS="
    --lit-gpt-data-path $DATAPATH \
    --custom-dataloader \
    --num-workers 0 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}
"

# --eval-interval 1000 - do validation after every 1000 arguments
# --eval-iters 100 - do validation for 100 iterations
# --save-interval 1000 - save the model after every 1000 iterations
# --log-interval 1 - print iteration lossees after every 1 iteration
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 100 \
"

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"


export OMP_NUM_THREADS=7 
run_cmd="srun -N ${NNODES} -n ${GPUS} -c7 --gpus-per-task=1 --gpu-bind=closest ./examples/get_rank_from_slurm.sh ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
