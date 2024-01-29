#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569


userid=$(whoami)
# These are the two things you need to change as per your setup
# 1. Make LD_LIBRARY_PATH point to wherever your plugin is installed
# this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/ccs/home/$userid/aws-ofi-rccl/build/lib"
# 2. Make PYTHONPATH point to your local clone of litgpt
export PYTHONPATH="$PYTHONPATH:/lustre/orion/scratch/$userid/csc547/lit-gpt-dev"

# The rest of the script should work as it is

echo "This TinyLLAMA script will work for <=512 GPUs."

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

## calculating the number of nodes and GPUs
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 

# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

# train_data_dir and val_data_dir are set to this as of now
DATADIR="/lustre/orion/csc569/proj-shared/language_datasets/"
DATASET="spj_star_combined_full_tinyllama_tokd"
DATAPATH="$DATADIR/$DATASET"


# these are redundant for tiny-llams, so ignore
MEGATRON_TOKENIZER_DIR="/lustre/orion/proj-shared/csc569/book_corpus_megatron"
VOCAB_FILE="${MEGATRON_TOKENIZER_DIR}/gpt2-vocab.json"
MERGE_FILE="${MEGATRON_TOKENIZER_DIR}/gpt2-merges.txt"


# we will save and load model checkpoints here
# if these are non-empty training will restart from the latest checkpoint here
# else training will start from scratch
CHECKPOINT_PATH="/lustre/orion/csc569/proj-shared/megatron-axonn-tiny-llama-1.1b/checkpoints"


# tiny-llama1.1B architecture shapes
# https://github.com/azshue/lit-gpt-dev/blob/tiny-llama/lit_gpt/config.py
NUM_LAYERS=22
NUM_HEADS=32
HIDDEN_SIZE=2048	
FFN_HIDDEN_SIZE=5632
NUM_QUERY_GROUPS=4

# batch size, seq length, and iterations
GLOBAL_BATCH_SIZE=512
SEQUENCE_LENGTH=2048
TOKENS_IN_BILLIONS=3000
TRAIN_ITERS=$(( TOKENS_IN_BILLIONS * 1000000000 / GLOBAL_BATCH_SIZE / SEQUENCE_LENGTH  + 100 )) 
echo "Number of training iterations : ${TRAIN_ITERS}"

## AxoNN parallelism args
## These do not affect the science
ROW_TENSOR_PARR=1
COLUMN_TENSOR_PARR=1
DEPTH_TENSOR_PARR=2
PIPE_PARR=1
CACHE_LAYERS=22
OVERLAP=True


## DERIVED ARGUMENTS (ignore)
MP=$(( ROW_TENSOR_PARR * COLUMN_TENSOR_PARR * DEPTH_TENSOR_PARR ))
DP=$(( GPUS / MP ))
MICRO_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / DP ))

# The following args enable LLaMA
# --swiglu makes ParallelMLP equivalent to LLAMAMLP
# --group-query-attention - enables group query attention
# --num-query-groups - number of query groups for group query attention
# --normalization RMSNorm - switch from layernorm to RMSNorm (someone confirm?)
# --use-rotary-position-embeddings - use RoPE embeddings instead of learned position embeddings
#
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
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-flash-attn \
    --swiglu \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS}
"

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
    --num-workers 0
"

# these args are for megatron dataloaders
# these are not needed for litgpt, but not passing them
# might give you errors
# THESE DO NOTHING
REDUNDANT_DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
"

DATA_ARGS="${DATA_ARGS} ${REDUNDANT_DATA_ARGS}"

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
