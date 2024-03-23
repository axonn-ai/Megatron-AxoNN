#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=4
#SBATCH --time=20
#SBATCH --output={output}

# Runs the "345M" parameter model

 source ~/.bashrc_old

cd {megatron_home}



NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export WORLD_SIZE=$GPUS
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read

DATA_DIR="$SCRATCH/gpt_data"
CHECKPOINT_PATH="$DATA_DIR/checkpoints"
VOCAB_FILE="$DATA_DIR/gpt2-vocab.json"
MERGE_FILE="$DATA_DIR/gpt2-merges.txt"
DATA_PATH="$DATA_DIR/BookCorpusDataset_text_document"

## ARCHITECTURE DETAILS
NUM_LAYERS={nlayers}
NUM_HEADS={nheads}
HIDDEN_SIZE={nhidden}

## PARALLELISM DETAILS
COLUMN_TENSOR_PARR={ctp}
ROW_TENSOR_PARR={rtp}
DEPTH_TENSOR_PARR={dtp}
PIPE_PARR=1
CACHE_LEVEL={lcache}
CACHE_LAYERS={ncache}
OVERLAP=True

NSYS_PROFILE=False
TORCH_PROFILE=False
PROFILE_NAME="test_10B_16x1"

## BATCH SIZES
MICRO_BATCH_SIZE={mbs}
GLOBAL_BATCH_SIZE={gbs}
SEQUENCE_LENGTH={sq}
TRAIN_ITERS=20

GPT_ARGS="
    --row-tensor-model-parallel-size $ROW_TENSOR_PARR \
    --column-tensor-model-parallel-size $COLUMN_TENSOR_PARR \
    --depth-tensor-model-parallel-size $DEPTH_TENSOR_PARR \
    --pipeline-model-parallel-size $PIPE_PARR \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQUENCE_LENGTH \
    --max-position-embeddings $SEQUENCE_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --init-method-std 0.006 \
    --lr 6e-5 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-1 \
    --adam-eps 1e-5 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --layer-caching-level $CACHE_LEVEL \
    --num-workers 2

"
if [[ $OVERLAP == "True" ]]
then
	GPT_ARGS="$GPT_ARGS \
		--overlap-axonn-comm \
		--overlap-axonn-reduce-scatter \
		--overlap-axonn-all-gather\
		--num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather $CACHE_LAYERS"
fi


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0
"



SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
"

if [[ $NSYS_PROFILE == "True" ]]
then
	echo "profiling with nsys"
	SCRIPT="nsys profile -s none \
		-t nvtx,cuda -o $PROFILE_NAME \
		--force-overwrite=true  \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		$SCRIPT \
		--profile-step-start 5 \
		--profile-step-end 10 \
		--profile
		"
fi

TRACE_PATH={trace_path}
PROFILE_RANKS="{profile_ranks}"

if [[ $TORCH_PROFILE == "True" ]]
then
	echo "profiling with torch profiler"
	mkdir -p $TRACE_PATH
	SCRIPT=" $SCRIPT \
		--profile-step-start 7 \
		--profile-step-end 10 \
		--profile \
		--path-for-traces $TRACE_PATH \
		--profile-ranks $PROFILE_RANKS
		"
fi

#--profile-ranks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

# add these args if you want to save and load checkpoints
#--save $CHECKPOINT_PATH \
# --load $CHECKPOINT_PATH

#export TORCH_INIT_FILE="/pscratch/sd/s/ssingh37/torch_cache/$SLURM_JOBID"
#rm $TORCH_INIT_FILE
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 shifter ./examples/get_rank_from_slurm.sh $SCRIPT" 

echo $run_cmd
eval $run_cmd
set +x
