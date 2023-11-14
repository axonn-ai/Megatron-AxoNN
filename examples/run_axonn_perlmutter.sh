#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --account=m2404_g
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00


# Runs a "10B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1


NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
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

SCRATCH_SUBSPACE="${SCRATCH}/GodonBellData/MegatronAX"

DATA_DIR="${SCRATCH_SUBSPACE}/gpt_data"
CHECKPOINT_PATH="${DATA_DIR}/checkpoints"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"

## ARCHITECTURE DETAILS
NUM_LAYERS=32
HIDDEN_SIZE=7168
NUM_HEADS=112

## PARALLELISM DETAILS
ROW_TENSOR_PARR=$1
COLUMN_TENSOR_PARR=$2
DEPTH_TENSOR_PARR=$3

if [ $((COLUMN_TENSOR_PARR * ROW_TENSOR_PARR * DEPTH_TENSOR_PARR)) -eq ${GPUS} ]; then
	echo "Tensor parallel split correct"
else 
	echo "Incorrect Tensor parallel split"
	exit 0
fi

## BATCH SIZES
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=$4
SEQUENCE_LENGTH=2048


GPT_ARGS="
    --row-tensor-model-parallel-size ${ROW_TENSOR_PARR} \
    --column-tensor-model-parallel-size ${COLUMN_TENSOR_PARR} \
    --depth-tensor-model-parallel-size ${DEPTH_TENSOR_PARR} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 2 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-flash-attn \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 1\
    --timing-log-level 2
"
#    --eval-interval 1 \
#    --eval-iters 1
#"

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
"
MODEL_SIZE=$(echo "scale=0; (12 * ${NUM_LAYERS} * (${HIDDEN_SIZE}^2) / 1000000000) + 1" | bc)

OUTDIR="${SCRATCH}/GodonBellData/MegatronAX/exp_data/exp-${MODEL_SIZE}B-${GPUS}GPU-rcdvary-bvary"
mkdir -p $OUTDIR
OUTFILE="${OUTDIR}/run_${ROW_TENSOR_PARR}x${COLUMN_TENSOR_PARR}x${DEPTH_TENSOR_PARR}_${GLOBAL_BATCH_SIZE}x${MICRO_BATCH_SIZE}x${SEQUENCE_LENGTH}.txt"

echo $OUTFILE

run_cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=4 ${SCRIPT} | tee ${OUTFILE}"

echo ${run_cmd}
eval ${run_cmd}
set +x
