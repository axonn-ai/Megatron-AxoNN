#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=4
#SBATCH --time={time}
#SBATCH --output={output}

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

DATA_DIR="$SCRATCH/gpt_data"
VOCAB_FILE="$DATA_DIR/gpt2-vocab.json"
MERGE_FILE="$DATA_DIR/gpt2-merges.txt"
DATA_PATH="$DATA_DIR/BookCorpusDataset_text_document"

## ARCHITECTURE
NUM_LAYERS={nlayers}
NUM_HEADS={nheads}
HIDDEN_SIZE={nhidden}

## PARALLELISM
# fsdp:    Gc=1, Gr=1, Gd=4*N  (pure depth / full FSDP across all GPUs)
# fsdp_tp: Gc=2, Gr=2, Gd=N    (2x2 intra-node NVLink TP + inter-node depth)
COLUMN_TENSOR_PARR={ctp}
ROW_TENSOR_PARR={rtp}
DEPTH_TENSOR_PARR={dtp}
PIPE_PARR=1

## SPARSITY
GRAD_SPARSITY={grad_sparsity}
GRAD_SAMPLE_PCT={grad_sample_pct}

## BATCH SIZES
MICRO_BATCH_SIZE={mbs}
GLOBAL_BATCH_SIZE={gbs}
SEQUENCE_LENGTH={sq}
TRAIN_ITERS={train_iters}
SEED={seed}

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
    --num-workers 2 \
    --seed $SEED \
    --grad-sparsity $GRAD_SPARSITY \
    --grad-sample-pct $GRAD_SAMPLE_PCT \
    --overlap-axonn-comm \
    --overlap-axonn-reduce-scatter \
    --overlap-axonn-all-gather
"

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

# --save $CHECKPOINT_PATH \
# --load $CHECKPOINT_PATH

SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
"

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 shifter ./examples/get_rank_from_slurm.sh $SCRIPT"

echo $run_cmd
eval $run_cmd
