#!/bin/bash
# Parametric base job — do not submit directly; use launch_all.sh
# Required env vars (set by launch_all.sh via sbatch --export):
#   JOB_LABEL        short label for logs/TB (e.g. g16_dense_nobd)
#   DO_PRUNE         0 or 1  (enables AXONN_PRUNE_AR + USE_SPARSE_AR)
#   DO_ERROR_ACCUM   0 or 1  (only meaningful when DO_PRUNE=1)
#   DO_BREAKDOWN     0 or 1  (enables AXONN_TIME_OPS)

set -euo pipefail

# ===========================================================================
# Experiment config — shared across all runs
# GBS is computed dynamically below as 2 * GPUS (MBS=2 per GPU)
# ===========================================================================
MODEL=1XL
MODE=fsdp
# SPARSITY=0.995
SPARSITY=0.99
SAMPLE_PCT=1
SEQ_LEN=512
TRAIN_ITERS=100
SEED=42
NCHANNELS=64
# ===========================================================================

SCRIPT_DIR="/global/u1/e/egencer/scratch/sparsecomms/Megatron-AxoNN"
cd "$SCRIPT_DIR"
mkdir -p logs

module load pytorch/2.8.0

if [ -d "$SCRIPT_DIR/../.venv" ]; then
    source "$SCRIPT_DIR/../.venv/bin/activate"
fi

# --- Sparse comm flags (derived from DO_PRUNE / DO_ERROR_ACCUM / DO_BREAKDOWN) ---
export AXONN_TIME_OPS=$DO_BREAKDOWN
export AXONN_PRUNE_TRITON=1
export AXONN_PRUNE_ERROR_DTYPE=float16
# export AXONN_PRUNE_ERROR_DTYPE=float8_e5m2
# export AXONN_PRUNE_ERROR_DTYPE=float32
# export AXONN_PRUNE_ERROR_DTYPE=fp8e4

if (( DO_PRUNE )); then
    export USE_SPARSE_AR=1
    export AXONN_PRUNE_AR=1
    export AXONN_PRUNE_FP8_SCALE=0
    export AXONN_PRUNE_ERROR_ACCUMULATE=$DO_ERROR_ACCUM
    export AXONN_PRUNE_MEASURE_SPARSITY=1
else
    export USE_SPARSE_AR=0
    export AXONN_PRUNE_AR=0
    export AXONN_PRUNE_ERROR_ACCUMULATE=0
fi

export USE_SPARSE_RS=0
export AXONN_PRUNE_RS=0
export AXONN_PRUNE_SPARSITY=$SPARSITY
export AXONN_PRUNE_SAMPLE_PCT=$SAMPLE_PCT
export SPARSE_COMMS_LOG_SPARSITY=0
export NCCL_RS_SHIM_TIMING=0
export NCCL_RS_SHIM_STATS=0

# --- NCCL / Libfabric ---
export LD_PRELOAD="/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/build/ncclx/lib/libnccl.so.2"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:3; 1:2; 2:1; 3:0"

export NCCL_BUFFSIZE=4404032
export NCCL_CCD_FORMAT_MASK=5
export NCCL_CCD_DENSE_THRESHOLD=0.5
export NCCL_CCD_DENSE_INTRA_THRESHOLD=0.6
export NCCL_CCD_AG_DENSE_THRESHOLD=0.1
export NCCL_CCD_CHANNELS=$NCHANNELS
export NCCL_MAX_NCHANNELS=$NCHANNELS

# --- Distributed ---
NNODES=${NNODES:-$SLURM_JOB_NUM_NODES}
GPUS=$(( NNODES * 4 ))
GBS=$(( 2 * GPUS ))    # MBS=2 per GPU at all scales
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

# --- Data ---
DATA_DIR="$SCRATCH/sparsecomms/gpt_data"
VOCAB_FILE="$DATA_DIR/gpt2-vocab.json"
MERGE_FILE="$DATA_DIR/gpt2-merges.txt"
DATA_PATH="$DATA_DIR/BookCorpusDataset_text_document"

# --- Model architecture ---
case $MODEL in
  125M)  NUM_LAYERS=12;  HIDDEN_SIZE=768;   NUM_HEADS=12  ;;
  350M)  NUM_LAYERS=24;  HIDDEN_SIZE=1024;  NUM_HEADS=16  ;;
  760M)  NUM_LAYERS=24;  HIDDEN_SIZE=1536;  NUM_HEADS=16  ;;
  1B)    NUM_LAYERS=24;  HIDDEN_SIZE=2048;  NUM_HEADS=32  ;;
  1BH)   NUM_LAYERS=29;  HIDDEN_SIZE=2048;  NUM_HEADS=32  ;;
  1XL)   NUM_LAYERS=48;  HIDDEN_SIZE=1600;  NUM_HEADS=25  ;;
  2B)    NUM_LAYERS=32;  HIDDEN_SIZE=2560;  NUM_HEADS=32  ;;
  6B)    NUM_LAYERS=32;  HIDDEN_SIZE=4096;  NUM_HEADS=32  ;;
  5B)    NUM_LAYERS=24;  HIDDEN_SIZE=4096;  NUM_HEADS=32  ;;
  10B)   NUM_LAYERS=32;  HIDDEN_SIZE=5120;  NUM_HEADS=40  ;;
  *) echo "Unknown model $MODEL"; exit 1 ;;
esac

# --- Parallelism ---
case $MODE in
  fsdp)    CTP=1; RTP=1; DTP=1 ;;
  fsdp_tp) CTP=2; RTP=2; DTP=$NNODES ;;
  *) echo "Unknown mode $MODE"; exit 1 ;;
esac

if (( GBS % DTP != 0 )); then
  echo "ERROR: GBS=$GBS not divisible by DTP=$DTP"; exit 1
fi
MBS=$(( GBS / GPUS ))

TB_DIR="$SCRIPT_DIR/tensorboard/${SLURM_JOB_ID}_${JOB_LABEL}_scaling"

echo "=================================================="
echo "SLURM job:      $SLURM_JOB_ID  label=$JOB_LABEL"
echo "Model:          $MODEL  (layers=$NUM_LAYERS  hidden=$HIDDEN_SIZE  heads=$NUM_HEADS)"
echo "Mode:           $MODE  (Gc=$CTP  Gr=$RTP  Gd=$DTP)"
echo "Nodes:          $NNODES  GPUs: $GPUS"
echo "GBS=$GBS  MBS=$MBS  SEQ=$SEQ_LEN  Iters=$TRAIN_ITERS  Seed=$SEED"
echo "--- Sparse flags ---"
echo "DO_PRUNE=$DO_PRUNE  DO_ERROR_ACCUM=$DO_ERROR_ACCUM  DO_BREAKDOWN=$DO_BREAKDOWN"
echo "AXONN_PRUNE_AR=$AXONN_PRUNE_AR  USE_SPARSE_AR=$USE_SPARSE_AR"
echo "AXONN_PRUNE_ERROR_ACCUMULATE=$AXONN_PRUNE_ERROR_ACCUMULATE  AXONN_PRUNE_ERROR_DTYPE=$AXONN_PRUNE_ERROR_DTYPE"
echo "AXONN_TIME_OPS=$AXONN_TIME_OPS"
echo "Sparsity=$SPARSITY  SamplePct=$SAMPLE_PCT"
echo "TB_DIR=$TB_DIR"
echo "=================================================="

GPT_ARGS="
    --column-tensor-model-parallel-size $CTP \
    --row-tensor-model-parallel-size $RTP \
    --depth-tensor-model-parallel-size $DTP \
    --pipeline-model-parallel-size 1 \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
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
    --grad-sparsity $SPARSITY \
    --grad-sample-pct $SAMPLE_PCT \
    --overlap-axonn-comm \
    --overlap-axonn-reduce-scatter \
    --overlap-axonn-all-gather \
    --num-layers-for-caching-weights-in-depth-tensor-parallel-all-gather 0 \
    --layer-caching-level 0
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

mkdir -p "$TB_DIR"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --tensorboard-dir $TB_DIR \
    --tensorboard-log-interval 1 \
    --log-batch-size-to-tensorboard \
    --log-memory-to-tensorboard
"

# Breakdown logging adds timer output to tensorboard
if (( DO_BREAKDOWN )); then
    OUTPUT_ARGS="$OUTPUT_ARGS --log-timers-to-tensorboard"
fi

srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 \
    ./examples/get_rank_from_slurm.sh \
    python -u pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl
