#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --time=01:00:00
#SBATCH --account=m5083_g
#SBATCH --job-name=sparse_sanity
#SBATCH --output=logs/sparse_%j.out
#SBATCH --error=logs/sparse_%j.err

set -euo pipefail

# ===========================================================================
# Experiment config — edit these before submitting
# ===========================================================================
MODEL=20B         # 5B 10B 20B 40B 60B 80B 160B 320B 640B
MODE=fsdp         # fsdp | fsdp_tp
SPARSITY=0.0      # 0.0 = baseline, 0.99 = 99% pruning
GBS=512           # global batch size (must be divisible by DTP, see below)
SEQ_LEN=2048
TRAIN_ITERS=100
SEED=42
# ===========================================================================

SCRIPT_DIR="/global/u1/e/egencer/scratch/sparsecomms/Megatron-AxoNN"
cd "$SCRIPT_DIR"   # Megatron-AxoNN root
mkdir -p logs

# --- Modules ---
module load pytorch/2.8.0
module load nccl
module load cudatoolkit/12.4
module load PrgEnv-gnu
module load cray-mpich
module load craype-accel-nvidia80

# Activate venv if present
if [ -d "$SCRIPT_DIR/../.venv" ]; then
    source "$SCRIPT_DIR/../.venv/bin/activate"
fi

# --- NCCL / Libfabric (Perlmutter Slingshot-11) ---
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:3; 1:2; 2:1; 3:0"
export CUDA_VISIBLE_DEVICES=3,2,1,0
export OMP_NUM_THREADS=8

# --- Distributed ---
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

# --- Data ---
DATA_DIR="$SCRATCH/sparsecomms/gpt_data"
VOCAB_FILE="$DATA_DIR/gpt2-vocab.json"
MERGE_FILE="$DATA_DIR/gpt2-merges.txt"
DATA_PATH="$DATA_DIR/BookCorpusDataset_text_document"

# --- Model architecture (matches model-table.md) ---
case $MODEL in
  5B)   NUM_LAYERS=24;  HIDDEN_SIZE=4096;  NUM_HEADS=32  ;;
  10B)  NUM_LAYERS=32;  HIDDEN_SIZE=5120;  NUM_HEADS=40  ;;
  20B)  NUM_LAYERS=32;  HIDDEN_SIZE=7168;  NUM_HEADS=56  ;;
  40B)  NUM_LAYERS=38;  HIDDEN_SIZE=9216;  NUM_HEADS=72  ;;
  60B)  NUM_LAYERS=56;  HIDDEN_SIZE=9216;  NUM_HEADS=72  ;;
  80B)  NUM_LAYERS=42;  HIDDEN_SIZE=12288; NUM_HEADS=96  ;;
  160B) NUM_LAYERS=84;  HIDDEN_SIZE=12288; NUM_HEADS=96  ;;
  320B) NUM_LAYERS=96;  HIDDEN_SIZE=16384; NUM_HEADS=128 ;;
  640B) NUM_LAYERS=192; HIDDEN_SIZE=16384; NUM_HEADS=128 ;;
  *) echo "Unknown model $MODEL"; exit 1 ;;
esac

# --- Parallelism config ---
# fsdp:    Gc=1, Gr=1, Gd=4*N  — pure depth across all GPUs
# fsdp_tp: Gc=2, Gr=2, Gd=N    — 2x2 intra-node NVLink TP + inter-node depth
case $MODE in
  fsdp)    CTP=1; RTP=1; DTP=$GPUS ;;
  fsdp_tp) CTP=2; RTP=2; DTP=$NNODES ;;
  *) echo "Unknown mode $MODE (use fsdp or fsdp_tp)"; exit 1 ;;
esac

# mbs must be divisible by dtp (AxoNN drop() shards batch across depth group)
if (( GBS % DTP != 0 )); then
  echo "ERROR: GBS=$GBS not divisible by DTP=$DTP. Minimum GBS for this config: $DTP"
  exit 1
fi
MBS=$(( GBS / DTP ))

echo "=================================================="
echo "SLURM job:   $SLURM_JOB_ID"
echo "Model:       $MODEL  (layers=$NUM_LAYERS hidden=$HIDDEN_SIZE heads=$NUM_HEADS)"
echo "Mode:        $MODE  (Gc=$CTP Gr=$RTP Gd=$DTP)"
echo "Sparsity:    $SPARSITY"
echo "Nodes:       $NNODES  GPUs: $GPUS"
echo "GBS=$GBS  MBS=$MBS  SEQ=$SEQ_LEN"
echo "Iters:       $TRAIN_ITERS  Seed: $SEED"
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

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0
"

srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 \
    ./examples/get_rank_from_slurm.sh \
    python -u pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl
