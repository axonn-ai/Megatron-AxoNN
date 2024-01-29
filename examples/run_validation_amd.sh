#!/bin/bash
#SBATCH -N 2
#SBATCH --account=csc569
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=16
#SBATCH --time=02:00:00

# Runs the "128" parameter model for validation

#module load cray-python
#. /lustre/orion/scratch/ssingh37/csc547/venv_axonn_pt_2.1/bin/activate
#. /ccs/home/prajwal/venvs/axonnenv/bin/activate 
#module load amd-mixed/5.6.0 #this should match with the rocm version your pytorch uses

## these lines enable CUDA aware MPI
#module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"

## this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/prajwal/csc547/aws-ofi-rccl/build/lib"
#export NCCL_DEBUG=INFO
export FI_CXI_ATS=0

## this improves cross node bandwidth for some cases
export NCCL_CROSS_NIC=1

export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# data/checkpoint args
DATA_DIR="/lustre/orion/csc547/proj-shared/parallel_deep_learning/book_corpus"
SCRATCH_DIR="/lustre/orion/scratch/prajwal/csc547/MegatronAxonn"


CHECKPOINT_PATH="${SCRATCH_DIR}/validation/ground/checkpoints"
VOCAB_FILE="${DATA_DIR}/gpt2-vocab.json"
MERGE_FILE="${DATA_DIR}/gpt2-merges.txt"
DATA_PATH="${DATA_DIR}/BookCorpusDataset_text_document"
DATA_CACHE_PATH="${SCRATCH_DIR}/validation"

## ARCHITECTURE DETAILS
NUM_LAYERS=12
NUM_HEADS=12
HIDDEN_SIZE=768 #$(( 128 * NUM_HEADS ))

NSYS_PROFILE=False

## BATCH SIZES
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
SEQUENCE_LENGTH=2048

GPT_ARGS="
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    --seq-length ${SEQUENCE_LENGTH} \
    --max-position-embeddings ${SEQUENCE_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 6.0e-4 \
    --train-iters 800000 \
    --lr-decay-iters 600000 \
    --lr-decay-style linear \
    --min-lr 6.0e-5 \
    --lr-warmup-iters 750 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --use-amd \
    --use-flash-attn \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --save-interval 1000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-cache-path $DATA_CACHE_PATH
"
# --no-gradient-accumulation-fusion is neede on AMD
# --use-amd disables features incompatible with AMD
#

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 1
"



SCRIPT="python -u pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
"

if [[ ${NSYS_PROFILE} == "True" ]]
then
	echo "profiling with nsys"
	SCRIPT="nsys profile -s none \
		-t nvtx,cuda -o test.qdrep \
		--force-overwrite=true  \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		${SCRIPT} \
		--profile-step-start 5 \
		--profile-step-end 10 \
		--profile
		"
fi



export OMP_NUM_THREADS=7 
run_cmd="srun -N ${NNODES} -n ${GPUS} -c7 --gpus-per-task=1 --gpu-bind=closest ${SCRIPT}" 

echo ${run_cmd}
eval ${run_cmd}
set +x
