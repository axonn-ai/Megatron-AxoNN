#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --job-name=scl_all
#SBATCH --output=logs/scl_all_%j.out
#SBATCH --error=logs/scl_all_%j.err

# Run all scaling experiment combinations sequentially within a single large allocation.
# Requests 16 nodes (64 GPUs) — the largest config — and runs all scales within it.
# Smaller runs will under-utilize the allocation but avoid queue time for multiple jobs.
# Alternatively, use launch_all.sh to submit separate jobs per scale.

set -euo pipefail

: "${MEGATRON_HOME:?Set MEGATRON_HOME to Megatron-AxoNN root}"
mkdir -p "$MEGATRON_HOME/logs"

run_one() {
    local nnodes=$1 label=$2 do_prune=$3 do_error_accum=$4 do_breakdown=$5
    local gpus=$(( nnodes * 4 ))
    local full_label="g${gpus}_${label}"
    echo ""
    echo "====== RUN: $full_label ($(date)) ======"
    JOB_LABEL=$full_label NNODES=$nnodes DO_PRUNE=$do_prune DO_ERROR_ACCUM=$do_error_accum DO_BREAKDOWN=$do_breakdown \
        bash "$MEGATRON_HOME/breakdown_experiment_scaling_80gb/base_job.sh"
}

for NNODES in 2 4 8 16; do
    #        nnodes  label              prune  error_accum  breakdown
    run_one  $NNODES  dense_bd              0       0            1
    sleep 30
    run_one  $NNODES  sparse_noea_bd        1       0            1
    sleep 30
    run_one  $NNODES  sparse_ea_bd          1       1            1
    sleep 60
done

echo ""
echo "All runs complete."
