#!/bin/bash
# Submit all breakdown experiment combinations across GPU scales 8→64.
# Usage: bash launch_all.sh [--dry-run]
#
# For each node count in (2, 4, 8, 16) → (8, 16, 32, 64 GPUs):
#   1. g<N>_dense_nobd       — no pruning, dense AR, no breakdown logging
#   2. g<N>_dense_bd         — no pruning, dense AR, with breakdown logging
#   3. g<N>_sparse_noea_nobd — prune+sparse AR, no error accum, no breakdown logging
#   4. g<N>_sparse_noea_bd   — prune+sparse AR, no error accum, with breakdown logging
#   5. g<N>_sparse_ea_nobd   — prune+sparse AR, with error accum, no breakdown logging
#   6. g<N>_sparse_ea_bd     — prune+sparse AR, with error accum, with breakdown logging

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_JOB="$SCRIPT_DIR/base_job.sh"
LOG_DIR="/global/u1/e/egencer/scratch/sparsecomms/Megatron-AxoNN/logs"
MAX_JOBS=5
POLL_INTERVAL=60
mkdir -p "$LOG_DIR"

DRY_RUN=0
INTERACTIVE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=1 ;;
        --interactive) INTERACTIVE=1 ;;
    esac
done
if (( DRY_RUN ));    then echo "[dry-run] Would submit the following jobs:"; fi
if (( INTERACTIVE )); then echo "[interactive] Running directly via srun (no sbatch)"; fi

count_jobs() {
    squeue --me --noheader 2>/dev/null | wc -l
}

wait_for_slot() {
    while true; do
        local current
        current=$(count_jobs)
        if (( current < MAX_JOBS )); then
            return
        fi
        echo "  [$(date '+%H:%M:%S')] Queue full ($current / $MAX_JOBS), waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

submit() {
    local nnodes=$1
    local label=$2
    local do_prune=$3
    local do_error_accum=$4
    local do_breakdown=$5
    local run=$6
    local gpus=$(( nnodes * 4 ))
    local full_label="g${gpus}_${label}_r${run}"

    local sbatch_cmd=(
        sbatch
        --nodes=$nnodes
        --ntasks-per-node=4
        --gpus-per-node=4
        --constraint=gpu\&hbm80g
        --qos=premium
        --time=00:05:00
        --account=m5083_g
        --job-name="scl_${full_label}"
        --output="${LOG_DIR}/scl_${full_label}_%j.out"
        --error="${LOG_DIR}/scl_${full_label}_%j.err"
        --export="ALL,JOB_LABEL=${full_label},DO_PRUNE=${do_prune},DO_ERROR_ACCUM=${do_error_accum},DO_BREAKDOWN=${do_breakdown}"
        "$BASE_JOB"
    )

    if (( DRY_RUN )); then
        echo "  ${sbatch_cmd[*]}"
    elif (( INTERACTIVE )); then
        echo "Running interactively: label=$full_label  DO_PRUNE=$do_prune  DO_ERROR_ACCUM=$do_error_accum  DO_BREAKDOWN=$do_breakdown"
        JOB_LABEL=$full_label DO_PRUNE=$do_prune DO_ERROR_ACCUM=$do_error_accum DO_BREAKDOWN=$do_breakdown \
            bash "$BASE_JOB"
    else
        wait_for_slot
        local jid
        jid=$("${sbatch_cmd[@]}" | awk '{print $NF}')
        echo "Submitted $jid: nodes=$nnodes  label=$full_label  DO_PRUNE=$do_prune  DO_ERROR_ACCUM=$do_error_accum  DO_BREAKDOWN=$do_breakdown"
    fi
}

for RUN in 1 2 3; do
    echo ""
    echo "=== Run ${RUN} ==="
    for NNODES in 2 4 8 16; do
        GPUS=$(( NNODES * 4 ))
        echo ""
        echo "  --- ${GPUS} GPUs (${NNODES} nodes) ---"
        # submit $NNODES  dense_nobd        0  0  0  $RUN
        submit $NNODES  dense_bd          0  0  1  $RUN
        # submit $NNODES  sparse_noea_nobd  1  0  0  $RUN
        submit $NNODES  sparse_noea_bd    1  0  1  $RUN
        # submit $NNODES  sparse_ea_nobd    1  1  0  $RUN
        submit $NNODES  sparse_ea_bd      1  1  1  $RUN
    done
done

echo ""
echo "Done."
