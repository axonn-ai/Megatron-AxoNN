#!/bin/bash
# Usage: ./run_gdb.sh <binary> [args...]
#
# Runs the given binary under gdb (batch mode) only for SLURM rank 0.
# All other ranks run the binary directly to avoid GDB startup overhead
# slowing down collective initialization (e.g. ncclCommInitRank).
#
# Output goes to gdb_rank${SLURM_PROCID}.log for rank 0.

EXEC="$1"
shift

RANK="${SLURM_PROCID:-0}"

if [ "$RANK" != "0" ]; then
    exec "$EXEC" "$@"
fi

LOGFILE="gdb_rank${RANK}.log"
echo "===== gdb for SLURM_PROCID=${RANK} =====" | tee "$LOGFILE"
echo "Running: $EXEC $*" | tee -a "$LOGFILE"

timeout 300 gdb -batch \
    -ex "set pagination off" \
    -ex "set print thread-events off" \
    -ex "handle SIGPIPE nostop noprint" \
    -ex "run" \
    -ex "bt full" \
    -ex "thread apply all bt" \
    -ex "quit" \
    --args "$EXEC" "$@" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}
[ $EXIT_CODE -eq 124 ] && echo "===== TIMEOUT =====" | tee -a "$LOGFILE"
exit $EXIT_CODE
