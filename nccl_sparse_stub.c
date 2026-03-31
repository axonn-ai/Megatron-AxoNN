/**
 * nccl_sparse_stub.c — LD_PRELOAD shim that provides sparse NCCL symbols
 * and forwards every call to the corresponding stock dense collective.
 *
 * Build:
 *   gcc -fPIC -shared -O2 -o libnccl_sparse_stub.so nccl_sparse_stub.c \
 *       -I/path/to/nccl/include
 *
 * Usage:
 *   LD_PRELOAD=./libnccl_sparse_stub.so ./your_program
 */

#include <nccl.h>

ncclResult_t ncclAllGatherSparse(
    const void* sendbuff, void* recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream)
{
    return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t ncclAllReduceSparse(
    const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclReduceScatterSparse(
    const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}
