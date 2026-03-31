// ncclx_shim.cpp
//
// Loads libnccl.so.2 (the NCCLx build) into a fresh dlmopen linker namespace so its
// symbols never collide with PyTorch's built-in libnccl.  All NCCL calls go through
// explicit function pointers obtained via dlsym inside that namespace.
//
// Exposed to Python via pybind11:
//   load_ncclx(path)           -- call once before anything else
//   get_unique_id()            -> bytes[128]
//   create_comm(uid, rank, n)  -> uint64 handle
//   reduce_scatter(send, recv, recv_count, dtype_int, comm, stream)
//   all_gather(send, recv, send_count, dtype_int, comm, stream)
//   destroy_comm(comm)
//
// dtype_int values match NCCL enum: float16=6, float32=7, float64=8, bfloat16=9

#define _GNU_SOURCE
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Minimal NCCL ABI types (stable across NCCL 2.x; no nccl.h required)
// ---------------------------------------------------------------------------
#define NCCL_UNIQUE_ID_BYTES 128
struct NcclUniqueId { char internal[NCCL_UNIQUE_ID_BYTES]; };
typedef void*  NcclComm_t;
typedef int    NcclResult_t;

// ---------------------------------------------------------------------------
// Function pointer table
// ---------------------------------------------------------------------------
static void* g_handle = nullptr;

static NcclResult_t (*g_GetUniqueId)(NcclUniqueId*)                                          = nullptr;
static NcclResult_t (*g_CommInitRank)(NcclComm_t*, int, NcclUniqueId, int)                   = nullptr;
static NcclResult_t (*g_ReduceScatter)(const void*, void*, size_t, int, int, NcclComm_t, void*) = nullptr;
static NcclResult_t (*g_AllGather)(const void*, void*, size_t, int, NcclComm_t, void*)       = nullptr;
static NcclResult_t (*g_AllReduce)(const void*, void*, size_t, int, int, NcclComm_t, void*)  = nullptr;
static NcclResult_t (*g_CommDestroy)(NcclComm_t)                                              = nullptr;
static const char*  (*g_GetErrorString)(NcclResult_t)                                         = nullptr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void* load_sym(void* h, const char* name) {
    void* s = dlsym(h, name);
    if (!s) throw std::runtime_error(std::string("dlsym(") + name + "): " + dlerror());
    return s;
}

static void check(NcclResult_t r, const char* op) {
    if (r != 0) {
        std::string msg = std::string(op) + " returned ";
        if (g_GetErrorString) msg += g_GetErrorString(r);
        else                  msg += std::to_string(r);
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Load NCCLx into a brand-new linker namespace.  Idempotent.
void load_ncclx(const std::string& path) {
    if (g_handle) return;

    g_handle = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_handle)
        throw std::runtime_error("dlmopen(" + path + "): " + std::string(dlerror()));

    g_GetUniqueId   = (NcclResult_t(*)(NcclUniqueId*))                                          load_sym(g_handle, "ncclGetUniqueId");
    g_CommInitRank  = (NcclResult_t(*)(NcclComm_t*, int, NcclUniqueId, int))                    load_sym(g_handle, "ncclCommInitRank");
    g_ReduceScatter = (NcclResult_t(*)(const void*, void*, size_t, int, int, NcclComm_t, void*))load_sym(g_handle, "ncclReduceScatter");
    g_AllGather     = (NcclResult_t(*)(const void*, void*, size_t, int, NcclComm_t, void*))     load_sym(g_handle, "ncclAllGather");
    g_AllReduce     = (NcclResult_t(*)(const void*, void*, size_t, int, int, NcclComm_t, void*))load_sym(g_handle, "ncclAllReduce");
    g_CommDestroy   = (NcclResult_t(*)(NcclComm_t))                                              load_sym(g_handle, "ncclCommDestroy");
    g_GetErrorString= (const char* (*)(NcclResult_t))                                            load_sym(g_handle, "ncclGetErrorString");
}

// Generate a fresh unique ID on this rank (only rank 0 needs to call this).
py::bytes get_unique_id() {
    if (!g_handle) throw std::runtime_error("call load_ncclx() first");
    NcclUniqueId uid;
    check(g_GetUniqueId(&uid), "ncclGetUniqueId");
    return py::bytes(uid.internal, NCCL_UNIQUE_ID_BYTES);
}

// Create a communicator from a 128-byte unique-ID blob.
// Returns an opaque uint64 handle (the ncclComm_t pointer cast to int).
uint64_t create_comm(py::bytes uid_bytes, int rank, int size) {
    if (!g_handle) throw std::runtime_error("call load_ncclx() first");

    std::string s = uid_bytes;
    if ((int)s.size() != NCCL_UNIQUE_ID_BYTES)
        throw std::invalid_argument("uid must be exactly 128 bytes");

    NcclUniqueId uid;
    std::memcpy(uid.internal, s.data(), NCCL_UNIQUE_ID_BYTES);

    NcclComm_t comm = nullptr;
    check(g_CommInitRank(&comm, size, uid, rank), "ncclCommInitRank");
    return reinterpret_cast<uint64_t>(comm);
}

// ReduceScatter: sendbuf holds (recv_count * world_size) elements,
//                recvbuf receives recv_count elements for this rank.
// dtype_int: 6=float16, 7=float32, 8=float64, 9=bfloat16  (NCCL enum values)
// stream: cudaStream_t cast to uint64
void reduce_scatter(uint64_t sendbuf, uint64_t recvbuf, size_t recv_count,
                    int dtype_int, uint64_t comm_handle, uint64_t stream) {
    check(g_ReduceScatter(
        reinterpret_cast<const void*>(sendbuf),
        reinterpret_cast<void*>(recvbuf),
        recv_count, dtype_int, /*ncclSum=*/0,
        reinterpret_cast<NcclComm_t>(comm_handle),
        reinterpret_cast<void*>(stream)
    ), "ncclReduceScatter");
}

// AllGather: sendbuf holds send_count elements for this rank,
//            recvbuf receives (send_count * world_size) elements.
void all_gather(uint64_t sendbuf, uint64_t recvbuf, size_t send_count,
                int dtype_int, uint64_t comm_handle, uint64_t stream) {
    check(g_AllGather(
        reinterpret_cast<const void*>(sendbuf),
        reinterpret_cast<void*>(recvbuf),
        send_count, dtype_int,
        reinterpret_cast<NcclComm_t>(comm_handle),
        reinterpret_cast<void*>(stream)
    ), "ncclAllGather");
}

void destroy_comm(uint64_t comm_handle) {
    if (!comm_handle) return;
    check(g_CommDestroy(reinterpret_cast<NcclComm_t>(comm_handle)), "ncclCommDestroy");
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NCCLx shim: dlmopen-isolated NCCLx communicator for ReduceScatter/AllGather";
    m.def("load_ncclx",      &load_ncclx,      "Load NCCLx into a private linker namespace");
    m.def("get_unique_id",   &get_unique_id,   "Return 128-byte ncclUniqueId (rank 0 only)");
    m.def("create_comm",     &create_comm,     "Create comm from uid bytes, rank, world_size");
    m.def("reduce_scatter",  &reduce_scatter,  "ReduceScatter (sendbuf, recvbuf, recv_count, dtype_int, comm, stream)");
    m.def("all_gather",      &all_gather,      "AllGather    (sendbuf, recvbuf, send_count, dtype_int, comm, stream)");
    m.def("destroy_comm",    &destroy_comm,    "Destroy communicator");
}
