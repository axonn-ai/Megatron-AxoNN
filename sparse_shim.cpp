/**
 * LD_PRELOAD shim for intercepting NCCL/NCCLX ReduceScatter calls.
 *
 * Build:
 *   gcc -fPIC -shared -O2 -o libnccl_rs_sparse_shim.so nccl_rs_sparse_shim.c -ldl -lcudart -I/path/to/nccl/include
 *
 * Usage:
 *   USE_SPARSE_RS=1 LD_PRELOAD=./libnccl_rs_sparse_shim.so ./your_program
 *   NCCL_RS_SHIM_TIMING=1 USE_SPARSE_RS=1 LD_PRELOAD=./libnccl_rs_sparse_shim.so ./your_program
 *   NCCL_RS_SHIM_STATS=1 NCCL_RS_SHIM_TIMING=1 USE_SPARSE_RS=1 LD_PRELOAD=./libnccl_rs_sparse_shim.so ./your_program
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <nccl.h>

/* Configuration (set once at init) */
static int g_use_sparse = 0;
static int g_timing = 0;
static int g_debug = 0;
static int g_stats_enabled = 0;

/* Function pointer type */
typedef ncclResult_t (*ncclReduceScatter_fn)(
    const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);

/* Real function pointers */
static ncclReduceScatter_fn real_ncclReduceScatter = NULL;
static ncclReduceScatter_fn real_ncclReduceScatterSparse = NULL;

/* Statistics */
static struct {
  unsigned long long n_calls;
  unsigned long long n_sparse;
  unsigned long long n_timed;
  unsigned long long n_skipped;
  double gpu_ms;
  double sparse_gpu_ms;
  double dense_gpu_ms;
} g_stats = {0};

/* Event ring buffer for async timing */
#define MAX_EVENTS 16
static cudaEvent_t g_ev_starts[MAX_EVENTS];
static cudaEvent_t g_ev_ends[MAX_EVENTS];
static int g_ev_is_sparse[MAX_EVENTS];
static int g_ev_head = 0;  /* next slot to write */
static int g_ev_tail = 0;  /* next slot to read/drain */
static int g_ev_count = 0; /* number of in-flight events */
static int g_ev_initialized = 0;

static void init_events(void) {
  if (g_ev_initialized) return;
  for (int i = 0; i < MAX_EVENTS; i++) {
    cudaEventCreate(&g_ev_starts[i]);
    cudaEventCreate(&g_ev_ends[i]);
  }
  g_ev_initialized = 1;
}

static void destroy_events(void) {
  if (!g_ev_initialized) return;
  for (int i = 0; i < MAX_EVENTS; i++) {
    if (g_ev_starts[i]) cudaEventDestroy(g_ev_starts[i]);
    if (g_ev_ends[i]) cudaEventDestroy(g_ev_ends[i]);
  }
  g_ev_initialized = 0;
}

/* Collect one completed event from tail if ready */
static void try_collect_one(void) {
  if (g_ev_count == 0) return;
  if (cudaEventQuery(g_ev_ends[g_ev_tail]) != cudaSuccess) return;

  float gpu_ms = 0;
  if (cudaEventElapsedTime(&gpu_ms, g_ev_starts[g_ev_tail], g_ev_ends[g_ev_tail]) == cudaSuccess) {
    g_stats.gpu_ms += gpu_ms;
    g_stats.n_timed++;
    if (g_ev_is_sparse[g_ev_tail]) {
      g_stats.sparse_gpu_ms += gpu_ms;
    } else {
      g_stats.dense_gpu_ms += gpu_ms;
    }
  }
  g_ev_tail = (g_ev_tail + 1) % MAX_EVENTS;
  g_ev_count--;
}

/* Drain all remaining events (blocking, for cleanup) */
static void drain_all(void) {
  while (g_ev_count > 0) {
    cudaEventSynchronize(g_ev_ends[g_ev_tail]);
    float gpu_ms = 0;
    if (cudaEventElapsedTime(&gpu_ms, g_ev_starts[g_ev_tail], g_ev_ends[g_ev_tail]) == cudaSuccess) {
      g_stats.gpu_ms += gpu_ms;
      g_stats.n_timed++;
      if (g_ev_is_sparse[g_ev_tail]) {
        g_stats.sparse_gpu_ms += gpu_ms;
      } else {
        g_stats.dense_gpu_ms += gpu_ms;
      }
    }
    g_ev_tail = (g_ev_tail + 1) % MAX_EVENTS;
    g_ev_count--;
  }
}

__attribute__((destructor))
static void print_stats(void) {
  /* Drain any remaining events */
  if (g_timing && g_ev_initialized) {
    drain_all();
    destroy_events();
  }

  if (!g_stats.n_calls) return;

  fprintf(stderr, "\n[NCCL_RS_SHIM] ===== Statistics =====\n");
  fprintf(stderr, "[NCCL_RS_SHIM] Calls: %llu (sparse: %llu, dense: %llu)\n",
          g_stats.n_calls, g_stats.n_sparse, g_stats.n_calls - g_stats.n_sparse);

  if (g_timing && g_stats.n_timed > 0) {
    fprintf(stderr, "[NCCL_RS_SHIM] Timed: %llu (skipped: %llu)\n",
            g_stats.n_timed, g_stats.n_skipped);
    fprintf(stderr, "[NCCL_RS_SHIM] Avg GPU: %.3f ms\n", g_stats.gpu_ms / g_stats.n_timed);

    /* For per-category avg, we need to track timed counts separately */
    unsigned long long n_sparse_timed = g_stats.n_timed - (g_stats.n_calls - g_stats.n_sparse - g_stats.n_skipped);
    unsigned long long n_dense_timed = g_stats.n_timed - n_sparse_timed;

    if (g_stats.n_sparse && g_stats.sparse_gpu_ms > 0) {
      fprintf(stderr, "[NCCL_RS_SHIM]   Sparse: %.3f ms (total %.1f ms)\n",
              g_stats.sparse_gpu_ms / g_stats.n_sparse, g_stats.sparse_gpu_ms);
    }
    unsigned long long n_dense = g_stats.n_calls - g_stats.n_sparse;
    if (n_dense && g_stats.dense_gpu_ms > 0) {
      fprintf(stderr, "[NCCL_RS_SHIM]   Dense:  %.3f ms (total %.1f ms)\n",
              g_stats.dense_gpu_ms / n_dense, g_stats.dense_gpu_ms);
    }
    if (g_stats.n_sparse && n_dense && g_stats.sparse_gpu_ms > 0 && g_stats.dense_gpu_ms > 0) {
      double speedup = (g_stats.dense_gpu_ms / n_dense) / (g_stats.sparse_gpu_ms / g_stats.n_sparse);
      fprintf(stderr, "[NCCL_RS_SHIM] Speedup: %.2fx\n", speedup);
    }
  }
  fprintf(stderr, "[NCCL_RS_SHIM] ====================\n\n");
}

static void __attribute__((constructor)) shim_init(void) {
  const char* v;
  v = getenv("USE_SPARSE_RS"); g_use_sparse = (v && v[0] == '1');
  v = getenv("NCCL_RS_SHIM_TIMING"); g_timing = (v && v[0] == '1');
  v = getenv("NCCL_RS_SHIM_DEBUG"); g_debug = (v && v[0] == '1');
  v = getenv("NCCL_RS_SHIM_STATS"); g_stats_enabled = (v && v[0] == '1');

  real_ncclReduceScatter = (ncclReduceScatter_fn)dlsym(RTLD_NEXT, "ncclReduceScatter");
  real_ncclReduceScatterSparse = (ncclReduceScatter_fn)dlsym(RTLD_NEXT, "ncclReduceScatterSparse");

  // if(real_ncclReduceScatterSparse) {
  //   printf("Shim started - Sparse NCCL RS found!\n");
  // } else {
  //   printf("Shim started - Sparse NCCL RS NOT found!\n"); 
  // }

  if (g_timing) {
    init_events();
  }

  if (g_debug) {
    fprintf(stderr, "[NCCL_RS_SHIM] sparse=%d timing=%d\n", g_use_sparse, g_timing);
  }
}

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream) {

  /* Fast path: both disabled - minimal overhead */
  if (!g_use_sparse && !g_timing && !g_stats_enabled) {
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  }

  int is_sparse = (g_use_sparse && real_ncclReduceScatterSparse);
  int will_time = 0;

  if (g_timing) {
    /* Try to collect one completed event (non-blocking) */
    try_collect_one();

    if (g_ev_count < MAX_EVENTS) {
      /* Have space, record start event */
      cudaEventRecord(g_ev_starts[g_ev_head], stream);
      will_time = 1;
    } else {
      /* Queue full, skip timing this call */
      if (g_debug) {
        fprintf(stderr, "[NCCL_RS_SHIM] WARNING: event queue full, skipping timing\n");
      }
      g_stats.n_skipped++;
    }
  }

  /* Execute the actual collective */
  ncclResult_t result;
  if (is_sparse) {
    result = real_ncclReduceScatterSparse(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  } else {
    result = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  }

  if (will_time) {
    /* Record end event and advance head */
    cudaEventRecord(g_ev_ends[g_ev_head], stream);
    g_ev_is_sparse[g_ev_head] = is_sparse;
    g_ev_head = (g_ev_head + 1) % MAX_EVENTS;
    g_ev_count++;
  }

  /* Update call stats */
  g_stats.n_calls++;
  if (is_sparse) g_stats.n_sparse++;

  if (g_debug && will_time) {
    fprintf(stderr, "[NCCL_RS_SHIM] %s count=%zu (queued, %d in flight)\n",
            is_sparse ? "SPARSE" : "DENSE", recvcount, g_ev_count);
  }

  return result;
}

ncclResult_t pncclReduceScatter(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  return ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t ncclReduceScatterSparse(
    const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  if (real_ncclReduceScatterSparse) {
    return real_ncclReduceScatterSparse(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  }
  if (real_ncclReduceScatter) {
    return real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  }
  return ncclSystemError;
}
