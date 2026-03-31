# ncclx_shim.py
#
# Python wrapper around the NCCLx C++ shim.
#
# Usage (in AxoNN or any parallel code):
#
#   from megatron.core.ncclx_shim import NCCLxComm
#
#   # Create once, keyed to a torch.distributed process group
#   comm = NCCLxComm(group=my_depth_group)
#
#   # Replace AxoNN's reduce_scatter call with:
#   comm.reduce_scatter(input_tensor, output_tensor)   # async on current stream
#   comm.all_gather(input_tensor, output_tensor)
#
# Requires:
#   - torch.distributed already init'd (used only for unique-ID broadcast)
#   - NCCLX_LIB_PATH env var (or the default path below) pointing to libnccl.so.2
#     that was built from the NCCLx source.
#   - LD_PRELOAD must NOT contain that same library (remove it from submit scripts
#     when using this shim; torch.distributed will use PyTorch's own libnccl).

import os
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load as _cpp_load

# ---------------------------------------------------------------------------
# Build / load the C++ extension (JIT, cached after first build)
# ---------------------------------------------------------------------------
_SHIM_DIR   = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR  = os.path.join(
    os.environ.get("SCRATCH", "/tmp"), "ncclx_shim_build"
)

_ext = _cpp_load(
    name="_ncclx_shim",
    sources=[os.path.join(_SHIM_DIR, "ncclx_shim.cpp")],
    extra_ldflags=["-ldl"],
    build_directory=_BUILD_DIR,
    verbose=False,
)

# ---------------------------------------------------------------------------
# Path to the NCCLx libnccl.so.2 (the dlmopen target, NOT LD_PRELOAD'd)
# ---------------------------------------------------------------------------
NCCLX_LIB_PATH: str = os.environ.get(
    "NCCLX_LIB_PATH",
    "/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/build/ncclx/lib/libnccl.so.2",
)

# ---------------------------------------------------------------------------
# dtype → NCCL datatype int  (stable NCCL 2.x enum values)
# ---------------------------------------------------------------------------
_DTYPE_TO_NCCL = {
    torch.float16:  6,   # ncclFloat16
    torch.float32:  7,   # ncclFloat32
    torch.float64:  8,   # ncclFloat64
    torch.bfloat16: 9,   # ncclBfloat16
    torch.int32:    2,   # ncclInt32
    torch.int64:    4,   # ncclInt64
}


class NCCLxComm:
    """
    An NCCLx communicator that mirrors a torch.distributed process group.

    The unique ID is exchanged via torch.distributed so no separate bootstrap
    is needed.  All collectives are issued on the current CUDA stream and return
    immediately (asynchronous w.r.t. the host); call
    ``torch.cuda.current_stream().synchronize()`` if you need a host barrier.
    """

    def __init__(self, group: "dist.ProcessGroup | None" = None):
        """
        Parameters
        ----------
        group : torch.distributed ProcessGroup or None
            The process group this communicator covers.  None means the default
            (world) group.
        """
        _ext.load_ncclx(NCCLX_LIB_PATH)

        self._group = group
        rank       = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        # ---- exchange unique ID via torch.distributed ----------------------
        # Rank 0 within the group generates it; others receive it.
        if rank == 0:
            uid_bytes  = _ext.get_unique_id()           # bytes, length 128
            uid_tensor = torch.frombuffer(bytearray(uid_bytes), dtype=torch.uint8).cuda()
        else:
            uid_tensor = torch.zeros(128, dtype=torch.uint8, device="cuda")

        # src must be the *global* rank of group-rank-0
        src_global = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast(uid_tensor, src=src_global, group=group)

        uid_bytes = bytes(uid_tensor.cpu().numpy())

        # ---- create NCCLx communicator -------------------------------------
        self._comm  = _ext.create_comm(uid_bytes, rank, world_size)
        self._rank  = rank
        self._size  = world_size

    # -----------------------------------------------------------------------
    # Collectives
    # -----------------------------------------------------------------------

    def reduce_scatter(
        self,
        input:  torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """
        ReduceScatter (sum) on the current CUDA stream.

        ``input``  shape: (..., N * world_size)  — full tensor on every rank
        ``output`` shape: (..., N)               — this rank's shard
        ``output.numel()`` must equal ``input.numel() // world_size``.
        """
        assert input.is_contiguous() and output.is_contiguous(), \
            "ncclx_shim: tensors must be contiguous"
        assert input.is_cuda and output.is_cuda, \
            "ncclx_shim: tensors must be on CUDA"
        dtype = _DTYPE_TO_NCCL.get(input.dtype)
        if dtype is None:
            raise ValueError(f"ncclx_shim: unsupported dtype {input.dtype}")
        stream = torch.cuda.current_stream().cuda_stream
        _ext.reduce_scatter(
            input.data_ptr(), output.data_ptr(),
            output.numel(), dtype, self._comm, stream,
        )

    def all_gather(
        self,
        input:  torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """
        AllGather on the current CUDA stream.

        ``input``  shape: (..., N)               — this rank's shard
        ``output`` shape: (..., N * world_size)  — full tensor gathered
        """
        assert input.is_contiguous() and output.is_contiguous(), \
            "ncclx_shim: tensors must be contiguous"
        assert input.is_cuda and output.is_cuda, \
            "ncclx_shim: tensors must be on CUDA"
        dtype = _DTYPE_TO_NCCL.get(input.dtype)
        if dtype is None:
            raise ValueError(f"ncclx_shim: unsupported dtype {input.dtype}")
        stream = torch.cuda.current_stream().cuda_stream
        _ext.all_gather(
            input.data_ptr(), output.data_ptr(),
            input.numel(), dtype, self._comm, stream,
        )

    def synchronize(self) -> None:
        """Block the host until all pending NCCLx operations on the current stream finish."""
        torch.cuda.current_stream().synchronize()

    def __del__(self):
        if hasattr(self, "_comm") and self._comm:
            try:
                _ext.destroy_comm(self._comm)
            except Exception:
                pass
            self._comm = 0
