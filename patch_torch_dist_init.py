from torch import distributed
import torch.distributed.distributed_c10d as c10d

def monkeypatch_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


@monkeypatch_method(distributed)
def new_group(ranks=None, timeout=c10d.default_pg_timeout, backend=None, pg_options=None):
    #global _pg_group_ranks

    default_pg = c10d._get_default_group()
    default_backend, default_store = c10d._pg_map[default_pg]
    global_rank = default_pg.rank()
    global_world_size = default_pg.size()

    # Default to the same backend as the global process group
    # if the backend is not specified.
    if not backend:
        backend = default_backend

    # checks the input ranks
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise RuntimeError(
                "the new group's world size should be less or "
                "equal to the world size set by "
                "init_process_group"
            )
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise RuntimeError(
                    "The new group's rank should be within the "
                    "the world_size set by init_process_group"
                )
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    backend = c10d.Backend(backend)
    pg = c10d._new_process_group_helper(
        group_world_size,
        group_rank,
        ranks,
        backend,
        default_store,
        pg_options=pg_options,
        timeout=timeout,
    )

    # Create the global rank to group rank mapping

    c10d._pg_group_ranks[pg] = {
        global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
    }

    # barrier at the end to ensure that once we return from this method, all
    # process groups including global variables are updated correctly on all
    # ranks.
    if backend == c10d.Backend.MPI:
        # MPI doesn't have store.
        barrier()
    else:
        # Use store based barrier here since barrier() used a bunch of
        # default devices and messes up NCCL internal state.
        #_store_based_barrier(global_rank, default_store, timeout)
        # Set sequence numbers for gloo and nccl process groups.
        if pg != c10d.GroupMember.NON_GROUP_MEMBER and c10d.get_backend(pg) in [
            c10d.Backend.GLOO,
            c10d.Backend.NCCL,
        ]:
            pg._set_sequence_number_for_group()

    return pg

