# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""
import os
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import core
from megatron.core import tensor_parallel, mpu
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

from axonn import axonn as ax
from axonn.intra_layer import clear_weights_cache
from megatron.global_vars import set_global_variables
from megatron import get_args
from megatron.initialize import _initialize_distributed, _set_random_seed, initialize_megatron 
from megatron.training import get_model, get_optimizer_param_scheduler
from apex.optimizers import FusedAdam as Adam
from megatron.data.data_samplers import build_pretraining_data_loader
import types
from megatron.core import tensor_parallel
from megatron.training import get_flops, get_params, get_mem
from axonn.intra_layer import optimize_communication
from functools import partial
from contextlib import nullcontext


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(output, label):
    loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), label)
    loss = torch.mean(loss)
    return loss

    # Check individual rank losses are not NaN prior to DP all-reduce.
    #args = get_args()
    #if args.check_for_nan_in_loss_and_grad:
    #    global_rank = torch.distributed.get_rank()
    #    assert not loss.isnan(), (
    #        f'Rank {global_rank}: found NaN in local forward loss calculation. '
    #        f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
    #    )
#
    # Reduce loss for logging.
    #averaged_loss = average_losses_across_data_parallel_group([loss])

    #return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def set_device_and_init_torch_dist():
    from mpi4py import MPI
    import os

    world_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    # assign a unique GPU to each MPI process on a node    
    local_rank = world_rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    # create a process group across all processes 
    torch.distributed.init_process_group(
                init_method = init_method, 
                backend="nccl",
                world_size=world_size,
                rank=world_rank
    )

    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_input_shape(self):
    args = get_args()
    return [args.seq_length, 
            core.utils.divide(args.micro_batch_size, args.depth_tensor_model_parallel_size), 
            core.utils.divide(args.hidden_size, args.column_tensor_model_parallel_size)]

def get_log(iteration, 
            iter_loss,
            elapsed_time_per_iteration, 
            learning_rate, 
            batch_size,
            loss_scale,
            grad_norm,
            num_zeros_in_grad,
            params_norm):
    args = get_args()
    log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
    log_string += ' consumed samples: {:12d} |'.format(
        args.consumed_train_samples)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
        elapsed_time_per_iteration * 1000.0)
    log_string += ' learning rate: {:.3E} |'.format(learning_rate)
    log_string += ' global batch size: {:5d} |'.format(batch_size)
    log_string += ' loss: {:.6E} |'.format(iter_loss)
    #for key in total_loss_dict:
    #    if key not in [advanced_iters_key, skipped_iters_key,
    #                   nan_iters_key]:
    #        avg = total_loss_dict[key].item() / \
    #              float(max(1, total_loss_dict[advanced_iters_key]))
    #        if avg > 0.0:
    #            log_string += ' {}: {:.6E} |'.format(key, avg)
    #        total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
    log_string += ' loss scale: {:.1f} |'.format(loss_scale)
    if grad_norm is not None:
        log_string += ' grad norm: {:.3f} |'.format(grad_norm)
    if num_zeros_in_grad is not None:
        log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
    if params_norm is not None:
        log_string += ' params norm: {:.3f} |'.format(params_norm)
    #log_string += ' number of skipped iterations: {:3d} |'.format(
    #    total_loss_dict[skipped_iters_key])
    #log_string += ' number of nan iterations: {:3d} |'.format(
    #    total_loss_dict[nan_iters_key])
    log_string += ' theoretical FLOP/s: {:.3f} TFLOP/s | '.format(get_flops(elapsed_time_per_iteration))
    log_string += ' model size: {:.3f} B params | '.format(get_params())
    log_string += ' memory used by tensors {:.3f} GB '.format(get_mem())
    return log_string

def get_context(model):
    args = get_args()
    if args.overlap_axonn_comm:
        ctx = partial(optimize_communication, 
                      overlap_all_reduce=True, 
                      overlap_reduce_scatter=args.overlap_axonn_reduce_scatter, 
                      cache_weights=args.cache_weights_in_depth_tensor_parallelism, 
                      overlap_all_gather=args.overlap_axonn_all_gather, 
                      model=model)
    else:
        ctx = nullcontext
    
    return ctx

if __name__ == "__main__":
    set_device_and_init_torch_dist()
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
  
    args = get_args()
    timers = get_timers()
    args.model_type = ModelType.encoder_or_decoder
    model = model_provider(pre_process=ax.config.inter_layer_parallel_rank == 0, 
                           post_process=ax.config.inter_layer_parallel_rank == ax.config.G_inter - 1).cuda()
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    
    model, optimizer = ax.register_model_and_optimizer(model, optimizer)
    lr_scheduler = get_optimizer_param_scheduler(optimizer)
    ax.register_loss_fn(loss_func)

    train_samples = args.global_batch_size * 30
    if mpu.get_tensor_model_parallel_rank() == 0:
        train_ds, _, _ = train_valid_test_datasets_provider((train_samples,0,0))
    else:
        train_ds = None
    
    if torch.distributed.get_rank() == 0:
        print(f"Length of training dataset = {len(train_ds)}")

    if mpu.get_tensor_model_parallel_rank() == 0:
        actual_micro_batch_size = args.micro_batch_size
        args.micro_batch_size = core.utils.divide(args.global_batch_size, ax.config.G_data)
        train_dataloader = build_pretraining_data_loader(train_ds, consumed_samples=0)
        train_iterator = iter(cyclic_iter(train_dataloader))
        args.micro_batch_size = actual_micro_batch_size
    else:
        train_iterator = None
    
    if torch.distributed.get_rank() == 0:
        print(f"Length of training dataloader = {len(train_dataloader)}")
        #x = next(train_iterator)
        #print(x['text'].shape)
    
    batch, labels, _, _, _ = get_batch(train_iterator)
  
    assert args.global_batch_size % (ax.config.G_data * args.micro_batch_size) == 0
    num_micro_batches = args.global_batch_size // ax.config.G_data // args.micro_batch_size
    model.get_input_shape = types.MethodType(get_input_shape, model)
    model.get_output_shape = types.MethodType(get_input_shape, model)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_loss_dict = {}
    args.consumed_samples = 0

    for iteration in range(args.train_iters):
        start_event.record()
        ctx = get_context(model)
        with ctx():
            iter_loss = ax.run_batch(batch, labels, num_microbatches=num_micro_batches, micro_batch_size=args.micro_batch_size) 
        optimizer.step()
        clear_weights_cache()
        lr_scheduler.step(increment=args.global_batch_size)
        args.consumed_samples += args.global_batch_size * args.seq_length
        end_event.record()
        torch.cuda.synchronize()
        time = start_event.elapsed_time(end_event) / 1000
        tflops = get_flops(time) 
        
        log = get_log(iteration, 
            iter_loss,
            elapsed_time_per_iteration=time, 
            learning_rate=optimizer.param_groups[0]['lr'], 
            batch_size=args.global_batch_size,
            loss_scale=ax.loss_scale,
            grad_norm=None,
            num_zeros_in_grad=None,
            params_norm=None)

        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            print(log)
