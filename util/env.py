import torch
import torch.distributed as dist
import numpy as np
import random
import logging


def logging_distributed_set(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    # builtin_print = __builtin__.print
    # def _print(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if is_master or force:
    #         builtin_print(*args, **kwargs)
    #
    # __builtin__.print = _print
    # logging.info = _print

    def _print(*args, **kwargs):
        pass

    if not is_master:
        __builtin__.print = _print
        logging.info = _print


def init_distributed_mode(args, gpu, ngpus_per_node, world_size, rank):
    assert (torch.cuda.is_available())
    torch.cuda.set_device(gpu)
    args.gpu = gpu
    args.dist_rank = rank * ngpus_per_node + gpu
    args.world_size = world_size
    args.rank = rank
    args.ngpus_per_node = ngpus_per_node
    if world_size == 1:
        args.distributed = False
        return

    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.dist_rank)
    # Initializes the default distributed process group, and this will also initialize the distributed package.
    # Using this function before calling any other methods in distributed.
    args.distributed = True


def init_seeds(distributed, seed=0):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    random.seed(seed)
    np.random.seed(seed)

    if distributed:
        torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs.
    else:
        torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
