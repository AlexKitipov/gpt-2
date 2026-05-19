import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed():
    """
    Initializes torch.distributed using environment variables set by torchrun.
    """
    if "RANK" not in os.environ:
        return None, None, None

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def wrap_ddp(model, local_rank):
    """
    Wraps the model in DistributedDataParallel.
    """
    return DDP(model, device_ids=[local_rank], output_device=local_rank)
