import logging
import os

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel

logger = logging.getLogger(__name__)
rank = 0
world_size = 1


def init(args):
    global rank, world_size
    if args.ddp:
        assert args.rank is not None and args.world_size is not None
        rank = args.rank
        world_size = args.world_size
    if world_size == 1:
        return
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend=args.ddp_backend,
        init_method='file://' + os.path.abspath(args.rendezvous_file),
        world_size=world_size,
        rank=rank)
    logger.debug("Distributed rendezvous went well, rank %d/%d", rank, world_size)


def average(metrics, count=1.):
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def wrap(model):
    if world_size == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())


def barrier():
    if world_size > 1:
        torch.distributed.barrier()


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    if world_size == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        sampler = DistributedSampler(dataset)
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        return klass(dataset, *args, shuffle=shuffle)
