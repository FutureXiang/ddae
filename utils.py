import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# ===== Configs =====

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

def get_optimizer(parameters, opt, lr):
    if not hasattr(opt, 'optim'):
        return torch.optim.Adam(parameters, lr=lr)
    elif opt.optim == 'AdamW':
        return torch.optim.AdamW(parameters, **opt.optim_args, lr=lr)
    else:
        raise NotImplementedError()

# ===== Multi-GPU training =====

def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


def DataLoaderDDP(dataset, batch_size, shuffle=True):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,
    )
    return dataloader, sampler

def print0(*args, **kwargs):
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        print(*args, **kwargs)
