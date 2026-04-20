import random, numpy as np, os
import torch
import torch.distributed as dist

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(batch, device):
    if type(batch) is list:
        batch = [move_to_device(b, device) for b in batch]
    elif type(batch) is dict:
        batch = {k:move_to_device(v, device) for k,v in batch.items()}
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    else:
        return batch
    return batch

def init_DDP():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device_id = rank % torch.cuda.device_count()
    return rank, device_id