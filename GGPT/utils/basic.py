import random
import numpy as np
import torch
import torch.distributed as dist

def Print(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False