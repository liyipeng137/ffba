import torch 
from torch.utils.data import DataLoader, DistributedSampler
import hydra
from hydra.utils import instantiate

def collate_fn(batch):
    #Every batch has different number of points, so we cannot stack them directly
    return batch

def build_val_dataloader(cfg):
    val_dataset = instantiate(cfg.valdataset_configs)
    if cfg.common_config.ddp:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.common_config.num_workers, collate_fn=collate_fn, sampler=val_sampler, drop_last=False)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.common_config.num_workers, collate_fn=collate_fn, shuffle=False, drop_last=False)
    return val_dataloader