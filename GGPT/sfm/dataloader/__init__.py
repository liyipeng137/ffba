import torch
from torch.utils.data import DataLoader
import torch.utils.data
from hydra.utils import instantiate

class Val_ComposedDataset(torch.utils.data.Dataset):
    def __init__(self, valdataset_configs):
        super().__init__()
        self.datasets, self.index_mapping = [], []
        for dataset_i, cfg in enumerate(valdataset_configs):
            dataset = instantiate(cfg)
            self.index_mapping.extend( [(dataset_i, i) for i in range(len(dataset))] )
            self.datasets.append(dataset)

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        dataset_i, sample_i = self.index_mapping[idx]
        batch = self.datasets[dataset_i][sample_i]
        return batch


def get_valComposedDataLoader(cfg):
    dataset = Val_ComposedDataset(cfg.valdataset_configs)
    dataset.training = False 
    if cfg.common_config.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, sampler=sampler, num_workers=cfg.common_config.num_workers, collate_fn=lambda x: x)
    else:
        dataloader = DataLoader(dataset, shuffle=False, num_workers=cfg.common_config.num_workers, collate_fn=lambda x: x)
    return dataloader
