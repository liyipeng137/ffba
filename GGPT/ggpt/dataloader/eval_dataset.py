import torch 
import os
from ggpt.dataloader.base_dataset import BaseDataset
from glob import glob

class EvalDataset(BaseDataset):
    def __init__(self, data_dict, **kwargs):
        super().__init__(**kwargs)
        self.mode = 'val'
        self.data_dict = data_dict
        self.scene_list = []
        for dataset_name, folders in self.data_dict.items():
            for folder in glob(folders):
                scene_name = os.path.basename(folder.rstrip('/'))
                if os.path.isfile(os.path.join(folder, 'ff_outputs.bin')) \
                    and os.path.isfile(os.path.join(folder, 'sfm_dlt_outputs.bin')) \
                    and os.path.isfile(os.path.join(folder, 'gt.bin')):
                    self.scene_list.append( {'dataset_name': dataset_name,
                                             'scene_name': scene_name,
                                             'folder': folder} )
                else:
                    print(f"Warning: missing files for folder {scene_name} in dataset {dataset_name}, skipped.")   
                    continue 
        # print(f"EvalDataset: found {len(self.scene_list)} scenes for evaluation.")
    def __len__(self):
        return len(self.scene_list)
    
    def load_scene(self, idx):
        ff_data = torch.load(os.path.join(self.scene_list[idx]['folder'], 'ff_outputs.bin'), map_location='cpu')
        geo_data = torch.load(os.path.join(self.scene_list[idx]['folder'], 'sfm_dlt_outputs.bin'), map_location='cpu')
        gt_data = torch.load(os.path.join(self.scene_list[idx]['folder'], 'gt.bin'), map_location='cpu')
        scene = {
            'dataset_name': self.scene_list[idx]['dataset_name'],
            'scene_name': self.scene_list[idx]['scene_name'],
            'ff_pts': ff_data['points'].float(), # (N,H,W,3)
            'ff_conf': ff_data['points_conf'].float(), # (N,H,W)
            'ff_extrinsics': ff_data['extrinsics'].float(), # (N,4,4)
            'ff_intrinsics': ff_data['intrinsics'].float(), # (N,3,3)
            'geo_pts': geo_data['points'].float(), # (N,H,W,3)
            'geo_msks': geo_data['point_masks'], # (N,H,W)
            'geo_extrinsics': geo_data['extrinsics'].float() if 'extrinsics' in geo_data else None,
            'geo_intrinsics': geo_data['intrinsics'].float() if 'intrinsics' in geo_data else None,
            'gt_pts': gt_data['points'].float(),
            'gt_msks': gt_data['point_masks'],
            'gt_extrinsics': gt_data['extrinsics'].float() if 'extrinsics' in gt_data else None, # (N,4,4)
            'gt_intrinsics': gt_data['intrinsics'].float() if 'intrinsics' in gt_data else None, # (N,3,3)
            'images': ff_data['images_ff'].float(), # (N,H,W,3)
        }
        return scene 
