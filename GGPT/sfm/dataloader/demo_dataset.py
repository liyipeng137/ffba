import numpy as np 
import os 
import torch 
import cv2
import json, hashlib
import torch.distributed as dist
import math

from .transform2D_utils import crop_image_depth_and_intrinsic_by_pp, resize_image_depth_and_intrinsic
import PIL

class DemoDataset(torch.utils.data.Dataset):
    def __init__(self,
        folders, name,
        **kwargs,
        ):
        super().__init__()
        from omegaconf import OmegaConf; 
        folders = OmegaConf.to_container(folders)
        self.folders = folders
        self.name = name
        if isinstance(self.folders, str):
            self.folders = {str(0):self.folders}
        elif isinstance(self.folders, dict):
            pass
        elif not isinstance(self.folders, list):
            self.folders = {str(i):folder for i, folder in enumerate(self.folders)}

    def __len__(self):
        return len(self.folders)
    
    def get_all_seqnames(self):
        return sorted(self.folders.keys())

    def __getitem__(self, idx):
        seq_name = list(self.folders.keys())[idx]
        folder = self.folders[seq_name]
        img_names = [img for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        images = []
        for img_name in img_names:
            img = PIL.Image.open(os.path.join(folder, img_name)).convert('RGB')
            img = torch.from_numpy(np.array(img)).float()/255.0
            images.append(img)
        batch = {"img_names": img_names,"seq_name":seq_name, "scene_name":self.name, "dataset_name":self.name, 'intr_convention': "opencv" }
        batch["images"] = torch.stack(images, dim=0)
        return batch