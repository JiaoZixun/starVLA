"""DexYCB Wrist Dataset for Spatial Pretraining."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DexYCBWristDataset(Dataset):
    """DexYCB preprocessed wrist trajectory dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_transform=None,
        data_root_prefix: str = "/data1/jiaozx/starVLA",
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.data_root_prefix = Path(data_root_prefix)

        # Load preprocessed arrays
        self.current_poses = np.load(self.data_root / f"{split}_current_pose.npy")
        self.relative_waypoints = np.load(self.data_root / f"{split}_relative_waypoints.npy")
        self.future_valid = np.load(self.data_root / f"{split}_future_valid.npy")

        # Load metadata
        self.samples = []
        with open(self.data_root / f"{split}_samples.jsonl") as f:
            for line in f:
                self.samples.append(json.loads(line))

        assert len(self.samples) == len(self.current_poses)

        # Image transform
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.image_transform = image_transform

        print(f"DexYCB {split}: {len(self)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]

        # Load RGB image
        rgb_path = self.data_root_prefix / sample_meta["rgb_path"]
        image = Image.open(rgb_path).convert("RGB")
        image_tensor = self.image_transform(image)

        # Load geometry
        current_pose = torch.from_numpy(self.current_poses[idx]).float()  # [4, 4]
        relative_waypoints = torch.from_numpy(self.relative_waypoints[idx]).float()  # [4, 4, 4]
        future_valid = torch.from_numpy(self.future_valid[idx])  # [4]

        # Instruction (simple template for DexYCB)
        instruction = f"Predict future hand motion"

        # Gripper (not used for hand, set to open=1.0)
        future_gripper = torch.ones(4, dtype=torch.float32)
        future_gripper_valid = torch.zeros(4, dtype=torch.bool)  # Not supervised

        return {
            "image": image_tensor,
            "instruction": instruction,
            "current_pose": current_pose,
            "geometry_labels": {
                "current_pose": current_pose,
                "relative_waypoints": relative_waypoints,
                "future_valid": future_valid,
                "future_gripper": future_gripper,
                "future_gripper_valid": future_gripper_valid,
            },
            "action": torch.zeros(8, 7),  # Dummy, not used
            "meta": sample_meta,
        }
