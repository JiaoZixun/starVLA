"""DexYCB wrist-geometry dataset with the same sample contract as the LIBERO geometry loader.

Sidecars come from ``examples/human2robots/DexYCBSpatial/prepare_dexycb_keypoints.py``:
``current_pose`` [N,4,4] (table frame, Panda grip_site axis convention),
``relative_waypoints`` [N,K,4,4] = R_t^T R_f / R_t^T (p_f - p_t), ``future_valid`` [N,K].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def collate_human_wrist_batch(batch):
    return batch


class HumanWristGeometryDataset(Dataset):
    def __init__(self, sidecar_root, split="train", waypoint_tau=(1, 2, 4, 8), image_size=224):
        self.split_dir = Path(sidecar_root).expanduser().resolve() / split
        metadata = json.loads((self.split_dir.parent / "metadata.json").read_text())
        if tuple(metadata["tau_slots"]) != tuple(int(t) for t in waypoint_tau):
            raise ValueError(f"Sidecar tau {metadata['tau_slots']} != configured waypoint_tau {list(waypoint_tau)}")
        self.records = [json.loads(line) for line in (self.split_dir / "samples.jsonl").read_text().splitlines() if line]
        self.current_pose = np.load(self.split_dir / "current_pose.npy", mmap_mode="r")
        self.relative_waypoints = np.load(self.split_dir / "relative_waypoints.npy", mmap_mode="r")
        self.future_valid = np.load(self.split_dir / "future_valid.npy", mmap_mode="r")
        count, k = len(self.records), len(metadata["tau_slots"])
        if (self.current_pose.shape, self.relative_waypoints.shape, self.future_valid.shape) != ((count, 4, 4), (count, k, 4, 4), (count, k)):
            raise ValueError(f"Inconsistent human sidecar arrays in {self.split_dir}")
        self.image_size = (int(image_size), int(image_size)) if np.isscalar(image_size) else tuple(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as image:
            # Same resize as the LIBERO loader (_pack_sample: Image.resize(image_size)).
            image = image.convert("RGB").resize(self.image_size)
        return {
            "domain": "human",
            "image": [image],
            "lang": record["lang"],
            "geometry_current": {"eef_pose": np.asarray(self.current_pose[index], dtype=np.float32).copy()},
            "geometry_labels": {
                "relative_waypoints": np.asarray(self.relative_waypoints[index], dtype=np.float32).copy(),
                "future_valid": np.asarray(self.future_valid[index], dtype=np.bool_).copy(),
            },
            "sequence_id": record["sequence_id"],
            "frame_index": int(record["frame_index"]),
        }
