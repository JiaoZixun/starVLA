"""Manifest-backed DexYCB wrist geometry dataset.

The preprocessing job writes one JSON object per sampled RGB frame to
``samples.jsonl`` and dense float32/bool arrays beside it. Arrays are mmap'd
so workers do not copy the full geometry corpus into memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def collate_human_wrist_batch(batch):
    """Keep human samples as a list, without importing the LIBERO data stack."""
    return batch


class HumanWristGeometryDataset(Dataset):
    def __init__(self, data_root_dir, sequence_manifest, geometry_sidecar_root=None, waypoint_tau=(1, 2, 4, 8), **kwargs):
        self.root = Path(data_root_dir).expanduser().resolve()
        self.sidecar = Path(geometry_sidecar_root or self.root).expanduser().resolve()
        manifest = Path(sequence_manifest)
        self.manifest_path = manifest if manifest.is_absolute() else self.sidecar / manifest
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Human geometry manifest does not exist: {self.manifest_path}")
        self.waypoint_tau = tuple(int(t) for t in waypoint_tau)
        if not self.waypoint_tau or tuple(sorted(set(self.waypoint_tau))) != self.waypoint_tau or self.waypoint_tau[0] < 1:
            raise ValueError(f"waypoint_tau must be strictly increasing positive offsets, got {self.waypoint_tau}")
        self.records = [json.loads(line) for line in self.manifest_path.read_text().splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Human geometry manifest is empty: {self.manifest_path}")
        self.arrays = {}
        for name, dtype, shape_tail in (
            ("current_pose", np.float32, (2, 4, 4)),
            ("relative_waypoints", np.float32, (2, len(self.waypoint_tau), 4, 4)),
            ("position_valid", np.bool_, (2, len(self.waypoint_tau))),
            ("rotation_valid", np.bool_, (2, len(self.waypoint_tau))),
            ("effector_type", np.int64, (2,)),
        ):
            path = self.manifest_path.parent / f"{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing human geometry array: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (len(self.records), *shape_tail) or array.dtype != dtype:
                raise ValueError(f"Invalid {name}: shape={array.shape}, dtype={array.dtype}; expected {(len(self.records), *shape_tail)}, {dtype}")
            self.arrays[name] = array
        if not np.isfinite(self.arrays["current_pose"]).all():
            raise ValueError("current_pose contains non-finite values")
        if not np.isfinite(self.arrays["relative_waypoints"]).all():
            raise ValueError("relative_waypoints contains non-finite values")
        side = self.arrays["effector_type"]
        if np.any((side < 0) | (side > 2)):
            raise ValueError("effector_type must use 0=empty, 1=left, 2=right")
        invalid_slots = side == 0
        if np.any(self.arrays["position_valid"] & invalid_slots[..., None]) or np.any(
            self.arrays["rotation_valid"] & invalid_slots[..., None]
        ):
            raise ValueError("Empty effector slots cannot carry geometry labels")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        return {
            "domain": "human",
            "image": [image],
            "lang": str(record.get("lang", "Predict the future motion of the visible hand.")),
            "sequence_id": record.get("sequence_id"),
            "frame_index": int(record.get("frame_index", index)),
            "geometry_current": {"effector_pose": np.asarray(self.arrays["current_pose"][index]).copy()},
            "effector_type": np.asarray(self.arrays["effector_type"][index]).copy(),
            "geometry_labels": {
                "relative_waypoints": np.asarray(self.arrays["relative_waypoints"][index]).copy(),
                "position_valid": np.asarray(self.arrays["position_valid"][index]).copy(),
                "rotation_valid": np.asarray(self.arrays["rotation_valid"][index]).copy(),
            },
        }
