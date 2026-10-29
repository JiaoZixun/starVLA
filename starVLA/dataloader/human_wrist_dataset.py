"""DexYCB wrist-geometry dataset with the same sample contract as the LIBERO geometry loader.

Sidecars come from ``examples/human2robots/DexYCBSpatial/prepare_dexycb_keypoints.py`` (v1) or
``prepare_dexycb_cotrain.py`` (v2):
``current_pose`` [N,4,4] (table frame, Panda grip_site axis convention),
``relative_waypoints`` [N,K,4,4] = R_t^T R_f / R_t^T (p_f - p_t), ``future_valid`` [N,K].
v2 adds per-sequence language in ``samples.jsonl`` and, when present,
``future_phase`` [N,K] (+ ``future_phase_valid``) and ``future_object`` [N,K,3] (+ ``future_object_valid``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Optional v2 label arrays: (label key, valid key, trailing shape after [N,K]).
OPTIONAL_SLOT_LABELS = (
    ("future_phase", "future_phase_valid", ()),
    ("future_object", "future_object_valid", (3,)),
)


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
        self.optional_labels = {}
        for key, valid_key, trailing in OPTIONAL_SLOT_LABELS:
            if not (self.split_dir / f"{key}.npy").is_file():
                continue
            values = np.load(self.split_dir / f"{key}.npy", mmap_mode="r")
            valid = np.load(self.split_dir / f"{valid_key}.npy", mmap_mode="r")
            if values.shape != (count, k, *trailing) or valid.shape != (count, k):
                raise ValueError(f"Inconsistent {key} sidecar arrays in {self.split_dir}")
            self.optional_labels[key] = (valid_key, values, valid)
        self.image_size = (int(image_size), int(image_size)) if np.isscalar(image_size) else tuple(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as image:
            # Same resize as the LIBERO loader (_pack_sample: Image.resize(image_size)).
            image = image.convert("RGB").resize(self.image_size)
        labels = {
            "relative_waypoints": np.asarray(self.relative_waypoints[index], dtype=np.float32).copy(),
            "future_valid": np.asarray(self.future_valid[index], dtype=np.bool_).copy(),
        }
        for key, (valid_key, values, valid) in self.optional_labels.items():
            labels[key] = np.asarray(values[index]).copy()
            labels[valid_key] = np.asarray(valid[index], dtype=np.bool_).copy()
        return {
            "domain": "human",
            "image": [image],
            "lang": record["lang"],
            "geometry_current": {"eef_pose": np.asarray(self.current_pose[index], dtype=np.float32).copy()},
            "geometry_labels": labels,
            "sequence_id": record["sequence_id"],
            "frame_index": int(record["frame_index"]),
        }


def collate_human_grounding_batch(batch):
    """Each item is [real, counterfactual]; flatten so pairs sit in the same update."""
    return [example for pair in batch for example in pair]


class HumanGroundingDataset(Dataset):
    """DexYCB v3 (``prepare_dexycb_grounding.py``): two views, all object positions, hand closure.

    ``__getitem__`` returns a pair sharing the same (augmented) images:
    the real sample ("pick up the <grasped>", waypoints + closure + grounding on the grasped
    object) and a language counterfactual ("pick up the <other object on the table>", grounding
    only, on that object). The pair makes the instruction, not the scene, decide the target.
    Grounding slot 0 = target object; slot 1 (place goal) does not exist in DexYCB and is masked.
    """

    def __init__(self, sidecar_root, split="train", waypoint_tau=(1, 2, 4, 8), image_size=224, augment=None, counterfactual=True):
        from starVLA.dataloader.grounding_augment import GroundingAugment

        self.split_dir = Path(sidecar_root).expanduser().resolve() / split
        metadata = json.loads((self.split_dir.parent / "metadata.json").read_text())
        if metadata.get("format") != "starvla_dexycb_grounding_sidecar_v3":
            raise ValueError(f"Expected a v3 grounding sidecar in {self.split_dir.parent}")
        if tuple(metadata["tau_slots"]) != tuple(int(t) for t in waypoint_tau):
            raise ValueError(f"Sidecar tau {metadata['tau_slots']} != configured waypoint_tau {list(waypoint_tau)}")
        self.records = [json.loads(line) for line in (self.split_dir / "samples.jsonl").read_text().splitlines() if line]
        self.arrays = {
            key: np.load(self.split_dir / f"{key}.npy", mmap_mode="r")
            for key in ("current_pose", "relative_waypoints", "future_valid", "future_closure", "future_closure_valid",
                        "object_wrist", "object_valid", "object_uv", "object_uv_valid")
        }
        if any(len(value) != len(self.records) for value in self.arrays.values()):
            raise ValueError(f"Inconsistent human grounding arrays in {self.split_dir}")
        self.image_size = (int(image_size), int(image_size)) if np.isscalar(image_size) else tuple(image_size)
        self.augment = GroundingAugment(augment)
        self.counterfactual = bool(counterfactual)

    def __len__(self):
        return len(self.records)

    def _example(self, record, index, object_index, uv, uv_valid, images, real: bool):
        a = self.arrays
        k = a["future_valid"].shape[1]
        labels = {
            "relative_waypoints": np.asarray(a["relative_waypoints"][index], dtype=np.float32).copy(),
            "future_valid": np.asarray(a["future_valid"][index], dtype=np.bool_).copy() if real else np.zeros(k, np.bool_),
            "future_gripper": np.asarray(a["future_closure"][index], dtype=np.float32).copy(),
            "future_gripper_valid": np.asarray(a["future_closure_valid"][index], dtype=np.bool_).copy() if real else np.zeros(k, np.bool_),
            "object_xyz": np.stack((a["object_wrist"][index, object_index], np.zeros(3))).astype(np.float32),
            "object_xyz_valid": np.asarray([bool(a["object_valid"][index, object_index]), False]),
            "object_uv": np.stack((uv[object_index], np.zeros(2))).astype(np.float32),
            "object_uv_valid": np.asarray([bool(uv_valid[object_index]), False]),
        }
        return {
            "domain": "human", "image": images, "lang": f"pick up the {record['objects'][object_index]}",
            "geometry_current": {"eef_pose": np.asarray(a["current_pose"][index], dtype=np.float32).copy()},
            "geometry_labels": labels, "sequence_id": record["sequence_id"], "frame_index": int(record["frame_index"]),
            "counterfactual": not real,
        }

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        images = []
        for key in ("image", "image2"):
            with Image.open(record[key]) as image:
                images.append(image.convert("RGB").resize(self.image_size))
        uv = np.asarray(self.arrays["object_uv"][index], dtype=np.float32)
        uv_valid = np.asarray(self.arrays["object_uv_valid"][index], dtype=np.bool_)
        images, uv, uv_valid = self.augment(images, uv, uv_valid)
        grasp = int(record["grasp_index"])
        pair = [self._example(record, index, grasp, uv, uv_valid, images, real=True)]
        if self.counterfactual:
            others = [i for i in range(len(record["objects"])) if i != grasp and self.arrays["object_valid"][index, i]]
            if others:
                other = others[int(torch.randint(len(others), ()))]
                pair.append(self._example(record, index, other, uv, uv_valid, images, real=False))
        return pair
