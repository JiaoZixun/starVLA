import json

import numpy as np
from PIL import Image

from starVLA.dataloader.human_wrist_dataset import HumanWristGeometryDataset


def test_human_wrist_dataset_reads_manifest_and_mmap_arrays(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(image_path)
    manifest = tmp_path / "h100.jsonl"
    manifest.write_text(json.dumps({"image": "frame.png", "sequence_id": "s0", "frame_index": 3}) + "\n")
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
    waypoints = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4, 4)).copy()
    np.save(tmp_path / "current_pose.npy", pose)
    np.save(tmp_path / "relative_waypoints.npy", waypoints)
    np.save(tmp_path / "position_valid.npy", np.array([[[1, 1, 0, 0], [0, 0, 0, 0]]], dtype=np.bool_))
    np.save(tmp_path / "rotation_valid.npy", np.array([[[1, 1, 0, 0], [0, 0, 0, 0]]], dtype=np.bool_))
    np.save(tmp_path / "effector_type.npy", np.array([[1, 0]], dtype=np.int64))

    sample = HumanWristGeometryDataset(tmp_path, manifest)[0]

    assert sample["domain"] == "human"
    assert sample["image"][0].size == (8, 6)
    assert sample["geometry_current"]["effector_pose"].shape == (2, 4, 4)
    assert sample["geometry_labels"]["relative_waypoints"].shape == (2, 4, 4, 4)
    assert sample["geometry_labels"]["position_valid"].tolist() == [[True, True, False, False], [False] * 4]
