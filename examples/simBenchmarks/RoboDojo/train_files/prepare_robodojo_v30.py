#!/usr/bin/env python3
"""Create a lightweight StarVLA adapter for RoboDojo LeRobot v3.0 data.

The adapter contains symlinks to the 112 GB data/video trees and adds the
metadata names expected by StarVLA's v3 loader. No dataset bytes are copied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


DEFAULT_RAW = "/home/data/robodojo_cache/robodojo_data_huggingface_lerobot_v3.0_repo/data/RoboDojo_lerobot_v30_video"
DEFAULT_ADAPTER = "/home/data/robodojo_cache/starvla_robodojo_v30/RoboDojo_lerobot_v30_video"

MODALITY = {
    "action": {
        "left_joints": {"start": 0, "end": 6, "original_key": "action"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
        "right_joints": {"start": 7, "end": 13, "original_key": "action"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
    },
    "state": {
        "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
        "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
        "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def _link_or_validate(src: Path, dst: Path) -> None:
    relative_target = Path(os.path.relpath(src, dst.parent))
    if dst.is_symlink():
        if dst.readlink() == relative_target:
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {dst}")
    dst.symlink_to(relative_target, target_is_directory=src.is_dir())


def prepare(raw: Path, adapter: Path) -> Path:
    required = [raw / "meta/info.json", raw / "meta/stats.json", raw / "meta/episodes", raw / "data", raw / "videos"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing v3 dataset entries: " + ", ".join(missing))
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "meta").mkdir(exist_ok=True)
    for name in tqdm(("data", "videos"), desc="Linking RoboDojo v3 trees"):
        _link_or_validate(raw / name, adapter / name)
    for name in ("info.json", "stats.json", "tasks.parquet", "episodes"):
        _link_or_validate(raw / "meta" / name, adapter / "meta" / name)
    (adapter / "meta/modality.json").write_text(json.dumps(MODALITY, indent=2) + "\n", encoding="utf-8")
    stats = json.loads((raw / "meta/stats.json").read_text(encoding="utf-8"))
    stat_names = ("min", "max", "mean", "std", "q01", "q99")
    statistics = {
        key: {name: stats[key][name] for name in stat_names}
        for key in ("action", "observation.state")
    }
    for key, values in statistics.items():
        if any(len(value) != 14 for value in values.values()):
            raise ValueError(f"Expected 14-dimensional statistics for {key}")
    payload = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": statistics,
    }
    stats_path = adapter / "meta/stats_gr00t.json"
    if not stats_path.exists() or json.loads(stats_path.read_text(encoding="utf-8")) != payload:
        temp_path = stats_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(stats_path)
    with (adapter / "meta/info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0 metadata, got {info.get('codebase_version')!r}")
    print(f"[RoboDojo] adapter ready: {adapter}")
    print(f"[RoboDojo] episodes={info.get('total_episodes')} frames={info.get('total_frames')}")
    return adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path(os.environ.get("RAW_ROBODOJO_DATA", DEFAULT_RAW)))
    adapter_default = Path(DEFAULT_ADAPTER)
    if os.environ.get("ADAPTER_DATA_ROOT"):
        adapter_default = Path(os.environ["ADAPTER_DATA_ROOT"]) / "RoboDojo_lerobot_v30_video"
    parser.add_argument("--adapter", type=Path, default=adapter_default)
    args = parser.parse_args()
    prepare(args.raw.expanduser().resolve(), args.adapter.expanduser().resolve())


if __name__ == "__main__":
    main()
