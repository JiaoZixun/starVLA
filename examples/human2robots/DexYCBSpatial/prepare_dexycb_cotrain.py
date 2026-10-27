#!/usr/bin/env python3
"""Build DexYCB v2 sidecars for human/robot co-training.

Same wrist SE(3) pose definition as ``prepare_dexycb_keypoints.py`` (v1), plus:

* time rescaling: slot k reads the pose ``FUTURE_SECONDS[k] / time_scale`` seconds ahead, so
  human wrist displacement per slot matches LIBERO (humans move ~1.5-1.9x faster at equal time);
* per-sequence language ``pick up the {object}`` from ``ycb_ids[ycb_grasp_ind]``;
* ``future_phase`` [N,K]: approach/grasp/transport at each slot (no release in DexYCB);
  holding starts when the grasped object has moved > ``lift_threshold`` m from its first pose;
* ``future_object`` [N,K,3]: grasped-object centre in the *current* wrist frame at each slot.

Phase windows are given in robot seconds and converted with the same time scale.
Source RGB files are referenced in place, never copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dexycb_keypoints import (  # noqa: E402
    FUTURE_SECONDS, SOURCE_FPS, SPLITS, TAU_SLOTS, VIEW, interpolate, load_yaml, panda_tcp_basis, process_sequence,
    table_from_camera,
)


def _load_phase_labels():
    # Load the shared module by path: importing the starVLA.dataloader package pulls in training deps.
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "starVLA" / "dataloader" / "phase_labels.py"
    spec = importlib.util.spec_from_file_location("starvla_phase_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_phase_labels = _load_phase_labels()
TRANSPORT, phases_from_holding = _phase_labels.TRANSPORT, _phase_labels.phases_from_holding

FORMAT = "starvla_dexycb_cotrain_sidecar_v2"
# Median |dp| (m) of LIBERO-10 at tau=1/2/4/8 (20 Hz), measured from libero_spatial_oft_v1 sidecars.
LIBERO_DP_MEDIAN = (0.0064, 0.0128, 0.0255, 0.0502)


def object_name(models: list[str], ycb_id: int) -> str:
    # DexYCB ycb_ids are 1-based indices into the sorted model directory, e.g. 002_master_chef_can.
    return models[ycb_id - 1].split("_", 1)[1].replace("_", " ")


def process_with_object(job):
    sequence_dir, view, b_tcp, root, models = job
    sequence = process_sequence((sequence_dir, view, b_tcp, root))
    meta = load_yaml(sequence_dir / "meta.yml")
    grasp_index = int(meta["ycb_grasp_ind"])
    count = int(meta["num_frames"])
    object_cam = np.full((count, 3), np.nan, dtype=np.float64)
    for frame in range(count):
        with np.load(sequence_dir / view / f"labels_{frame:06d}.npz") as label:
            pose = label["pose_y"][grasp_index]
        if np.any(pose):
            object_cam[frame] = pose[:3, 3]
    t_table_cam = table_from_camera(root, meta["extrinsics"], view)
    sequence["object_table"] = object_cam @ t_table_cam[:3, :3].T + t_table_cam[:3, 3]
    sequence["object_valid"] = np.isfinite(sequence["object_table"]).all(axis=1)
    sequence["object_name"] = object_name(models, int(meta["ycb_ids"][grasp_index]))
    return sequence


def interpolate_point(points, valid, frame_time):
    lower, upper = int(np.floor(frame_time + 1e-9)), int(np.ceil(frame_time - 1e-9))
    if upper >= len(valid) or not valid[lower] or not valid[upper]:
        return None
    alpha = frame_time - lower
    return (1 - alpha) * points[lower] + alpha * points[upper]


def holding_mask(sequence, lift_threshold: float) -> np.ndarray:
    """Holding from the first frame the object has moved > lift_threshold to the end (DexYCB only lifts)."""
    points, valid = sequence["object_table"], sequence["object_valid"]
    holding = np.zeros(len(valid), dtype=np.bool_)
    if not valid.any():
        return holding
    displacement = np.linalg.norm(points - points[np.flatnonzero(valid)[0]], axis=1)
    moved = np.flatnonzero(valid & (displacement > lift_threshold))
    if len(moved):
        holding[moved[0]:] = True
    return holding


def build_samples(sequence, time_scale: float, phase_window_frames: int, lift_threshold: float):
    poses, valid = sequence["poses"], sequence["valid"]
    seq_key = "/".join(Path(sequence["sequence_dir"]).parts[-2:])
    phase = phases_from_holding(holding_mask(sequence, lift_threshold), phase_window_frames)
    rows = []
    for frame in np.flatnonzero(valid):
        current = poses[frame]
        r_t = current[:3, :3].T
        rel = np.broadcast_to(np.eye(4), (len(TAU_SLOTS), 4, 4)).copy()
        fut_valid = np.zeros(len(TAU_SLOTS), dtype=bool)
        fut_phase = np.zeros(len(TAU_SLOTS), dtype=np.int64)
        fut_phase_valid = np.zeros(len(TAU_SLOTS), dtype=bool)
        fut_object = np.zeros((len(TAU_SLOTS), 3))
        fut_object_valid = np.zeros(len(TAU_SLOTS), dtype=bool)
        for k, seconds in enumerate(FUTURE_SECONDS):
            frame_time = frame + seconds / time_scale * SOURCE_FPS
            future = interpolate(poses, valid, frame_time)
            if future is not None:
                rel[k, :3, :3] = r_t @ future[:3, :3]
                rel[k, :3, 3] = r_t @ (future[:3, 3] - current[:3, 3])
                fut_valid[k] = True
            nearest = int(round(frame_time))
            if nearest < len(phase):
                fut_phase[k], fut_phase_valid[k] = phase[nearest], True
            point = interpolate_point(sequence["object_table"], sequence["object_valid"], frame_time)
            if point is not None:
                fut_object[k] = r_t @ (point - current[:3, 3])
                fut_object_valid[k] = True
        if fut_valid.any():
            rows.append((seq_key, int(frame), current, rel, fut_valid, fut_phase, fut_phase_valid, fut_object, fut_object_valid))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dexycb-root", type=Path, default=Path("/data1/jiaozx/data/DexYCB_data"))
    parser.add_argument("--output", type=Path, default=Path("playground/DerivedData/dexycb_cotrain_v2"))
    parser.add_argument("--panda-gripper-xml", type=Path, default=Path(
        "/data1/jiaozx/anaconda3/envs/fastvla-fla/lib/python3.11/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"))
    parser.add_argument("--view", default=VIEW)
    parser.add_argument("--time-scale", type=float, default=1.7,
                        help="human/robot speed ratio; slot k reads FUTURE_SECONDS[k]/time_scale s ahead")
    parser.add_argument("--phase-window-s", type=float, default=0.4, help="grasp/release window in robot seconds")
    parser.add_argument("--lift-threshold", type=float, default=0.01, help="object displacement (m) that starts holding")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    root, output = args.dexycb_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    phase_window_frames = int(round(args.phase_window_s / args.time_scale * SOURCE_FPS))
    b_tcp = panda_tcp_basis(args.panda_gripper_xml)
    models = sorted(os.listdir(root / "models"))
    sequence_dirs = sorted(p.parent for p in root.glob("*-subject-*/*/meta.yml") if (p.parent / args.view).is_dir())
    print(f"{len(sequence_dirs)} sequences; time_scale={args.time_scale}, phase window={phase_window_frames} frames", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [(d, args.view, b_tcp, root, models) for d in sequence_dirs]
        sequences = list(pool.map(process_with_object, jobs, chunksize=4))
    seq_dirs = {"/".join(Path(s["sequence_dir"]).parts[-2:]): s for s in sequences}

    output.mkdir(parents=True)
    report = {"split_subjects": {k: list(v) for k, v in SPLITS.items()}}
    for split, subjects in SPLITS.items():
        rows = [
            row for seq in sequences if seq["subject"] in subjects
            for row in build_samples(seq, args.time_scale, phase_window_frames, args.lift_threshold)
        ]
        split_dir = output / split
        split_dir.mkdir()
        arrays = {
            "current_pose": np.stack([r[2] for r in rows]).astype(np.float32),
            "relative_waypoints": np.stack([r[3] for r in rows]).astype(np.float32),
            "future_valid": np.stack([r[4] for r in rows]),
            "future_phase": np.stack([r[5] for r in rows]).astype(np.int64),
            "future_phase_valid": np.stack([r[6] for r in rows]),
            "future_object": np.stack([r[7] for r in rows]).astype(np.float32),
            "future_object_valid": np.stack([r[8] for r in rows]),
        }
        for name, value in arrays.items():
            np.save(split_dir / f"{name}.npy", value)
        with (split_dir / "samples.jsonl").open("w") as stream:
            for seq_key, frame, *_ in rows:
                seq = seq_dirs[seq_key]
                stream.write(json.dumps({
                    "image": f"{seq['sequence_dir']}/{args.view}/color_{frame:06d}.jpg",
                    "sequence_id": seq_key, "frame_index": frame, "timestamp": frame / SOURCE_FPS,
                    "lang": f"pick up the {seq['object_name']}", "object": seq["object_name"],
                }) + "\n")
        rel, fut = arrays["relative_waypoints"], arrays["future_valid"]
        dp = np.linalg.norm(rel[..., :3, 3], axis=-1)
        phase, phase_valid = arrays["future_phase"], arrays["future_phase_valid"]
        report[split] = {
            "samples": len(rows), "sequences": len({row[0] for row in rows}),
            "future_valid_rate": fut.mean(0).round(4).tolist(),
            "dp_median_m_per_slot": [round(float(np.median(dp[:, k][fut[:, k]])), 4) for k in range(len(TAU_SLOTS))],
            "libero_dp_median_m_per_slot": list(LIBERO_DP_MEDIAN),
            "phase_fraction_slot0": (
                np.bincount(phase[:, 0][phase_valid[:, 0]], minlength=4) / max(1, int(phase_valid[:, 0].sum()))
            ).round(4).tolist(),
            "transport_rate_slot0": round(float((phase[:, 0][phase_valid[:, 0]] == TRANSPORT).mean()), 4),
            "object_valid_rate": arrays["future_object_valid"].mean(0).round(4).tolist(),
            "object_distance_median_m_slot0": round(float(np.median(np.linalg.norm(
                arrays["future_object"][:, 0][arrays["future_object_valid"][:, 0]], axis=-1))), 4),
            "objects": sorted({seq_dirs[row[0]]["object_name"] for row in rows}),
        }
        print(split, json.dumps(report[split]), flush=True)
    report.update({
        "format": FORMAT, "source": str(root), "view": args.view,
        "position": "wrist keypoint 0", "rotation": "keypoint basis mapped to Panda grip_site axes (as v1)",
        "frame": "DexYCB apriltag table frame (z up, meters)",
        "relative_label": "R_t^T R_f, R_t^T (p_f - p_t)  (identical to LIBERO geometry sidecar)",
        "tau_slots": list(TAU_SLOTS), "future_seconds_robot": list(FUTURE_SECONDS),
        "time_scale": args.time_scale,
        "future_seconds_human": [round(s / args.time_scale, 6) for s in FUTURE_SECONDS],
        "source_fps_nominal": SOURCE_FPS,
        "language": "pick up the {object}, object = ycb_ids[ycb_grasp_ind]",
        "phases": ["approach", "grasp", "transport", "release"],
        "phase_rule": f"holding from first frame object moved > {args.lift_threshold} m; grasp = {phase_window_frames} frames before",
        "phase_window_s_robot": args.phase_window_s, "phase_window_frames_human": phase_window_frames,
        "object_label": "grasped object centre (pose_y[ycb_grasp_ind] translation) in the current wrist frame, meters",
    })
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
