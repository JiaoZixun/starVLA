#!/usr/bin/env python3
"""Build DexYCB v3 sidecars for object grounding: two views, every object's position, hand closure.

Differences from v2 (``prepare_dexycb_cotrain.py``):

* two third-person views, mirroring LIBERO's two-image input: primary ``932122062010`` (31 deg down,
  0.67 m high, closest to LIBERO agentview) and secondary ``932122060857`` (55 deg down). The wrist
  pose comes from the primary view's ``joint_3d``; the low upward-looking ``836212060125`` is unused.
* ``object_wrist`` [N,M,3]: every object on the table (M <= 5, padded) in the *current* wrist frame,
  and ``object_uv`` [N,M,2] its centre in the primary image, normalised (x=col, y=row) to [0, 1].
  Training builds language counterfactuals from these: same image, "pick up the <other object>",
  grounding target = that object.
* ``future_closure`` [N,K]: hand closed at each slot = grasped object moved > ``closure_threshold``
  (3 mm) from its first pose, the same binary "closed" signal the robot gets from its commanded gripper.
* waypoints use the v2 time rescaling (slot k reads FUTURE_SECONDS[k] / time_scale seconds ahead).
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
from prepare_dexycb_cotrain import LIBERO_DP_MEDIAN, interpolate_point, object_name  # noqa: E402
from prepare_dexycb_keypoints import (  # noqa: E402
    FUTURE_SECONDS, SOURCE_FPS, SPLITS, TAU_SLOTS, interpolate, load_yaml, panda_tcp_basis, process_sequence,
    table_from_camera,
)

FORMAT = "starvla_dexycb_grounding_sidecar_v3"
PRIMARY, SECONDARY = "932122062010", "932122060857"
MAX_OBJECTS = 5
WIDTH, HEIGHT = 640, 480


def process(job):
    sequence_dir, b_tcp, root, models = job
    sequence = process_sequence((sequence_dir, PRIMARY, b_tcp, root))
    meta = load_yaml(sequence_dir / "meta.yml")
    count, ycb_ids = int(meta["num_frames"]), [int(i) for i in meta["ycb_ids"]]
    intrinsics = load_yaml(root / "calibration" / "intrinsics" / f"{PRIMARY}_{WIDTH}x{HEIGHT}.yml")["color"]
    fx, fy, ppx, ppy = (float(intrinsics[k]) for k in ("fx", "fy", "ppx", "ppy"))
    t_table_cam = table_from_camera(root, meta["extrinsics"], PRIMARY)
    objects_cam = np.full((count, len(ycb_ids), 3), np.nan)
    for frame in range(count):
        with np.load(sequence_dir / PRIMARY / f"labels_{frame:06d}.npz") as label:
            pose = label["pose_y"].reshape(len(ycb_ids), 3, 4)
        present = np.any(pose.reshape(len(ycb_ids), -1), axis=1)
        objects_cam[frame, present] = pose[present, :, 3]
        if not (sequence_dir / SECONDARY / f"color_{frame:06d}.jpg").is_file():
            raise FileNotFoundError(sequence_dir / SECONDARY / f"color_{frame:06d}.jpg")
    z = objects_cam[..., 2]
    uv = np.stack(((fx * objects_cam[..., 0] / z + ppx) / WIDTH, (fy * objects_cam[..., 1] / z + ppy) / HEIGHT), axis=-1)
    sequence.update(
        objects_table=objects_cam @ t_table_cam[:3, :3].T + t_table_cam[:3, 3],
        objects_uv=uv, objects_valid=np.isfinite(objects_cam).all(-1) & (z > 0),
        names=[object_name(models, i) for i in ycb_ids], grasp_index=int(meta["ycb_grasp_ind"]),
    )
    return sequence


def closure_mask(sequence, threshold: float) -> np.ndarray:
    g = sequence["grasp_index"]
    points, valid = sequence["objects_table"][:, g], sequence["objects_valid"][:, g]
    closed = np.zeros(len(valid), bool)
    if valid.any():
        moved = np.flatnonzero(valid & (np.linalg.norm(points - points[np.flatnonzero(valid)[0]], axis=1) > threshold))
        if len(moved):
            closed[moved[0]:] = True
    return closed


def build_samples(sequence, time_scale: float, threshold: float):
    poses, valid = sequence["poses"], sequence["valid"]
    closed = closure_mask(sequence, threshold)
    m = len(sequence["names"])
    rows = []
    for frame in np.flatnonzero(valid):
        current = poses[frame]
        r_t = current[:3, :3].T
        rel = np.broadcast_to(np.eye(4), (len(TAU_SLOTS), 4, 4)).copy()
        fut_valid = np.zeros(len(TAU_SLOTS), bool)
        closure = np.zeros(len(TAU_SLOTS), np.float32)
        closure_valid = np.zeros(len(TAU_SLOTS), bool)
        for k, seconds in enumerate(FUTURE_SECONDS):
            frame_time = frame + seconds / time_scale * SOURCE_FPS
            future = interpolate(poses, valid, frame_time)
            if future is not None:
                rel[k, :3, :3] = r_t @ future[:3, :3]
                rel[k, :3, 3] = r_t @ (future[:3, 3] - current[:3, 3])
                fut_valid[k] = True
            nearest = int(round(frame_time))
            if nearest < len(closed):
                closure[k], closure_valid[k] = float(closed[nearest]), True
        objects = np.zeros((MAX_OBJECTS, 3), np.float32)
        uv = np.zeros((MAX_OBJECTS, 2), np.float32)
        obj_valid = np.zeros(MAX_OBJECTS, bool)
        uv_valid = np.zeros(MAX_OBJECTS, bool)
        for i in range(m):
            point = interpolate_point(sequence["objects_table"][:, i], sequence["objects_valid"][:, i], float(frame))
            if point is not None:
                objects[i], obj_valid[i] = r_t @ (point - current[:3, 3]), True
                uv[i] = sequence["objects_uv"][frame, i]
                uv_valid[i] = bool(np.all((uv[i] >= 0) & (uv[i] <= 1)))
        if fut_valid.any() and obj_valid[sequence["grasp_index"]]:
            rows.append((int(frame), current, rel, fut_valid, closure, closure_valid, objects, obj_valid, uv, uv_valid))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dexycb-root", type=Path, default=Path("/data1/jiaozx/data/DexYCB_data"))
    parser.add_argument("--output", type=Path, default=Path("playground/DerivedData/dexycb_grounding_v3"))
    parser.add_argument("--panda-gripper-xml", type=Path, default=Path(
        "/data1/jiaozx/anaconda3/envs/fastvla-fla/lib/python3.11/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"))
    parser.add_argument("--time-scale", type=float, default=1.7)
    parser.add_argument("--closure-threshold", type=float, default=0.003, help="grasped-object displacement (m) = hand closed")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root, output = args.dexycb_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    b_tcp = panda_tcp_basis(args.panda_gripper_xml)
    models = sorted(os.listdir(root / "models"))
    sequence_dirs = sorted(p.parent for p in root.glob("*-subject-*/*/meta.yml") if (p.parent / PRIMARY).is_dir())
    print(f"{len(sequence_dirs)} sequences; views {PRIMARY} + {SECONDARY}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        sequences = list(pool.map(process, [(d, b_tcp, root, models) for d in sequence_dirs], chunksize=4))

    output.mkdir(parents=True)
    report = {"split_subjects": {k: list(v) for k, v in SPLITS.items()}}
    for split, subjects in SPLITS.items():
        records, arrays = [], {k: [] for k in ("current_pose", "relative_waypoints", "future_valid", "future_closure",
                                               "future_closure_valid", "object_wrist", "object_valid", "object_uv", "object_uv_valid")}
        for seq in sequences:
            if seq["subject"] not in subjects:
                continue
            seq_key = "/".join(Path(seq["sequence_dir"]).parts[-2:])
            for frame, *values in build_samples(seq, args.time_scale, args.closure_threshold):
                for key, value in zip(arrays, values):
                    arrays[key].append(value)
                records.append({
                    "image": f"{seq['sequence_dir']}/{PRIMARY}/color_{frame:06d}.jpg",
                    "image2": f"{seq['sequence_dir']}/{SECONDARY}/color_{frame:06d}.jpg",
                    "sequence_id": seq_key, "frame_index": frame, "objects": seq["names"], "grasp_index": seq["grasp_index"],
                    "lang": f"pick up the {seq['names'][seq['grasp_index']]}",
                })
        split_dir = output / split
        split_dir.mkdir()
        stacked = {k: np.stack(v) for k, v in arrays.items()}
        for key, value in stacked.items():
            np.save(split_dir / f"{key}.npy", value.astype(np.float32) if value.dtype == np.float64 else value)
        with (split_dir / "samples.jsonl").open("w") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in records)
        rel, fut = stacked["relative_waypoints"], stacked["future_valid"]
        dp = np.linalg.norm(rel[..., :3, 3], axis=-1)
        grasp = np.asarray([r["grasp_index"] for r in records])
        idx = np.arange(len(records))
        report[split] = {
            "samples": len(records), "sequences": len({r["sequence_id"] for r in records}),
            "dp_median_m_per_slot": [round(float(np.median(dp[:, k][fut[:, k]])), 4) for k in range(len(TAU_SLOTS))],
            "libero_dp_median_m_per_slot": list(LIBERO_DP_MEDIAN),
            "closed_rate_slot0": round(float(stacked["future_closure"][:, 0][stacked["future_closure_valid"][:, 0]].mean()), 4),
            "objects_per_sample": np.bincount(stacked["object_valid"].sum(1)).tolist(),
            "target_uv_valid_rate": round(float(stacked["object_uv_valid"][idx, grasp].mean()), 4),
            "target_distance_m_p50_p95": np.percentile(np.linalg.norm(stacked["object_wrist"][idx, grasp], axis=-1), [50, 95]).round(4).tolist(),
            "target_uv_p5_p95": np.percentile(stacked["object_uv"][idx, grasp], [5, 95], axis=0).round(3).tolist(),
        }
        print(split, json.dumps(report[split]), flush=True)
    report.update({
        "format": FORMAT, "source": str(root), "views": [PRIMARY, SECONDARY], "view": PRIMARY,
        "tau_slots": list(TAU_SLOTS), "future_seconds_robot": list(FUTURE_SECONDS), "time_scale": args.time_scale,
        "relative_label": "R_t^T R_f, R_t^T (p_f - p_t)  (identical to LIBERO geometry sidecar)",
        "object_label": "object centres (pose_y translation) in the current wrist frame (m); uv in primary image",
        "closure_rule": f"closed from the first frame the grasped object moved > {args.closure_threshold} m",
    })
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
