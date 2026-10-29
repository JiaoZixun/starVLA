#!/usr/bin/env python3
"""Build object-grounding sidecars for LIBERO: where the next manipulated object and its place target are.

Per frame t of every demo (aligned with the ``libero_spatial_oft_v1`` pose sidecar offsets):

* ``holding``: commanded gripper closed (LeRobot action[:, 6] < 0.5, 1 = open); holds shorter than
  ``--min-hold`` frames are dropped and open gaps shorter than ``--min-gap`` frames are merged.
* ``target_world``: interaction point of the object handled next. While holding, the TCP itself
  (the object moves with the gripper); before a hold, the TCP position at that hold's first frame,
  i.e. where the gripper closes on the object. Invalid after the last release.
* ``goal_world``: where that object is put: TCP at the hold's release frame (last frame if the demo
  ends while holding). Same validity as the target.
* ``*_uv``: the same points in the *training* agentview image, normalised to [0, 1] as (x=col, y=row).
  Projection uses the per-task robosuite camera matrix; LeRobot frames are the raw render rotated by
  180 degrees (verified below), so x = (W - 1 - u + 0.5) / W, y = (v + 0.5) / H for camera pixel (u, v).

These are TCP interaction points, not object centres: LeRobot demos store no object poses. The
grasp TCP lies within the gripper finger span (< ~4 cm) of the grasped object. Model-side labels are
expressed in the current end-effector frame at load time.

Audits (stored in each manifest): camera matrix identical across init states, projected object
centres vs their segmentation centroids, and the 180-degree orientation of the LeRobot frames.
Requires NVIDIA EGL (MUJOCO_GL=egl) and the LIBERO source tree on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "libero_spatial_no_noops_1.0.0_lerobot": "libero_spatial",
    "libero_object_no_noops_1.0.0_lerobot": "libero_object",
    "libero_goal_no_noops_1.0.0_lerobot": "libero_goal",
    "libero_10_no_noops_1.0.0_lerobot": "libero_10",
}
FORMAT = "starvla_libero_grounding_sidecar_v1"
RES = 256  # LeRobot LIBERO frames are 256x256 renders


def hold_segments(gripper_action: np.ndarray, min_hold: int, min_gap: int) -> list[tuple[int, int]]:
    """[(start, end)) of closed-gripper spans; end == T if the demo ends while holding."""
    closed = np.asarray(gripper_action) < 0.5
    edges = np.diff(np.r_[0, closed.astype(np.int8), 0])
    # Drop spurious short closures first, then bridge short re-grasp openings between real holds.
    spans = [(s, e) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if e - s >= min_hold]
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged]


def episode_points(eef: np.ndarray, segments: list[tuple[int, int]]):
    count = len(eef)
    target = np.zeros((count, 3), np.float32)
    goal = np.zeros((count, 3), np.float32)
    valid = np.zeros(count, bool)
    holding = np.zeros(count, bool)
    for start, end in segments:
        holding[start:end] = True
    cursor = 0
    for start, end in segments:
        release = eef[min(end, count - 1)]
        # Frames before this hold (since the previous release) point at its grasp point.
        target[cursor:start], goal[cursor:start], valid[cursor:start] = eef[start], release, True
        target[start:end], goal[start:end], valid[start:end] = eef[start:end], release, True
        cursor = end
    return target, goal, valid, holding


def to_training_uv(matrix: np.ndarray, points: np.ndarray, size: int = RES):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    depth = homogeneous[:, 2]
    u, v = homogeneous[:, 0] / depth, homogeneous[:, 1] / depth
    uv = np.stack(((size - 1 - u + 0.5) / size, (v + 0.5) / size), axis=-1).astype(np.float32)
    inside = (depth > 0) & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    return uv, inside


def camera_audit(suite_name: str, task_languages: list[str], lerobot_frames: dict[str, np.ndarray]):
    """Per task: camera matrix, its invariance, projection error and LeRobot frame orientation."""
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    suite = benchmark.get_benchmark_dict()[suite_name]()
    by_language = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    result = {}
    for language in task_languages:
        if language not in by_language:
            raise KeyError(f"LeRobot task {language!r} is not in benchmark {suite_name}")
        task_id = by_language[language]
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            camera_heights=RES, camera_widths=RES,
        )
        env.seed(0)
        env.reset()
        sim, inner = env.env.sim, env.env
        matrices, errors, orientation, per_point = [], [], {}, {}
        for k, init_state in enumerate(suite.get_task_init_states(task_id)[:3]):
            obs = env.set_init_state(init_state)
            for _ in range(5):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            matrices.append(get_camera_transform_matrix(sim, "agentview", RES, RES))
            if k:
                continue
            seg = sim.render(camera_name="agentview", height=RES, width=RES, segmentation=True)
            # Camera-matrix pixels index the vertically flipped render; training frames are the render
            # rotated by 180 degrees, i.e. that image mirrored left-right.
            seg_train = seg[::-1][:, ::-1]
            body = np.where(seg_train[..., 0] == mujoco.mjtObj.mjOBJ_GEOM, sim.model.geom_bodyid[seg_train[..., 1]], -1)
            names = np.array([sim.model.body_id2name(b) if b >= 0 else "" for b in range(-1, sim.model.nbody)], dtype=object)
            body_names = names[body + 1]
            # Distance from each projected point to the nearest pixel of its own segmentation mask
            # (0 when it lands on the object). Body origins of tall/flat objects sit at their base,
            # so the mask centroid is not a fair reference.
            checks = [(obj, sim.data.body_xpos[inner.obj_body_id[obj]], obj + "_") for obj in inner.objects_dict]
            checks.append(("tcp", obs["robot0_eef_pos"], "gripper0_"))
            for label, point, prefix in checks:
                rows, cols = np.nonzero(np.char.startswith(body_names.astype(str), prefix))
                if len(rows) < 30:  # occluded or tiny
                    continue
                uv, _ = to_training_uv(matrices[-1], np.asarray(point)[None])
                pixel = uv[0] * RES - 0.5
                if label == "tcp":
                    # The TCP lies between the fingers, inside the gripper silhouette's bounding box.
                    inside = cols.min() <= pixel[0] <= cols.max() and rows.min() <= pixel[1] <= rows.max()
                    per_point["tcp_inside_gripper_bbox"] = bool(inside)
                    if not inside:
                        raise AssertionError(f"{language}: TCP projects outside the gripper ({pixel}, bbox cols "
                                             f"{cols.min()}-{cols.max()} rows {rows.min()}-{rows.max()})")
                    continue
                errors.append(float(np.min(np.hypot(cols - pixel[0], rows - pixel[1]))))
                per_point[label] = round(errors[-1], 2)
            raw = obs["agentview_image"].astype(np.float32)
            frame = lerobot_frames[language].astype(np.float32)
            orientation = {"raw": float(np.abs(raw - frame).mean()), "rot180": float(np.abs(raw[::-1, ::-1] - frame).mean())}
        env.close()
        drift = max(float(np.abs(m - matrices[0]).max()) for m in matrices)
        result[language] = {
            "task_id": task_id, "camera_matrix": matrices[0].tolist(), "matrix_drift_across_inits": drift,
            "projection_to_own_mask_px": {"median": float(np.median(errors)), "max": float(np.max(errors)), "per_point": per_point},
            "lerobot_frame_mad": orientation,
        }
        if drift > 1e-6:
            raise AssertionError(f"{language}: agentview matrix changes across init states ({drift})")
        if not orientation["rot180"] < orientation["raw"]:
            raise AssertionError(f"{language}: LeRobot frame is not the 180-degree rotated render: {orientation}")
        # Occluded objects (e.g. behind the basket) can project onto their hidden base; require
        # most points on their own mask rather than all of them.
        if np.median(errors) > 1.0 or np.mean(np.asarray(errors) <= 3.0) < 0.75:
            raise AssertionError(f"{language}: projection misses its own mask: {result[language]['projection_to_own_mask_px']}")
        print(f"  task {task_id}: projection-to-mask px median {np.median(errors):.1f} max {np.max(errors):.1f}, frame MAD {orientation}", flush=True)
    return result


def first_frame(dataset_dir: Path, episode: int) -> np.ndarray:
    import av

    path = dataset_dir / f"videos/chunk-{episode // 1000:03d}/observation.images.image/episode_{episode:06d}.mp4"
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def build(dataset_dir: Path, pose_root: Path, output_root: Path, min_hold: int, min_gap: int) -> dict:
    name = dataset_dir.name
    pose_manifest = json.loads((pose_root / name / "manifest.json").read_text())
    poses = np.load(pose_root / name / "obs_eef_pose.npy", mmap_mode="r")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (dataset_dir / "meta/tasks.jsonl").read_text().splitlines() if l}
    ranges = sorted(((int(e), r["offset"], r["length"]) for e, r in pose_manifest["episode_index_to_sidecar_range"].items()), key=lambda x: x[1])
    total = pose_manifest["source_total_frames"]
    arrays = {
        "target_world": np.zeros((total, 3), np.float32), "goal_world": np.zeros((total, 3), np.float32),
        "point_valid": np.zeros(total, bool), "holding": np.zeros(total, bool), "task_index": np.zeros(total, np.int16),
    }
    first_episode_of_task, stats = {}, {"episodes": 0, "episodes_without_hold": 0, "holds": 0, "max_eef_state_vs_pose_m": 0.0}
    for episode, offset, length in ranges:
        frame = pd.read_parquet(dataset_dir / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet",
                                columns=["action", "observation.state", "task_index", "frame_index"])
        if len(frame) != length or not (frame["frame_index"].to_numpy() == np.arange(length)).all():
            raise ValueError(f"{name} episode {episode}: parquet frames do not match the pose sidecar")
        eef = np.asarray(poses[offset:offset + length, :3, 3], dtype=np.float64)
        state = np.stack(frame["observation.state"].to_numpy())[:, :3]
        stats["max_eef_state_vs_pose_m"] = max(stats["max_eef_state_vs_pose_m"], float(np.abs(state - eef).max()))
        segments = hold_segments(np.stack(frame["action"].to_numpy())[:, 6], min_hold, min_gap)
        target, goal, valid, holding = episode_points(eef, segments)
        sl = slice(offset, offset + length)
        arrays["target_world"][sl], arrays["goal_world"][sl], arrays["point_valid"][sl], arrays["holding"][sl] = target, goal, valid, holding
        task_index = int(frame["task_index"].iloc[0])
        arrays["task_index"][sl] = task_index
        first_episode_of_task.setdefault(task_index, episode)
        stats["episodes"] += 1
        stats["episodes_without_hold"] += int(not segments)
        stats["holds"] += len(segments)

    languages = [tasks[i] for i in sorted(first_episode_of_task)]
    audit = camera_audit(DATASETS[name], languages, {tasks[i]: first_frame(dataset_dir, e) for i, e in first_episode_of_task.items()})
    matrices = {i: np.asarray(audit[tasks[i]]["camera_matrix"]) for i in first_episode_of_task}
    for key in ("target", "goal"):
        uv = np.zeros((total, 2), np.float32)
        inside = np.zeros(total, bool)
        for task_index, matrix in matrices.items():
            rows = arrays["task_index"] == task_index
            uv[rows], inside[rows] = to_training_uv(matrix, arrays[f"{key}_world"][rows])
        arrays[f"{key}_uv"], arrays[f"{key}_uv_valid"] = uv, inside & arrays["point_valid"]

    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        np.save(target_dir / f"{key}.npy", value)
    valid = arrays["point_valid"]
    stats.update({
        "frames": total, "point_valid_rate": round(float(valid.mean()), 4), "holding_rate": round(float(arrays["holding"].mean()), 4),
        "target_uv_valid_rate": round(float(arrays["target_uv_valid"].mean()), 4),
        "goal_uv_valid_rate": round(float(arrays["goal_uv_valid"].mean()), 4),
        "target_to_eef_distance_m_p50_p95": np.percentile(np.linalg.norm(arrays["target_world"][valid] - np.asarray(poses[:, :3, 3])[valid], axis=-1), [50, 95]).round(4).tolist(),
    })
    if stats["max_eef_state_vs_pose_m"] > 1e-4:
        raise AssertionError(f"{name}: observation.state eef differs from pose sidecar ({stats['max_eef_state_vs_pose_m']})")
    manifest = {
        "format": FORMAT, "dataset_name": name, "dataset_revision": pose_manifest["dataset_revision"],
        "source_dataset_path": pose_manifest["source_dataset_path"], "source_total_frames": total,
        "pose_sidecar": str(pose_root / name), "episode_index_to_sidecar_range": pose_manifest["episode_index_to_sidecar_range"],
        "hold_rule": f"action[:,6] < 0.5; holds < {min_hold} frames dropped, gaps < {min_gap} frames merged",
        "point_definition": "target = TCP at next hold start (TCP itself while holding); goal = TCP at that hold's release",
        "uv_definition": "training agentview image (render rotated 180 deg), x=col, y=row, normalised to [0,1]",
        "stats": stats, "camera_audit": audit,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(name, json.dumps(stats), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("playground/Datasets/LEROBOT_LIBERO_DATA"))
    parser.add_argument("--pose-root", type=Path, default=Path("playground/DerivedData/libero_spatial_oft_v1"))
    parser.add_argument("--output", type=Path, default=Path("playground/DerivedData/libero_grounding_v1"))
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), help="default: all four")
    parser.add_argument("--min-hold", type=int, default=3)
    parser.add_argument("--min-gap", type=int, default=3)
    args = parser.parse_args()
    if os.getenv("MUJOCO_GL") != "egl":
        raise RuntimeError("Set MUJOCO_GL=egl (NVIDIA EGL rendering) before running the camera audit")
    for name in args.dataset or sorted(DATASETS):
        build((args.data_root / name).resolve(), args.pose_root.resolve(), args.output.resolve(), args.min_hold, args.min_gap)


if __name__ == "__main__":
    main()
