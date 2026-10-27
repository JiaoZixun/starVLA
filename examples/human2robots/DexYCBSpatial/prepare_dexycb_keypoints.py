#!/usr/bin/env python3
"""Build DexYCB wrist SE(3) waypoint sidecars from the 21 annotated 3D keypoints.

Per frame, only ``joint_3d`` (21x3, camera frame, meters) is read:

* position  = wrist keypoint (joint 0)
* rotation  = hand semantic axes from keypoints, mapped onto the robosuite Panda
  ``grip_site`` (LIBERO ``robot0_eef``) axis convention:
      approach (TCP local +z)  <- wrist(0) -> middle MCP(9)
      closing  (finger_joint1 slide axis) <- little MCP(17) -> index MCP(5)
  MCP joints are direct children of the MANO root, so this frame follows the
  root rotation without running MANO.
* frame     = DexYCB apriltag table frame (z up, meters), the analogue of the
  LIBERO world frame (z up, meters).  Labels are relative, so the remaining
  fixed origin offset between the two worlds cancels in the targets.

Targets use the LIBERO sidecar definition: R_t^T R_f and R_t^T (p_f - p_t),
at 0.05/0.10/0.20/0.40 s (LIBERO tau=[1,2,4,8] at 20 Hz) from nominal 30 Hz.
Source RGB files are referenced in place, never copied.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

VIEW = "836212060125"
TAU_SLOTS = (1, 2, 4, 8)
FUTURE_SECONDS = (0.05, 0.10, 0.20, 0.40)
SOURCE_FPS = 30.0
SPLITS = {"train": tuple(range(1, 9)), "val": (9,), "test": (10,)}
LANG = "Predict the future motion of the end effector."
WRIST, INDEX_MCP, MIDDLE_MCP, LITTLE_MCP = 0, 5, 9, 17


def load_yaml(path: Path):
    with path.open() as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def _mjcf_rotation(element) -> Rotation:
    wxyz = np.fromstring(element.get("quat", "1 0 0 0"), dtype=np.float64, sep=" ")
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]])


def panda_tcp_basis(gripper_xml: Path) -> np.ndarray:
    """Semantic axes [closing, y, approach] as columns in the grip_site frame."""
    root = ET.parse(str(gripper_xml)).getroot()
    gripper = root.find("worldbody").find("body")
    eef = gripper.find(".//body[@name='eef']")
    finger = gripper.find(".//body[@name='leftfinger']")
    joint = finger.find("joint[@name='finger_joint1']")
    if eef is None or finger is None or joint is None or eef.find("site[@name='grip_site']") is None:
        raise ValueError(f"{gripper_xml} lacks eef/grip_site/leftfinger/finger_joint1")
    r_site, r_finger = _mjcf_rotation(eef), _mjcf_rotation(finger)
    axis = np.fromstring(joint.get("axis"), dtype=np.float64, sep=" ")
    x = r_site.inv().apply(r_finger.apply(axis / np.linalg.norm(axis)))
    z = r_site.inv().apply(r_finger.apply([0.0, 0.0, 1.0]))
    z -= x * np.dot(x, z)
    x, z = x / np.linalg.norm(x), z / np.linalg.norm(z)
    basis = np.column_stack((x, np.cross(z, x), z))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(basis), 1.0):
        raise AssertionError("Panda TCP basis is not a proper rotation")
    return basis


def hand_basis(joints: np.ndarray) -> np.ndarray:
    """Columns [closing, y, approach] of the hand in the keypoint frame; [N,21,3] -> [N,3,3]."""
    z = joints[:, MIDDLE_MCP] - joints[:, WRIST]
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    x = joints[:, INDEX_MCP] - joints[:, LITTLE_MCP]
    x -= z * np.sum(x * z, axis=-1, keepdims=True)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return np.stack((x, np.cross(z, x), z), axis=-1)


def table_from_camera(root: Path, extrinsics_name: str, view: str) -> np.ndarray:
    values = load_yaml(root / "calibration" / f"extrinsics_{extrinsics_name}" / "extrinsics.yml")["extrinsics"]
    to_h = lambda key: np.vstack((np.asarray(values[key], dtype=np.float64).reshape(3, 4), [0, 0, 0, 1]))
    return np.linalg.inv(to_h("apriltag")) @ to_h(view)


def process_sequence(job):
    sequence_dir, view, b_tcp, root = job
    meta = load_yaml(sequence_dir / "meta.yml")
    sides = meta.get("mano_sides", [])
    if len(sides) != 1:
        raise ValueError(f"{sequence_dir}: expected one annotated hand, got {sides}")
    count = int(meta["num_frames"])
    camera_dir = sequence_dir / view
    joints = np.full((count, 21, 3), np.nan, dtype=np.float64)
    joints_2d = np.full((count, 21, 2), np.nan, dtype=np.float64)
    for frame in range(count):
        with np.load(camera_dir / f"labels_{frame:06d}.npz") as label:
            joints[frame] = label["joint_3d"].reshape(21, 3)
            joints_2d[frame] = label["joint_2d"].reshape(21, 2)
        if not (camera_dir / f"color_{frame:06d}.jpg").is_file():
            raise FileNotFoundError(camera_dir / f"color_{frame:06d}.jpg")
    valid = np.isfinite(joints).all(axis=(1, 2)) & ~np.all(joints == -1.0, axis=(1, 2)) & (joints[:, WRIST, 2] > 0)
    t_table_cam = table_from_camera(root, meta["extrinsics"], view)
    poses = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
    if valid.any():
        # R_cam_E = B_hand(cam) @ B_tcp^T maps the TCP semantic axes onto the hand's.
        r_cam = hand_basis(joints[valid]) @ b_tcp.T
        poses[valid, :3, :3] = t_table_cam[:3, :3] @ r_cam
        poses[valid, :3, 3] = joints[valid, WRIST] @ t_table_cam[:3, :3].T + t_table_cam[:3, 3]
    return {
        "sequence_dir": str(sequence_dir), "subject": int(re.search(r"subject-(\d+)", sequence_dir.parent.name).group(1)),
        "side": sides[0], "poses": poses, "valid": valid,
        "wrist_cam": joints[:, WRIST], "wrist_2d": joints_2d[:, WRIST],
    }


def interpolate(poses, valid, frame_time):
    lower, upper = int(np.floor(frame_time + 1e-9)), int(np.ceil(frame_time - 1e-9))
    if upper >= len(valid) or not valid[lower] or not valid[upper]:
        return None
    alpha = frame_time - lower
    if upper == lower or alpha < 1e-8:
        return poses[lower]
    rotation = Slerp([0.0, 1.0], Rotation.from_matrix(poses[[lower, upper], :3, :3]))([alpha]).as_matrix()[0]
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = (1 - alpha) * poses[lower, :3, 3] + alpha * poses[upper, :3, 3]
    return out


def build_samples(sequence):
    poses, valid = sequence["poses"], sequence["valid"]
    seq_key = "/".join(Path(sequence["sequence_dir"]).parts[-2:])
    rows = []
    for frame in np.flatnonzero(valid):
        current = poses[frame]
        rel = np.broadcast_to(np.eye(4), (len(TAU_SLOTS), 4, 4)).copy()
        fut_valid = np.zeros(len(TAU_SLOTS), dtype=bool)
        for k, seconds in enumerate(FUTURE_SECONDS):
            future = interpolate(poses, valid, frame + seconds * SOURCE_FPS)
            if future is None:
                continue
            r_t = current[:3, :3].T
            rel[k, :3, :3] = r_t @ future[:3, :3]
            rel[k, :3, 3] = r_t @ (future[:3, 3] - current[:3, 3])
            fut_valid[k] = True
        if fut_valid.any():
            rows.append((seq_key, int(frame), current, rel, fut_valid))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexycb-root", type=Path, default=Path("/data1/jiaozx/data/DexYCB_data"))
    parser.add_argument("--output", type=Path, default=Path("playground/DerivedData/dexycb_keypoint_se3_v1"))
    parser.add_argument("--panda-gripper-xml", type=Path, default=Path(
        "/data1/jiaozx/anaconda3/envs/fastvla-fla/lib/python3.11/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"))
    parser.add_argument("--view", default=VIEW)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    root, output = args.dexycb_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    b_tcp = panda_tcp_basis(args.panda_gripper_xml)
    sequence_dirs = sorted(p.parent for p in root.glob("*-subject-*/*/meta.yml") if (p.parent / args.view).is_dir())
    print(f"{len(sequence_dirs)} sequences with view {args.view}; B_tcp columns=\n{b_tcp.round(6)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        sequences = list(pool.map(process_sequence, [(d, args.view, b_tcp, root) for d in sequence_dirs], chunksize=4))

    intrinsics = load_yaml(root / "calibration" / "intrinsics" / f"{args.view}_640x480.yml")["color"]
    fx, fy, ppx, ppy = (float(intrinsics[key]) for key in ("fx", "fy", "ppx", "ppy"))
    reproj = []
    for seq in sequences:
        w, uv = seq["wrist_cam"][seq["valid"]], seq["wrist_2d"][seq["valid"]]
        reproj.append(np.linalg.norm(np.stack((fx * w[:, 0] / w[:, 2] + ppx, fy * w[:, 1] / w[:, 2] + ppy), -1) - uv, axis=-1))
    reproj = np.concatenate(reproj)

    output.mkdir(parents=True)
    report = {"split_subjects": {k: list(v) for k, v in SPLITS.items()}}
    for split, subjects in SPLITS.items():
        rows = [row for seq in sequences if seq["subject"] in subjects for row in build_samples(seq)]
        seq_dirs = {"/".join(Path(s["sequence_dir"]).parts[-2:]): s["sequence_dir"] for s in sequences}
        split_dir = output / split
        split_dir.mkdir()
        current = np.stack([r[2] for r in rows]).astype(np.float32)
        rel = np.stack([r[3] for r in rows]).astype(np.float32)
        fut = np.stack([r[4] for r in rows])
        np.save(split_dir / "current_pose.npy", current)
        np.save(split_dir / "relative_waypoints.npy", rel)
        np.save(split_dir / "future_valid.npy", fut)
        with (split_dir / "samples.jsonl").open("w") as stream:
            for seq_key, frame, *_ in rows:
                stream.write(json.dumps({
                    "image": f"{seq_dirs[seq_key]}/{args.view}/color_{frame:06d}.jpg",
                    "sequence_id": seq_key, "frame_index": frame,
                    "timestamp": frame / SOURCE_FPS, "lang": LANG,
                }) + "\n")
        # Contract checks on every exported sample.
        r = current[:, :3, :3].astype(np.float64)
        ortho = np.abs(np.einsum("nji,njk->nik", r, r) - np.eye(3)).max()
        det = np.abs(np.linalg.det(r) - 1).max()
        cur64, rel64 = current.astype(np.float64), rel.astype(np.float64)
        recon = np.einsum("nij,nkjl->nkil", cur64, rel64)  # T_t @ dT_k
        index = {(seq_key, frame): i for i, (seq_key, frame, *_ ) in enumerate(rows)}
        closure = 0.0
        for i, (seq_key, frame, _, _, fv) in enumerate(rows[:: max(1, len(rows) // 5000)]):
            seq = next(s for s in sequences if s["sequence_dir"] == seq_dirs[seq_key])
            for k, seconds in enumerate(FUTURE_SECONDS):
                if fv[k]:
                    target = interpolate(seq["poses"], seq["valid"], frame + seconds * SOURCE_FPS)
                    closure = max(closure, np.abs(recon[index[(seq_key, frame)], k] - target).max())
        approach_z = current[:, 2, 2]
        report[split] = {
            "samples": len(rows), "sequences": len({row[0] for row in rows}),
            "future_valid_rate": fut.mean(0).round(4).tolist(),
            "max_orthonormal_error": float(ortho), "max_det_error": float(det),
            "max_closure_error_T_t_dT_vs_future": float(closure),
            "wrist_table_z_m": np.percentile(current[:, 2, 3], [1, 50, 99]).round(4).tolist(),
            "approach_axis_world_z": np.percentile(approach_z, [5, 50, 95]).round(3).tolist(),
            "translation_norm_m_per_tau": np.percentile(np.linalg.norm(rel[..., :3, 3], axis=-1)[fut].reshape(-1), [50, 95]).round(4).tolist(),
        }
        print(split, json.dumps(report[split]), flush=True)
    report.update({
        "source": str(root), "view": args.view, "fields_read": ["joint_3d (21x3, camera, meters)", "joint_2d (QC only)"],
        "position": "wrist keypoint 0", "rotation": "keypoint basis approach=0->9, closing=17->5 mapped to Panda grip_site axes",
        "B_tcp_columns": b_tcp.tolist(), "frame": "DexYCB apriltag table frame (z up, meters) via calibration extrinsics",
        "relative_label": "R_t^T R_f, R_t^T (p_f - p_t)  (identical to LIBERO geometry sidecar)",
        "tau_slots": list(TAU_SLOTS), "future_seconds": list(FUTURE_SECONDS), "source_fps_nominal": SOURCE_FPS,
        "interpolation": "linear translation + SLERP between adjacent valid frames only",
        "wrist_reprojection_px": np.percentile(reproj, [50, 95, 99]).round(3).tolist(),
        "valid_frames": int(sum(s["valid"].sum() for s in sequences)),
        "total_frames": int(sum(len(s["valid"]) for s in sequences)),
    })
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print("reprojection px p50/p95/p99", report["wrist_reprojection_px"], "valid", report["valid_frames"], "/", report["total_frames"])


if __name__ == "__main__":
    main()
