#!/usr/bin/env python3
"""Fast DexYCB preprocessing for LIBERO-Spatial pretraining.

Extracts wrist SE(3) trajectories from DexYCB captures and generates
relative waypoint labels compatible with LIBERO's WaypointModel.

Output: ~300MB (geometry labels only, preserves RGB paths)
Processing time: ~30-60 min on 8 CPU workers
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


# Fixed camera (verify exists locally)
CAMERA_SERIAL = "836212060125"

# LIBERO-compatible future times (20Hz base)
FUTURE_SECONDS = np.array([0.05, 0.10, 0.20, 0.40], dtype=np.float32)

# DexYCB nominal framerate
SOURCE_FPS = 30.0

# SE(3) calibration matrices (from LIBERO-hand-pretrain reference)
# These align MANO root to robot TCP conventions
C_SIDE = {
    "right": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
    "left": np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32),
}


def load_sequence_meta(sequence_dir):
    """Load sequence metadata."""
    meta_path = sequence_dir / "meta.yml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return meta


def load_wrist_pose(label_path):
    """Extract wrist position and MANO root rotation from label file."""
    data = np.load(label_path)

    # Wrist position: joint_3d[hand_idx=0, joint_idx=0, xyz]
    wrist_pos = data["joint_3d"][0, 0, :]  # [3,] camera coords

    # MANO root rotation: pose_m[hand_idx=0, axis_angle_xyz]
    rotvec = data["pose_m"][0, :3]  # [3,] axis-angle

    return wrist_pos, rotvec


def rotvec_to_matrix(rotvec):
    """Convert axis-angle to rotation matrix."""
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_6d(R):
    """Convert 3x3 rotation matrix to 6D representation (first 2 columns)."""
    return R[:, :2].reshape(-1)  # [9] -> [6]


def relative_se3(T_curr, T_future):
    """Compute relative SE(3): T_curr^-1 @ T_future."""
    R_curr, p_curr = T_curr[:3, :3], T_curr[:3, 3]
    R_future, p_future = T_future[:3, :3], T_future[:3, 3]

    # Relative rotation: R_curr^T @ R_future
    R_rel = R_curr.T @ R_future

    # Relative translation: R_curr^T @ (p_future - p_curr)
    p_rel = R_curr.T @ (p_future - p_curr)

    T_rel = np.eye(4, dtype=np.float32)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = p_rel
    return T_rel


def interpolate_pose(pose_before, pose_after, alpha):
    """Interpolate SE(3) pose: SLERP for rotation, linear for translation."""
    R_before, p_before = pose_before[:3, :3], pose_before[:3, 3]
    R_after, p_after = pose_after[:3, :3], pose_after[:3, 3]

    # SLERP for rotation
    rot_before = Rotation.from_matrix(R_before)
    rot_after = Rotation.from_matrix(R_after)

    # Handle quaternion sign consistency
    quat_before = rot_before.as_quat()
    quat_after = rot_after.as_quat()
    if np.dot(quat_before, quat_after) < 0:
        quat_after = -quat_after
        rot_after = Rotation.from_quat(quat_after)

    slerp = Slerp([0, 1], Rotation.concatenate([rot_before, rot_after]))
    R_interp = slerp(alpha).as_matrix()

    # Linear interpolation for translation
    p_interp = (1 - alpha) * p_before + alpha * p_after

    T_interp = np.eye(4, dtype=np.float32)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = p_interp
    return T_interp


def process_sequence(args):
    """Process one DexYCB sequence to extract training samples."""
    sequence_dir, camera_serial, output_dir = args
    sequence_name = sequence_dir.name
    subject_id = sequence_name.split("-")[-1]

    try:
        # Load metadata
        meta = load_sequence_meta(sequence_dir)
        mano_side = meta.get("mano_sides", [None])[0]
        if mano_side not in ["left", "right"]:
            return []

        calib_matrix = C_SIDE[mano_side]

        # Find camera directory
        camera_dir = sequence_dir / camera_serial
        if not camera_dir.exists():
            return []

        # List all frames
        label_files = sorted(camera_dir.glob("labels_*.npz"))
        if len(label_files) == 0:
            return []

        # Extract frame indices
        frame_indices = [int(f.stem.split("_")[1]) for f in label_files]
        frame_indices = sorted(set(frame_indices))

        # Load all poses
        poses = {}
        for frame_idx in frame_indices:
            label_path = camera_dir / f"labels_{frame_idx:06d}.npz"
            rgb_path = camera_dir / f"color_{frame_idx:06d}.jpg"

            if not label_path.exists() or not rgb_path.exists():
                continue

            try:
                wrist_pos, rotvec = load_wrist_pose(label_path)

                # Check validity
                if not np.isfinite(wrist_pos).all() or not np.isfinite(rotvec).all():
                    continue
                if wrist_pos[2] <= 0:  # Invalid camera z
                    continue

                # Build SE(3) pose in camera frame
                R_raw = rotvec_to_matrix(rotvec)
                R_calibrated = R_raw @ calib_matrix  # Apply hand-side calibration

                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R_calibrated
                T[:3, 3] = wrist_pos

                poses[frame_idx] = {
                    "T": T,
                    "rgb_path": str(rgb_path.relative_to(output_dir.parent.parent.parent)),
                    "timestamp": frame_idx / SOURCE_FPS,
                }
            except Exception as e:
                continue

        if len(poses) < 20:  # Require minimum sequence length
            return []

        # Generate training samples
        samples = []
        frame_indices_valid = sorted(poses.keys())

        for i, curr_frame in enumerate(frame_indices_valid):
            curr_pose_data = poses[curr_frame]
            T_curr = curr_pose_data["T"]
            timestamp_curr = curr_pose_data["timestamp"]

            # Compute future frame positions for each tau
            future_frames = timestamp_curr + FUTURE_SECONDS
            future_targets = []
            future_valid = []

            for future_time in future_frames:
                # Find interpolation frames
                future_frame_float = future_time * SOURCE_FPS
                frame_before_idx = int(np.floor(future_frame_float))
                frame_after_idx = int(np.ceil(future_frame_float))

                # Check bounds
                if frame_after_idx not in poses:
                    future_targets.append(np.eye(4, dtype=np.float32))
                    future_valid.append(False)
                    continue

                # Interpolate pose
                if frame_before_idx == frame_after_idx:
                    T_future = poses[frame_after_idx]["T"]
                else:
                    if frame_before_idx not in poses:
                        future_targets.append(np.eye(4, dtype=np.float32))
                        future_valid.append(False)
                        continue

                    alpha = future_frame_float - frame_before_idx
                    T_future = interpolate_pose(
                        poses[frame_before_idx]["T"],
                        poses[frame_after_idx]["T"],
                        alpha
                    )

                # Compute relative SE(3)
                T_rel = relative_se3(T_curr, T_future)
                future_targets.append(T_rel)
                future_valid.append(True)

            # Skip if no future is valid
            if not any(future_valid):
                continue

            sample = {
                "sequence_id": sequence_name,
                "subject_id": subject_id,
                "camera_serial": camera_serial,
                "frame_index": curr_frame,
                "timestamp": timestamp_curr,
                "rgb_path": curr_pose_data["rgb_path"],
                "mano_side": mano_side,
                "current_pose": T_curr,
                "relative_waypoints": np.stack(future_targets, axis=0),  # [4, 4, 4]
                "future_valid": np.array(future_valid, dtype=bool),  # [4]
            }
            samples.append(sample)

        return samples

    except Exception as e:
        print(f"Error processing {sequence_name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data1/jiaozx/data/DexYCB_data")
    parser.add_argument("--output", type=str, default="/data1/jiaozx/starVLA/data/preprocessed_dexycb")
    parser.add_argument("--camera", type=str, default=CAMERA_SERIAL)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subject directories
    subject_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("202")])
    print(f"Found {len(subject_dirs)} subjects")

    # Prepare processing tasks
    tasks = [(seq_dir, args.camera, output_dir) for seq_dir in subject_dirs]

    # Parallel processing
    print(f"Processing {len(tasks)} sequences with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process_sequence, tasks), total=len(tasks)))

    # Flatten samples
    all_samples = []
    for samples in results:
        all_samples.extend(samples)

    print(f"Extracted {len(all_samples)} training samples")

    # Split by subject (8 train / 1 val / 1 test)
    subjects = sorted(set(s["subject_id"] for s in all_samples))
    assert len(subjects) == 10, f"Expected 10 subjects, got {len(subjects)}"

    train_subjects = subjects[:8]
    val_subjects = subjects[8:9]
    test_subjects = subjects[9:10]

    splits = {"train": train_subjects, "val": val_subjects, "test": test_subjects}

    # Save splits
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for split_name, split_subjects in splits.items():
        with open(split_dir / f"{split_name}_subjects.txt", "w") as f:
            f.write("\n".join(split_subjects))

    # Separate samples by split
    samples_by_split = {split: [] for split in splits}
    for sample in all_samples:
        for split_name, split_subjects in splits.items():
            if sample["subject_id"] in split_subjects:
                samples_by_split[split_name].append(sample)
                break

    # Save data for each split
    for split_name, samples in samples_by_split.items():
        if len(samples) == 0:
            continue

        print(f"\n{split_name}: {len(samples)} samples")

        # Prepare arrays
        current_poses = np.stack([s["current_pose"] for s in samples])  # [N, 4, 4]
        relative_waypoints = np.stack([s["relative_waypoints"] for s in samples])  # [N, 4, 4, 4]
        future_valid = np.stack([s["future_valid"] for s in samples])  # [N, 4]

        # Save binary arrays
        np.save(output_dir / f"{split_name}_current_pose.npy", current_poses)
        np.save(output_dir / f"{split_name}_relative_waypoints.npy", relative_waypoints)
        np.save(output_dir / f"{split_name}_future_valid.npy", future_valid)

        # Save JSONL metadata
        with open(output_dir / f"{split_name}_samples.jsonl", "w") as f:
            for sample in samples:
                meta = {
                    "sequence_id": sample["sequence_id"],
                    "subject_id": sample["subject_id"],
                    "camera_serial": sample["camera_serial"],
                    "frame_index": sample["frame_index"],
                    "timestamp": float(sample["timestamp"]),
                    "rgb_path": sample["rgb_path"],
                    "mano_side": sample["mano_side"],
                }
                f.write(json.dumps(meta) + "\n")

    # Save metadata
    metadata = {
        "source": "DexYCB",
        "camera_serial": args.camera,
        "source_fps": SOURCE_FPS,
        "future_seconds": FUTURE_SECONDS.tolist(),
        "calibration_matrices": {k: v.tolist() for k, v in C_SIDE.items()},
        "splits": splits,
        "total_samples": len(all_samples),
        "samples_per_split": {k: len(v) for k, v in samples_by_split.items()},
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Preprocessing complete!")
    print(f"Output: {output_dir}")
    print(f"Total size: ~{sum((output_dir / f).stat().st_size for f in output_dir.glob('*') if f.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
