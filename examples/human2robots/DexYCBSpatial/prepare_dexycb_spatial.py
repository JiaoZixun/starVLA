#!/usr/bin/env python3
"""Build README-compatible DexYCB wrist-geometry sidecars.

The script references source RGB/labels in place and only writes manifests,
geometry arrays, and provenance metadata below ``--output``. It requires the
licensed MANO model files (MANO_RIGHT.pkl / MANO_LEFT.pkl) supplied separately.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp


VIEW = "836212060125"
TAU_SLOTS = (1, 2, 4, 8)  # Model/config waypoint slot IDs.
FUTURE_SECONDS = (0.05, 0.10, 0.20, 0.40)
SOURCE_FPS = 30.0
SEED = 42
IDENTITY = np.eye(4, dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_metadata_fingerprint(sequences, root: Path, view: str) -> str:
    """Fingerprint the local split/calibration metadata without reading RGBs."""
    paths = {sequence["sequence_dir"] / "meta.yml" for sequence in sequences}
    paths.update(sequence["betas_path"] for sequence in sequences)
    paths.add(root / "calibration" / "intrinsics" / f"{view}_640x480.yml")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_yaml(path: Path):
    # DexYCB intrinsics use a Python tuple tag for extrinsics.
    with path.open("r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def parse_subject_id(path: Path) -> int:
    match = re.search(r"subject-(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse DexYCB subject ID from {path}")
    return int(match.group(1))


def discover_sequences(root: Path, view: str):
    sequences = []
    for meta_path in sorted(root.glob("*-subject-*/**/meta.yml")):
        meta = load_yaml(meta_path)
        sequence_dir = meta_path.parent
        camera_dir = sequence_dir / view
        if view not in meta.get("serials", []) or not camera_dir.is_dir():
            continue
        subject_id = parse_subject_id(sequence_dir.parent)
        sequence_key = f"{sequence_dir.parent.name}/{sequence_dir.name}"
        color = {int(m.group(1)): p for p in camera_dir.glob("color_*.jpg")
                 if (m := re.fullmatch(r"color_(\d+)\.jpg", p.name))}
        labels = {int(m.group(1)): p for p in camera_dir.glob("labels_*.npz")
                  if (m := re.fullmatch(r"labels_(\d+)\.npz", p.name))}
        if len(color) != int(meta["num_frames"]):
            raise ValueError(f"{sequence_key}: camera {view} has {len(color)} RGB frames, expected {meta['num_frames']}")
        if set(color) != set(labels):
            raise ValueError(f"{sequence_key}: RGB/label frame indices differ; missing RGB labels={sorted(set(color)-set(labels))[:8]}, missing labels RGB={sorted(set(labels)-set(color))[:8]}")
        mano_sides = meta.get("mano_sides", [])
        mano_calib = meta.get("mano_calib", [])
        if len(mano_sides) != 1 or len(mano_calib) != 1:
            raise ValueError(f"{sequence_key}: expected one annotated hand, got sides={mano_sides}, calibration={mano_calib}")
        calibration_name = mano_calib[0]
        if not calibration_name.startswith("mano_"):
            calibration_name = f"mano_{calibration_name}"
        betas_path = root / "calibration" / calibration_name / "mano.yml"
        if not betas_path.is_file():
            raise FileNotFoundError(f"Missing DexYCB MANO shape calibration: {betas_path}")
        sequences.append({
            "sequence_key": sequence_key,
            "sequence_dir": sequence_dir,
            "camera_dir": camera_dir,
            "subject_id": subject_id,
            "num_frames": int(meta["num_frames"]),
            "color": color,
            "labels": labels,
            "side": mano_sides[0].lower(),
            "betas_path": betas_path,
        })
    if len(sequences) != 1000:
        raise ValueError(f"Expected 1000 DexYCB sequences with camera {view}; discovered {len(sequences)}")
    by_subject = defaultdict(list)
    for sequence in sequences:
        by_subject[sequence["subject_id"]].append(sequence)
    for subject_id, subject_sequences in by_subject.items():
        subject_sequences.sort(key=lambda x: x["sequence_dir"].name)
        if len(subject_sequences) != 100:
            raise ValueError(f"Subject {subject_id} has {len(subject_sequences)} sequences; expected 100")
    if set(by_subject) != set(range(1, 11)):
        raise ValueError(f"Expected DexYCB subjects 1..10, found {sorted(by_subject)}")
    return sequences


def choose_nested_sets(sequences, seed: int):
    sequences_by_subject = defaultdict(list)
    for sequence in sequences:
        sequences_by_subject[sequence["subject_id"]].append(sequence)
    rng = np.random.RandomState(seed)
    subject_order = list(range(1, 11))
    rng.shuffle(subject_order)
    counts_100 = {subject: 10 for subject in subject_order}
    counts_500 = {subject: 50 for subject in subject_order}
    ranked = {}
    for subject in range(1, 11):
        entries = sorted(sequences_by_subject[subject], key=lambda x: x["sequence_key"])
        local_rng = np.random.RandomState(seed + subject * 1009)
        order = local_rng.permutation(len(entries))
        ranked[subject] = [entries[int(index)] for index in order]
    h100 = [ranked[s][i] for s in range(1, 11) for i in range(counts_100[s])]
    h500 = [ranked[s][i] for s in range(1, 11) for i in range(counts_500[s])]
    h1000 = list(sequences)
    h100 = sorted(h100, key=lambda x: x["sequence_key"])
    h500 = sorted(h500, key=lambda x: x["sequence_key"])
    h1000 = sorted(h1000, key=lambda x: x["sequence_key"])
    if len(h100) != 100 or len(h500) != 500 or len(h1000) != 1000:
        raise AssertionError("Sequence group sizes are not 100 / 500 / 1000")
    keys100 = {x["sequence_key"] for x in h100}
    keys500 = {x["sequence_key"] for x in h500}
    if not keys100 <= keys500 or not keys500 <= {x["sequence_key"] for x in h1000}:
        raise AssertionError("Human sample groups are not nested")
    return {"h100": h100, "h500": h500, "h1000": h1000}, {
        "subject_order": subject_order,
        "h100_per_subject": counts_100,
        "h500_per_subject": counts_500,
    }


def load_mano_layers(mano_root: Path, sides):
    required = {"right" if side == "right" else "left" for side in sides}
    expected = {side: mano_root / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
                for side in required}
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "MANO package code is installed, but licensed model weights are missing. "
            "Pass --mano-root containing the required file(s): " + ", ".join(missing)
        )
    # Import only after the caller has selected the existing MANO-capable Conda env.
    import torch
    from manopth.manolayer import ManoLayer

    layers = {}
    for side in sorted(required):
        layer = ManoLayer(
            mano_root=str(mano_root), side=side, use_pca=True, ncomps=45,
            flat_hand_mean=False,
        ).cpu().eval()
        layers[side] = layer
    return layers, expected


def make_basis(joints):
    """Reference MANO wrist axes in root coordinates; axes are matrix columns."""
    joints = np.asarray(joints, dtype=np.float64)
    # DexYCB order: wrist=0, index MCP=5, middle MCP=9, little MCP=17.
    z = joints[9] - joints[0]
    x = joints[5] - joints[17]
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-8:
        raise ValueError("Degenerate MANO middle-MCP axis")
    z /= z_norm
    x -= z * np.dot(x, z)
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        raise ValueError("Degenerate MANO index-to-little axis")
    x /= x_norm
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    basis = np.column_stack((x, y, z))
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-5):
        raise ValueError("MANO reference basis is not a proper rotation")
    return basis


def _mjcf_quat(body):
    """Read an MJCF wxyz quaternion as a scipy Rotation."""
    value = np.fromstring(body.get("quat", "1 0 0 0"), dtype=np.float64, sep=" ")
    if value.shape != (4,):
        raise ValueError(f"Invalid MJCF quaternion on {body.get('name')}: {value}")
    return Rotation.from_quat(value[[1, 2, 3, 0]])


def panda_tcp_basis(gripper_xml: Path):
    """Derive operation axes in robosuite's actual Panda ``grip_site`` frame.

    The positive first finger slide axis sets x; the finger body's local +Z
    points from the palm toward the fingertips. Transform both axes from the
    checked gripper asset into its grip_site frame before orthogonalizing.
    """
    root = ET.parse(str(gripper_xml)).getroot()
    worldbody = root.find("worldbody")
    gripper_body = worldbody.find("body") if worldbody is not None else None
    if gripper_body is None:
        raise ValueError(f"No gripper root body in {gripper_xml}")
    eef_body = gripper_body.find(".//body[@name='eef']")
    finger_body = gripper_body.find(".//body[@name='leftfinger']")
    finger_joint = finger_body.find("joint[@name='finger_joint1']") if finger_body is not None else None
    if eef_body is None or finger_body is None or finger_joint is None:
        raise ValueError(f"Panda gripper XML is missing eef / leftfinger / finger_joint1: {gripper_xml}")

    # The grip_site frame is the eef body's frame. The common gripper-root
    # rotation cancels, leaving the relative body rotations below.
    r_site = _mjcf_quat(eef_body)
    r_finger = _mjcf_quat(finger_body)
    joint_axis = np.fromstring(finger_joint.get("axis", "0 0 1"), dtype=np.float64, sep=" ")
    if joint_axis.shape != (3,) or np.linalg.norm(joint_axis) < 1e-8:
        raise ValueError(f"Invalid finger_joint1 axis in {gripper_xml}: {joint_axis}")
    x = r_site.inv().apply(r_finger.apply(joint_axis / np.linalg.norm(joint_axis)))
    z = r_site.inv().apply(r_finger.apply(np.array([0.0, 0.0, 1.0])))
    x /= np.linalg.norm(x)
    z -= x * np.dot(x, z)
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-8:
        raise ValueError("Panda gripper opening and approach axes are degenerate")
    z /= z_norm
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    basis = np.column_stack((x, y, z))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(basis), 1.0):
        raise AssertionError("Panda TCP basis must be right-handed")
    return basis


def calibrate_hand_to_tcp(layers, robot_basis):
    calibration = {}
    for side, layer in layers.items():
        import torch
        with torch.no_grad():
            coeffs = torch.zeros((1, 48), dtype=torch.float32)
            betas = torch.zeros((1, 10), dtype=torch.float32)
            _, joints, _ = layer(coeffs, betas, torch.zeros((1, 3), dtype=torch.float32))
        # manopth joints are in mm. Scale is irrelevant for basis construction.
        hand_basis = make_basis(joints[0].cpu().numpy())
        c = hand_basis @ robot_basis.T
        if not np.isclose(np.linalg.det(c), 1.0, atol=1e-5):
            raise ValueError(f"{side} hand-to-TCP calibration is not a proper rotation")
        calibration[side] = {"B_hand": hand_basis, "B_panda_tcp": robot_basis, "C_hand": c}
    return calibration


def valid_pose(label):
    pose = np.asarray(label["pose_m"], dtype=np.float32).reshape(-1)
    joints = np.asarray(label["joint_3d"], dtype=np.float32).reshape(-1, 21, 3)[0]
    pose_valid = pose.shape == (51,) and np.isfinite(pose).all() and not np.all(pose == 0.0)
    wrist_valid = np.isfinite(joints[0]).all() and not np.all(joints[0] == -1.0) and joints[0, 2] > 0
    return bool(pose_valid and wrist_valid), pose, joints


def pose_from_label(pose_m, wrist_xyz, c_hand):
    pose_m = np.asarray(pose_m, dtype=np.float64).reshape(51)
    wrist_xyz = np.asarray(wrist_xyz, dtype=np.float64).reshape(3)
    # pose_m is stored in the same camera frame as joint_3d. The toolkit's
    # OpenGL flip is only for rendering and must not be applied to labels.
    rot_cv = Rotation.from_rotvec(pose_m[:3]).as_matrix()
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rot_cv @ c_hand
    aligned[:3, 3] = wrist_xyz
    return aligned.astype(np.float32)


def process_one_sequence(sequence, c_hand):
    """Extract source fields in frame order; invalid frames keep identity poses."""
    count = sequence["num_frames"]
    poses = np.broadcast_to(IDENTITY, (count, 2, 4, 4)).copy()
    valid = np.zeros(count, dtype=np.bool_)
    side_slot = 0 if sequence["side"] == "left" else 1
    for frame_index in range(count):
        with np.load(sequence["labels"][frame_index], allow_pickle=False) as label:
            is_valid, pose_m, joints = valid_pose(label)
            if not is_valid:
                continue
            poses[frame_index, side_slot] = pose_from_label(pose_m, joints[0], c_hand)
            valid[frame_index] = True
    return sequence["sequence_key"], {"poses": poses, "valid": valid, "side_slot": side_slot}


def process_all_frames(sequences, calibration, workers):
    """Read independent sequences concurrently to hide small-file I/O latency."""
    geometry = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(process_one_sequence, sequence, calibration[sequence["side"]]["C_hand"])
                for sequence in sequences]
        for completed, job in enumerate(as_completed(jobs), 1):
            key, value = job.result()
            geometry[key] = value
            if completed % 100 == 0:
                print(f"Decoded labels for {completed}/{len(sequences)} sequences", flush=True)
    return geometry


def interpolate_pose(sequence_geometry, frame_time: float, side_slot: int):
    """Linear translation + SO(3) SLERP at a fractional source-frame index."""
    poses = sequence_geometry["poses"]
    valid = sequence_geometry["valid"]
    lower = int(math.floor(frame_time + 1e-9))
    upper = int(math.ceil(frame_time - 1e-9))
    if lower < 0 or upper >= len(valid) or not valid[lower] or not valid[upper]:
        return None
    alpha = frame_time - lower
    if upper == lower or alpha <= 1e-8:
        return poses[lower, side_slot]
    if 1.0 - alpha <= 1e-8:
        return poses[upper, side_slot]
    pose0 = poses[lower, side_slot]
    pose1 = poses[upper, side_slot]
    rotation = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack((pose0[:3, :3], pose1[:3, :3]))),
    )([alpha]).as_matrix()[0]
    interpolated = np.eye(4, dtype=np.float32)
    interpolated[:3, :3] = rotation.astype(np.float32)
    interpolated[:3, 3] = ((1.0 - alpha) * pose0[:3, 3] + alpha * pose1[:3, 3]).astype(np.float32)
    return interpolated


def build_group_arrays(group, geometry, waypoint_slots, future_seconds):
    records = []
    poses = []
    waypoints = []
    pos_valid = []
    rot_valid = []
    effectors = []
    sample_meta = []
    for sequence in group:
        seq_geo = geometry[sequence["sequence_key"]]
        side_slot = seq_geo["side_slot"]
        for frame_index in range(sequence["num_frames"]):
            if not seq_geo["valid"][frame_index]:
                continue
            current = seq_geo["poses"][frame_index, side_slot]
            sample_wp = np.broadcast_to(IDENTITY, (2, len(waypoint_slots), 4, 4)).copy()
            sample_pos_valid = np.zeros((2, len(waypoint_slots)), dtype=np.bool_)
            sample_rot_valid = np.zeros((2, len(waypoint_slots)), dtype=np.bool_)
            for tau_index, delta_seconds in enumerate(future_seconds):
                future_frame = frame_index + delta_seconds * SOURCE_FPS
                future = interpolate_pose(seq_geo, future_frame, side_slot)
                if future is None:
                    continue
                delta = np.linalg.inv(current.astype(np.float64)) @ future.astype(np.float64)
                sample_wp[side_slot, tau_index] = delta.astype(np.float32)
                sample_pos_valid[side_slot, tau_index] = True
                sample_rot_valid[side_slot, tau_index] = True
            current_slots = np.broadcast_to(IDENTITY, (2, 4, 4)).copy()
            current_slots[side_slot] = current
            effectortype = np.zeros(2, dtype=np.int64)
            effectortype[side_slot] = 1 if side_slot == 0 else 2
            image_path = sequence["color"][frame_index].resolve()
            records.append({
                "image": str(image_path),
                "sequence_id": sequence["sequence_key"],
                "frame_index": int(frame_index),
                "lang": "Predict the future motion of the visible hand.",
            })
            poses.append(current_slots)
            waypoints.append(sample_wp)
            pos_valid.append(sample_pos_valid)
            rot_valid.append(sample_rot_valid)
            effectors.append(effectortype)
            sample_meta.append({
                "sequence_id": sequence["sequence_key"],
                "subject_id": sequence["subject_id"],
                "camera_id": VIEW,
                "frame_index": int(frame_index),
                "timestamp_seconds": float(frame_index / SOURCE_FPS),
                "fps_nominal": int(SOURCE_FPS),
                "hand_side": sequence["side"],
                "raw_image": str(image_path),
                "raw_label": str(sequence["labels"][frame_index].resolve()),
            })
    if not records:
        raise ValueError("Selected group contains no valid current-frame samples")
    return {
        "records": records,
        "sample_metadata": sample_meta,
        "current_pose": np.asarray(poses, dtype=np.float32),
        "relative_waypoints": np.asarray(waypoints, dtype=np.float32),
        "position_valid": np.asarray(pos_valid, dtype=np.bool_),
        "rotation_valid": np.asarray(rot_valid, dtype=np.bool_),
        "effector_type": np.asarray(effectors, dtype=np.int64),
    }


def validate_group(output_dir: Path, waypoint_slots):
    records_path = output_dir / "samples.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    expected = {
        "current_pose": (np.dtype(np.float32), (2, 4, 4)),
        "relative_waypoints": (np.dtype(np.float32), (2, len(waypoint_slots), 4, 4)),
        "position_valid": (np.dtype(np.bool_), (2, len(waypoint_slots))),
        "rotation_valid": (np.dtype(np.bool_), (2, len(waypoint_slots))),
        "effector_type": (np.dtype(np.int64), (2,)),
    }
    arrays = {}
    for name, (dtype, tail) in expected.items():
        array = np.load(output_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(records), *tail) or array.dtype != dtype:
            raise ValueError(f"{output_dir}/{name}.npy has {array.shape}/{array.dtype}; expected {(len(records), *tail)}/{dtype}")
        arrays[name] = array
    for start in range(0, len(records), 2048):
        end = min(start + 2048, len(records))
        current = arrays["current_pose"][start:end]
        target = arrays["relative_waypoints"][start:end]
        position = arrays["position_valid"][start:end]
        rotation = arrays["rotation_valid"][start:end]
        effector = arrays["effector_type"][start:end]
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite transform in {output_dir}, rows {start}:{end}")
        if np.any((effector < 0) | (effector > 2)):
            raise ValueError(f"Invalid effector type in {output_dir}, rows {start}:{end}")
        empty = effector == 0
        if np.any(position[empty]) or np.any(rotation[empty]):
            raise ValueError(f"Empty effector has valid target in {output_dir}, rows {start}:{end}")
        no_targets = ~(position | rotation)
        if np.any(np.abs(target[no_targets] - IDENTITY) > 1e-6):
            raise ValueError(f"Invalid target is not finite identity in {output_dir}, rows {start}:{end}")
    return {"num_samples": len(records), "array_shapes": {k: list(v.shape) for k, v in arrays.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexycb-root", type=Path, default=Path("/home/data/DexYCB_data"))
    parser.add_argument("--output", type=Path, default=Path("playground/DerivedData/dexycb_spatial_v1"))
    parser.add_argument("--mano-root", type=Path, default=None,
                        help="Directory containing licensed MANO_RIGHT.pkl / MANO_LEFT.pkl")
    parser.add_argument(
        "--panda-gripper-xml", type=Path,
        default=Path("/home/data/Libero/envs/libero-baselines/lib/python3.10/site-packages/robosuite/models/assets/grippers/panda_gripper.xml"),
        help="Panda gripper XML used by the LIBERO robosuite environment",
    )
    parser.add_argument("--view", default=VIEW)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent sequence readers for compressed per-frame label files")
    parser.add_argument("--check-only", action="store_true",
                        help="Audit source sequence/frame inventory without MANO weights or output files")
    parser.add_argument("--waypoint-slots", type=int, nargs="+", default=list(TAU_SLOTS))
    args = parser.parse_args()

    root = args.dexycb_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if tuple(args.waypoint_slots) != TAU_SLOTS:
        raise ValueError(f"Training config requires waypoint slots {list(TAU_SLOTS)} in this order")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not root.is_dir():
        raise FileNotFoundError(root)
    sequences = discover_sequences(root, args.view)
    groups, sampling = choose_nested_sets(sequences, args.seed)
    inventory = {
        "dexycb_root": str(root), "view": args.view,
        "sequence_count": len(sequences),
        "source_frame_count": sum(s["num_frames"] for s in sequences),
        "group_sequence_counts": {k: len(v) for k, v in groups.items()},
        "group_sampled_sequence_counts_by_subject": sampling,
        "sequence_count_by_subject": dict(Counter(s["subject_id"] for s in sequences)),
        "valid_current_frame_count": None,
        "source_files_copied": False,
    }
    if args.check_only:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return
    if args.mano_root is None:
        raise ValueError("--mano-root is required for calibrated geometry export")
    mano_root = args.mano_root.expanduser().resolve()
    panda_xml = args.panda_gripper_xml.expanduser().resolve()
    if not panda_xml.is_file():
        raise FileNotFoundError(f"Panda gripper XML not found: {panda_xml}")
    layers, mano_models = load_mano_layers(mano_root, [s["side"] for s in sequences])
    robot_basis = panda_tcp_basis(panda_xml)
    calibration = calibrate_hand_to_tcp(layers, robot_basis)
    # Avoid overwriting an existing derived release. This preserves reproducibility.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty derived-data directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    geometry = process_all_frames(sequences, calibration, args.workers)
    valid_count = sum(int(item["valid"].sum()) for item in geometry.values())
    inventory["valid_current_frame_count"] = valid_count
    inventory["source_timestamp_contract"] = "DexYCB provides synchronized frame indices, not per-frame timestamps in labels; nominal timestamp_seconds = frame_index / 30"
    inventory["waypoint_contract"] = {
        "model_waypoint_slots": list(TAU_SLOTS),
        "target_time_offsets_seconds": list(FUTURE_SECONDS),
        "nominal_source_fps": SOURCE_FPS,
        "fractional_frame_offsets": [round(t * SOURCE_FPS, 6) for t in FUTURE_SECONDS],
        "interpolation": "translation linear interpolation; rotation SLERP between adjacent valid source frames; exact-frame targets use the recorded pose",
        "interpolation_max_source_frame_gap": 1,
        "no_cross_sequence_or_invalid_frame_interpolation": True,
    }
    inventory["mano_model_files"] = {side: {"path": str(path), "sha256": sha256(path)}
                                     for side, path in mano_models.items()}
    inventory["axis_calibration"] = {
        side: {key: value.tolist() for key, value in values.items()}
        for side, values in calibration.items()
    }
    inventory["robot_tcp_frame"] = {
        "robot": "robosuite Panda",
        "pose_source": "LIBERO observation robot0_eef pose at the Panda gripper grip_site",
        "B_panda_tcp_columns": robot_basis.tolist(),
        "fixed_robot_C": np.eye(3).tolist(),
        "gripper_asset": {"path": str(panda_xml), "sha256": sha256(panda_xml)},
        "grip_site_frame_axes": "x=positive finger_joint1 slide; z=palm-to-fingertips; y=z cross x; derived from MJCF body rotations and joint axis",
    }
    inventory["rotation_convention"] = "column vectors; p_camera = R_camera_from_hand @ p_hand + t; pose_m root axis-angle and joint_3d are already in the camera label frame; visualization-only OpenGL flips are excluded"
    inventory["sampling_seed"] = args.seed
    inventory["selection"] = "H100 and H500 are nested subject-stratified sequence samples from the complete local DexYCB inventory; H1000 contains all 1000 sequences. No train/val/test split is created. Counts refer to capture sequences; each sequence expands to its valid annotated frames in the sample manifest."
    inventory["source_provenance"] = {
        "dataset": "DexYCB",
        "local_source_revision": "not pinned in the extracted dataset directory; no release/commit manifest found",
        "local_metadata_fingerprint_sha256": source_metadata_fingerprint(sequences, root, args.view),
        "sequence_order": "lexicographic capture-directory order per subject",
    }
    inventory["label_extraction"] = {
        "source_fields_used": ["pose_m[0, :3] (MANO root axis-angle)", "joint_3d[0, 0, :] (wrist position in meters)"],
        "frame_validity": "finite, nonzero pose_m and finite, non-sentinel wrist joint with positive camera z",
        "per_frame_mano_mesh_or_joint_decode": False,
        "one_time_mano_use": "neutral-pose MANO forward pass per side for fixed hand-axis calibration only",
        "parallel_sequence_readers": args.workers,
    }

    for name, group in groups.items():
        group_dir = output / name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_data = build_group_arrays(group, geometry, args.waypoint_slots, FUTURE_SECONDS)
        with (group_dir / "samples.jsonl").open("w") as stream:
            for record in group_data["records"]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (group_dir / "sample_metadata.jsonl").open("w") as stream:
            for metadata in group_data["sample_metadata"]:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for field in ("current_pose", "relative_waypoints", "position_valid", "rotation_valid", "effector_type"):
            np.save(group_dir / f"{field}.npy", group_data[field], allow_pickle=False)
        group_report = validate_group(group_dir, args.waypoint_slots)
        group_report["sequence_count"] = len(group)
        group_report["sequence_count_by_subject"] = dict(Counter(s["subject_id"] for s in group))
        inventory[name] = group_report
        print(f"{name}: {group_report['sequence_count']} sequences, {group_report['num_samples']} valid frames")

    (output / "metadata.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote README-compatible sidecars to {output}")


if __name__ == "__main__":
    main()
