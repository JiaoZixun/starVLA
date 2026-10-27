#!/usr/bin/env python3
"""Quick DexYCB preprocessing - skip validation."""

import json
import os
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def main():
    data_root = Path("/data1/jiaozx/data/DexYCB_data")
    output_dir = Path("data/preprocessed_dexycb")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all sequences
    all_sequences = []
    for subject_dir in sorted(data_root.glob("202*-subject-*")):
        for seq_dir in sorted(subject_dir.iterdir()):
            if seq_dir.is_dir() and (seq_dir / "meta.yml").exists():
                all_sequences.append(seq_dir)
    
    print(f"Found {len(all_sequences)} sequences")
    
    # Split train/val (90/10)
    np.random.seed(42)
    np.random.shuffle(all_sequences)
    split_idx = int(0.9 * len(all_sequences))
    train_seqs = all_sequences[:split_idx]
    val_seqs = all_sequences[split_idx:]
    
    for split_name, sequences in [("train", train_seqs), ("val", val_seqs)]:
        process_split(sequences, output_dir, split_name)
    
    print("Done!")

def process_split(sequences, output_dir, split_name):
    tau_frames = [1, 2, 4, 8]
    
    all_current_poses = []
    all_relative_waypoints = []
    all_future_valid = []
    all_samples = []
    
    for seq_path in tqdm(sequences, desc=f"Processing {split_name}"):
        try:
            # Load poses
            pose_data = np.load(seq_path / "pose.npz", allow_pickle=True)
            hand_poses = pose_data["pose_m"]  # [T, 4, 4] wrist poses
            
            if len(hand_poses) < 20:
                continue
            
            # Find RGB camera (use first available)
            cam_dirs = [d for d in seq_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if not cam_dirs:
                continue
            cam_id = cam_dirs[0].name
            
            # Process each frame as current
            for t in range(len(hand_poses) - 8):
                current_pose = hand_poses[t]
                
                # Compute relative waypoints
                waypoints = []
                valid = []
                for tau in tau_frames:
                    future_t = t + tau
                    if future_t < len(hand_poses):
                        future_pose = hand_poses[future_t]
                        # Relative transform: T_rel = T_current^-1 @ T_future
                        rel_pose = np.linalg.inv(current_pose) @ future_pose
                        waypoints.append(rel_pose)
                        valid.append(True)
                    else:
                        waypoints.append(np.eye(4))
                        valid.append(False)
                
                waypoints = np.stack(waypoints)  # [4, 4, 4]
                valid = np.array(valid)
                
                # Find RGB file
                rgb_path = seq_path / cam_id / "color" / f"{t:06d}.jpg"
                if not rgb_path.exists():
                    continue
                
                # Record
                all_current_poses.append(current_pose)
                all_relative_waypoints.append(waypoints)
                all_future_valid.append(valid)
                all_samples.append({
                    "rgb_path": str(rgb_path.relative_to("/data1/jiaozx/starVLA")),
                    "seq": seq_path.name,
                    "frame": t,
                })
        
        except Exception as e:
            print(f"Error in {seq_path.name}: {e}")
            continue
    
    # Save arrays
    np.save(output_dir / f"{split_name}_current_pose.npy", np.stack(all_current_poses))
    np.save(output_dir / f"{split_name}_relative_waypoints.npy", np.stack(all_relative_waypoints))
    np.save(output_dir / f"{split_name}_future_valid.npy", np.stack(all_future_valid))
    
    # Save metadata
    with open(output_dir / f"{split_name}_samples.jsonl", "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"{split_name}: {len(all_samples)} samples saved")

if __name__ == "__main__":
    main()
