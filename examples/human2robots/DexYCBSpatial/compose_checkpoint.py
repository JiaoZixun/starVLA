"""Compose a robot-ready checkpoint from human geometry and LIBERO baseline weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError(f"Expected a flat state_dict in {path}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-checkpoint", type=Path, required=True)
    parser.add_argument("--robot-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    human = load_checkpoint(args.human_checkpoint)
    robot = load_checkpoint(args.robot_baseline)
    robot_action = {key: value for key, value in robot.items() if key.startswith("action_model.")}
    human_action = {key for key in human if key.startswith("action_model.")}
    if not robot_action or not human_action:
        raise ValueError("Both checkpoints must contain action_model parameters")
    if set(robot_action) != human_action:
        missing = sorted(human_action - set(robot_action))
        extra = sorted(set(robot_action) - human_action)
        raise ValueError(f"Action-head key mismatch; missing={missing[:8]}, extra={extra[:8]}")
    for key, value in robot_action.items():
        if human[key].shape != value.shape:
            raise ValueError(f"Action-head shape mismatch for {key}: {tuple(human[key].shape)} != {tuple(value.shape)}")

    composed = dict(human)
    composed.update(robot_action)
    required_prefixes = ("qwen_vl_interface.", "waypoint_model.", "spatial_adapter.", "action_model.")
    missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in composed)]
    if missing:
        raise ValueError(f"Composed checkpoint is incomplete: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(composed, args.output)
    print(f"Saved {len(composed)} tensors to {args.output}")


if __name__ == "__main__":
    main()
