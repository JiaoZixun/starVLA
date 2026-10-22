#!/usr/bin/env python3
"""GPU checks for Spatial-Mem v3 (bridge + Qwen visual memory) before a long run.

1. Step-0 equivalence: with the reused Spatial 50k weights and zero-initialized
   bridge/memory outputs, v3 predicts the same actions as Spatial OFT.
2. Train/eval consistency: with those outputs randomized so every new path is
   active, the training sample path (dataset events + fp16 sidecar visuals) and
   the eval path (LiberoExecutionEventBuffer + the model's online visuals)
   produce matching events and actions over one episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.dataloader.memory_v2 import LiberoExecutionEventBuffer, MEMORY_STRIDE  # noqa: E402
from starVLA.model.framework.base_framework import baseframework, build_framework  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils  # noqa: E402

SPATIAL = "playground/Checkpoints/qwen35_0p8b_spatial_oft_libero_50k/checkpoints/steps_50000_pytorch_model.pt"


def load_v3(config_path: Path, local_model_path: str, fingerprint: str):
    cfg = OmegaConf.load(config_path)
    cfg.framework.qwenvl.local_model_path = local_model_path
    cfg.framework.memory.visual_fingerprint = fingerprint
    model = build_framework(cfg)
    model = TrainerUtils.load_pretrained_backbones(
        model, cfg.trainer.pretrained_checkpoint, reload_modules=cfg.trainer.reload_modules, strict_reload=True
    )
    model = TrainerUtils.freeze_backbones(model, freeze_modules=cfg.trainer.freeze_modules)
    return cfg, model.cuda().eval()


def sample_at(dataset, trajectory_id: int, step: int) -> dict:
    raw = dataset.get_step_data(trajectory_id, step)
    return dataset._pack_sample(dataset.transforms(raw))


def eval_example(sample: dict, events: list, episode_id: str, step: int) -> dict:
    return {
        "image": sample["image"],
        "lang": sample["lang"],
        "memory_state": sample["memory_state"],
        "geometry_current": {"eef_pose": sample["geometry_current"]["eef_pose"]},
        "memory_events": events,
        "memory_episode_id": episode_id,
        "env_step": step,
        "memory_contract": {
            "history_is_strictly_past": True,
            "decision_stride_env_steps": MEMORY_STRIDE,
            "capacity_events": 128,
            "source_frame_index": step,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/simBenchmarks/LIBERO/train_files/qwen35_0p8b_spatial_mem_oft_v3.yaml"))
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--dataset", default="libero_spatial_no_noops_1.0.0_lerobot")
    parser.add_argument("--trajectory-index", type=int, default=5)
    parser.add_argument("--steps", type=int, default=12, help="Number of stride-8 queries in the episode")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg0 = OmegaConf.load(args.config)
    fingerprint_file = Path(cfg0.datasets.vla_data.qwen_visual_sidecar_root) / args.dataset / "manifest.json"
    fingerprint = json.loads(fingerprint_file.read_text())["visual_fingerprint"]
    cfg, model = load_v3(args.config, args.local_model_path, fingerprint)
    dataset = make_LeRobotSingleDataset(
        Path(cfg.datasets.vla_data.data_root_dir), args.dataset, "libero_spatial_mem_oft_franka",
        data_cfg=cfg.datasets.vla_data,
    )
    trajectory_id = int(dataset.trajectory_ids[args.trajectory_index])
    length = int(dataset.trajectory_lengths[args.trajectory_index])
    steps = [k for k in range(0, length - MEMORY_STRIDE, MEMORY_STRIDE)][: args.steps]
    samples = [sample_at(dataset, trajectory_id, k) for k in steps]
    report: dict = {"trajectory_id": trajectory_id, "steps": steps, "fingerprint": fingerprint}

    # 1) step-0 equivalence against Spatial OFT
    spatial = baseframework.from_pretrained(
        SPATIAL,
        config_overrides=[
            "framework.spatial.adapter_version=v1",
            f"framework.qwenvl.local_model_path={args.local_model_path}",
        ],
    ).cuda().eval()
    spatial_inputs = [{k: v for k, v in s.items() if k in ("image", "lang", "geometry_current")} for s in samples]
    a_spatial = spatial.predict_action(examples=spatial_inputs)["normalized_actions"]
    a_v3 = model.predict_action(examples=samples)["normalized_actions"]
    report["step0_max_abs_diff_vs_spatial"] = float(np.abs(a_spatial - a_v3).max())
    report["action_abs_mean"] = float(np.abs(a_spatial).mean())
    del spatial
    torch.cuda.empty_cache()

    # 2) activate every new path, then compare train-sample and eval-client paths
    torch.manual_seed(0)
    for module in (model.proprio_layer_bridge.output_projection, model.action_memory_output, model.waypoint_twist_head):
        torch.nn.init.normal_(module.weight, std=0.02)
    buffer = LiberoExecutionEventBuffer(capacity=128, include_visual=False)
    max_event_diff = 0.0
    diffs, effect = [], []
    for k, sample in zip(steps, samples):
        if k:
            last = sample["memory_events"][-1]
            for i, action in enumerate(last["executed_actions"]):
                buffer.record_executed_action(action, k - MEMORY_STRIDE + i)
        events = buffer.begin_query(eval_example(sample, [], "check_episode", k), k)
        if len(events) != len(sample["memory_events"]):
            raise AssertionError(f"step {k}: eval has {len(events)} events, train has {len(sample['memory_events'])}")
        for ev, tr in zip(events, sample["memory_events"]):
            for key in ("start_step", "end_step"):
                if ev[key] != tr[key]:
                    raise AssertionError(f"step {k}: event {key} {ev[key]} != {tr[key]}")
            for key in ("state_start", "state_end", "executed_actions", "pose_start_in_episode", "pose_end_in_episode"):
                max_event_diff = max(max_event_diff, float(np.abs(ev[key] - tr[key]).max()))
        a_train = model.predict_action(examples=[sample])["normalized_actions"]
        a_eval = model.predict_action(examples=[eval_example(sample, events, "check_episode", k)])["normalized_actions"]
        a_nohist = model.predict_action(examples=[{**sample, "memory_events": []}])["normalized_actions"]
        diffs.append(float(np.abs(a_train - a_eval).max()))
        effect.append(float(np.abs(a_train - a_nohist).max()))
    report["event_field_max_abs_diff"] = max_event_diff
    report["train_vs_eval_action_max_abs_diff"] = max(diffs)
    report["history_effect_max_abs"] = max(effect)
    report["per_step_train_vs_eval"] = diffs
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
