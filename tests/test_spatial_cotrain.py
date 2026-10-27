import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.dataloader.cotrain_dataloader import HumanRobotCotrainLoader
from starVLA.dataloader.human_wrist_dataset import HumanWristGeometryDataset
from starVLA.dataloader.phase_labels import (
    APPROACH,
    GRASP,
    RELEASE,
    TRANSPORT,
    phases_from_holding,
    robot_holding_from_gripper_action,
    slot_phases,
)
from starVLA.model.framework.VLM4A.QwenSpatialOFT import (
    ARM_ID,
    PHASES,
    QwenSpatialOFT,
    WaypointModel,
    anchor_features,
)


def _pose(yaw=0.0, tilt=0.0, position=(0.0, 0.0, 0.0)):
    """Top-down gripper (approach = world -z), rotated about world z by yaw then tilted about world x."""
    cy, sy, ct, st = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
    yaw_r = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    tilt_r = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    down = np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4)
    pose[:3, :3] = tilt_r @ yaw_r @ down
    pose[:3, 3] = position
    return torch.tensor(pose, dtype=torch.float32)


def test_gravity_anchor_ignores_position_and_heading():
    base = anchor_features(_pose()[None], "gravity")
    moved = anchor_features(_pose(yaw=1.3, position=(0.5, -0.4, 1.1))[None], "gravity")
    tilted = anchor_features(_pose(tilt=0.4)[None], "gravity")
    assert base.shape == (1, 9)
    assert torch.allclose(base, moved, atol=1e-6)
    # Top-down approach axis (+z local) points along gravity.
    assert torch.allclose(base[0, :3], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.all(base[0, 3:] == 0)
    assert not torch.allclose(base, tilted, atol=1e-3)


def test_absolute_anchor_is_the_previous_input():
    pose = _pose(yaw=0.3, position=(0.1, 0.2, 0.9))[None]
    features = anchor_features(pose, "absolute")
    assert torch.allclose(features[0, :3], torch.tensor([0.1, 0.2, 0.9]))
    with pytest.raises(ValueError):
        anchor_features(pose, "table")


def test_phases_from_holding_marks_grasp_transport_release():
    holding = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    phase = phases_from_holding(holding, window=2)
    assert phase.tolist() == [
        APPROACH, APPROACH, APPROACH, GRASP, GRASP, TRANSPORT, TRANSPORT, TRANSPORT, RELEASE, RELEASE, APPROACH, APPROACH,
    ]
    # A regrasp inside the release window is labelled grasp.
    regrasp = phases_from_holding(np.array([1, 1, 0, 0, 1, 1], dtype=bool), window=2)
    assert regrasp.tolist() == [TRANSPORT, TRANSPORT, GRASP, GRASP, TRANSPORT, TRANSPORT]
    assert phases_from_holding(np.zeros(4, dtype=bool), window=3).tolist() == [APPROACH] * 4


def test_robot_holding_uses_libero_gripper_convention():
    # LIBERO LeRobot actions: 1 = open, 0 = close.
    assert robot_holding_from_gripper_action(np.array([1.0, 0.0, 0.0, 1.0])).tolist() == [False, True, True, False]
    with pytest.raises(ValueError):
        robot_holding_from_gripper_action(np.array([-1.0, 1.0]))
    labels, valid = slot_phases(np.array([0, 1, 2, 3]), [1, 3, 5])
    assert labels.tolist() == [1, 3, 0] and valid.tolist() == [True, True, False]


def test_cotrain_loader_interleaves_and_restarts_each_source():
    robot = [["r0"], ["r1"], ["r2"]]
    human = [["h0"], ["h1"]]
    loader = HumanRobotCotrainLoader(robot, human, human_every=3)
    stream = iter(loader)
    order = [next(stream)[0] for _ in range(9)]
    assert order == ["r0", "r1", "h0", "r2", "r0", "h1", "r1", "r2", "h0"]
    assert len(loader) == 3
    with pytest.raises(ValueError):
        HumanRobotCotrainLoader(robot, human, human_every=1)


def _waypoint_model(**kwargs):
    torch.manual_seed(0)
    return WaypointModel(32, width=16, heads=2, ffn_width=32, layers=1, num_waypoints=4, bridge_layers=2, **kwargs)


def _bridge_inputs(batch=2, tokens=6, width=32, layers=2):
    aq = [torch.randn(batch, 3, width) for _ in range(layers)]
    raw = [torch.randn(batch, tokens, width) for _ in range(layers)]
    return aq, raw


def test_waypoint_model_arm_id_selects_domain_embedding():
    model = _waypoint_model()
    with torch.no_grad():
        model.translation_head.weight.normal_()
    aq, raw = _bridge_inputs()
    pose = torch.stack([_pose(), _pose(yaw=0.5)])
    tau = torch.tensor([0.125, 0.25, 0.5, 1.0])
    robot = model(None, pose, tau, bridge_aq=aq, bridge_raw=raw)
    default = model(None, pose, tau, bridge_aq=aq, bridge_raw=raw, arm_id=ARM_ID["robot"])
    human = model(None, pose, tau, bridge_aq=aq, bridge_raw=raw, arm_id=ARM_ID["human"])
    assert len(robot) == 3
    assert torch.equal(robot[0], default[0])
    assert not torch.allclose(robot[0], human[0])
    *_, features = model(None, pose, tau, bridge_aq=aq, bridge_raw=raw, return_features=True)
    assert features.shape == (2, 4, 16)


def test_auxiliary_heads_exist_only_when_enabled():
    plain = _waypoint_model()
    assert not hasattr(plain, "phase_head") and not hasattr(plain, "object_head")
    assert plain.auxiliary_outputs(torch.zeros(2, 4, 16)) == {}
    # Old checkpoints (no aux heads) still load strictly into a model without them.
    _waypoint_model().load_state_dict(plain.state_dict(), strict=True)
    full = _waypoint_model(num_phases=len(PHASES), predict_object=True, anchor_mode="gravity")
    outputs = full.auxiliary_outputs(torch.zeros(2, 4, 16))
    assert outputs["phase_logits"].shape == (2, 4, 4)
    assert outputs["object_position"].shape == (2, 4, 3)


def _slot_owner(phase_weight, object_weight):
    return SimpleNamespace(
        waypoint_tau=(1, 2, 4, 8), phase_weight=phase_weight, object_weight=object_weight,
        _stack_slot_labels=QwenSpatialOFT._stack_slot_labels,
    )


def _labels(with_phase=True, with_object=True):
    label = {}
    if with_phase:
        label.update(future_phase=np.array([0, 1, 2, 2]), future_phase_valid=np.array([True, True, True, False]))
    if with_object:
        label.update(
            future_object=np.full((4, 3), 0.05, dtype=np.float32), future_object_valid=np.array([True, True, False, False])
        )
    return [label, dict(label)]


def test_slot_losses_are_masked_and_weighted():
    owner = _slot_owner(phase_weight=0.1, object_weight=0.5)
    logits = torch.zeros(2, 4, 4, requires_grad=True)
    objects = torch.zeros(2, 4, 3, requires_grad=True)
    losses = QwenSpatialOFT._slot_supervision_losses(
        owner, _labels(), {"phase_logits": logits, "object_position": objects}, torch.device("cpu"), "human"
    )
    assert torch.isclose(losses["phase_loss"], torch.log(torch.tensor(4.0)))
    # 0.05 m error / 0.1 = 0.5 per axis -> Huber 0.125.
    assert torch.isclose(losses["object_loss"], torch.tensor(0.125))
    assert torch.isclose(losses["slot_loss"], 0.1 * losses["phase_loss"] + 0.5 * losses["object_loss"])
    losses["slot_loss"].backward()
    assert objects.grad[:, 2:].abs().sum() == 0  # invalid slots get no gradient


def test_robot_batch_without_object_labels_skips_object_loss():
    owner = _slot_owner(phase_weight=0.1, object_weight=0.5)
    outputs = {"phase_logits": torch.zeros(2, 4, 4), "object_position": torch.zeros(2, 4, 3)}
    losses = QwenSpatialOFT._slot_supervision_losses(owner, _labels(with_object=False), outputs, torch.device("cpu"), "robot")
    assert "object_loss" not in losses and "phase_loss" in losses
    with pytest.raises(ValueError, match="future_phase"):
        QwenSpatialOFT._slot_supervision_losses(owner, _labels(with_phase=False), outputs, torch.device("cpu"), "robot")
    mixed = _labels()
    del mixed[1]["future_phase"]
    with pytest.raises(ValueError, match="every example"):
        QwenSpatialOFT._slot_supervision_losses(owner, mixed, outputs, torch.device("cpu"), "human")


def test_slot_losses_off_by_default():
    losses = QwenSpatialOFT._slot_supervision_losses(_slot_owner(0.0, 0.0), _labels(), {}, torch.device("cpu"), "robot")
    assert set(losses) == {"slot_loss"} and losses["slot_loss"] == 0


def _write_sidecar(root, with_v2=True, count=3, k=4):
    split = root / "train"
    split.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"tau_slots": [1, 2, 4, 8]}))
    image = root / "frame.jpg"
    Image.new("RGB", (8, 8)).save(image)
    rows = [{"image": str(image), "sequence_id": "s", "frame_index": i, "lang": "pick up the banana"} for i in range(count)]
    (split / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    np.save(split / "current_pose.npy", np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)))
    np.save(split / "relative_waypoints.npy", np.tile(np.eye(4, dtype=np.float32), (count, k, 1, 1)))
    np.save(split / "future_valid.npy", np.ones((count, k), dtype=bool))
    if with_v2:
        np.save(split / "future_phase.npy", np.full((count, k), 2, dtype=np.int64))
        np.save(split / "future_phase_valid.npy", np.ones((count, k), dtype=bool))
        np.save(split / "future_object.npy", np.zeros((count, k, 3), dtype=np.float32))
        np.save(split / "future_object_valid.npy", np.ones((count, k), dtype=bool))


def test_trainer_logs_latest_metrics_of_both_domains(monkeypatch):
    monkeypatch.setenv("STARVLA_DISABLE_DEEPSPEED", "1")
    from starVLA.training.train_starvla import VLATrainer

    merge = VLATrainer._merge_domain_metrics.__get__(SimpleNamespace())
    assert merge({"_domain": "robot", "total_loss": 1.0, "timing/data": 0.1}) == {"total_loss": 1.0, "timing/data": 0.1}
    after_human = merge({"_domain": "human", "total_loss": 0.2, "timing/data": 0.2})
    assert after_human == {"total_loss": 1.0, "timing/data": 0.2, "human/total_loss": 0.2}
    after_robot = merge({"_domain": "robot", "total_loss": 0.9, "timing/data": 0.3})
    assert after_robot == {"total_loss": 0.9, "timing/data": 0.3, "human/total_loss": 0.2}


def test_human_dataset_reads_optional_v2_labels(tmp_path):
    _write_sidecar(tmp_path / "v2")
    sample = HumanWristGeometryDataset(tmp_path / "v2", image_size=8)[1]
    assert sample["domain"] == "human" and sample["lang"] == "pick up the banana"
    labels = sample["geometry_labels"]
    assert labels["future_phase"].tolist() == [2, 2, 2, 2] and labels["future_object"].shape == (4, 3)
    _write_sidecar(tmp_path / "v1", with_v2=False)
    v1_labels = HumanWristGeometryDataset(tmp_path / "v1", image_size=8)[0]["geometry_labels"]
    assert set(v1_labels) == {"relative_waypoints", "future_valid"}
