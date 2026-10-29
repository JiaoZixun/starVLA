"""Object-grounding tokens, grounding losses, label-consistent augmentation, DexYCB v3 pairs, LIBERO hold points."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.grounding_augment import GroundingAugment
from starVLA.dataloader.human_wrist_dataset import HumanGroundingDataset, collate_human_grounding_batch
from starVLA.model.framework.VLM4A.QwenSpatialOFT import QwenSpatialOFT, SpatialAdapter, WaypointModel

_spec = importlib.util.spec_from_file_location(
    "prep_grounding", Path(__file__).resolve().parents[1] / "examples/simBenchmarks/LIBERO/train_files/prepare_grounding_sidecars.py"
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _bridge_inputs(batch=2, layers=2, vlm=16, chunk=3, length=7):
    aq = [torch.randn(batch, chunk, vlm) for _ in range(layers)]
    raw = [torch.randn(batch, length, vlm) for _ in range(layers)]
    return aq, raw


def test_object_slots_predict_and_condition_the_adapter():
    torch.manual_seed(0)
    model = WaypointModel(16, width=8, heads=2, ffn_width=16, num_waypoints=4, bridge_layers=2, anchor_mode="gravity", num_object_slots=2)
    adapter = SpatialAdapter(16, width=8, heads=2, num_waypoints=4, gated=True, bridge_layers=2, ffn_width=16, num_object_slots=2)
    aq, raw = _bridge_inputs()
    tau = torch.tensor([1, 2, 4, 8.0]) / 8
    translation, rotation, gripper = model(None, torch.eye(4).repeat(2, 1, 1), tau, None, bridge_aq=aq, bridge_raw=raw)
    assert translation.shape == (2, 4, 3) and gripper.shape == (2, 4)
    grounding = model.last_grounding
    assert grounding["xyz"].shape == (2, 2, 3) and grounding["uv"].shape == (2, 2, 2)
    # Zero-initialised heads: xyz = 0, uv = image centre.
    assert torch.all(grounding["xyz"] == 0) and torch.allclose(grounding["uv"], torch.full((2, 2, 2), 0.5))
    out, residual, _ = adapter(torch.randn(2, 3, 16), translation, rotation, gripper, tau, bridge_aq=aq, bridge_raw=raw, grounding=grounding)
    assert out.shape == (2, 3, 16)
    # Gradients from the action path reach the object tokens' projection.
    adapter.output_projection.weight.data.normal_()
    out, _, _ = adapter(torch.randn(2, 3, 16), translation, rotation, gripper, tau, bridge_aq=aq, bridge_raw=raw, grounding=grounding)
    out.sum().backward()
    assert adapter.object_projection[0].weight.grad.abs().sum() > 0


def test_old_configs_build_no_object_slots():
    model = WaypointModel(16, width=8, heads=2, ffn_width=16, num_waypoints=4, bridge_layers=2)
    assert not hasattr(model, "grounding_xyz_head") and model.num_object_slots == 0
    adapter = SpatialAdapter(16, width=8, heads=2, num_waypoints=4, bridge_layers=2, ffn_width=16)
    assert not hasattr(adapter, "object_projection")


def _labels(valid_xyz=(True, False), valid_uv=(True, True)):
    return [{
        "object_xyz": np.asarray([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]], np.float32), "object_xyz_valid": np.asarray(valid_xyz),
        "object_uv": np.asarray([[0.5, 0.5], [0.5 + 1 / 64, 0.5]], np.float32), "object_uv_valid": np.asarray(valid_uv),
    }]


def test_grounding_losses_masked_and_weighted():
    prediction = {"xyz": torch.zeros(1, 2, 3, requires_grad=True), "uv": torch.full((1, 2, 2), 0.5, requires_grad=True)}
    owner = SimpleNamespace(
        num_object_slots=2, grounding_xyz_weight=0.5, grounding_uv_weight=2.0,
        waypoint_model=SimpleNamespace(last_grounding=prediction), _stack_slot_labels=QwenSpatialOFT._stack_slot_labels,
    )
    losses = QwenSpatialOFT._grounding_losses(owner, _labels(), torch.device("cpu"))
    # xyz: 0.05 m / 0.1 -> 0.5 on one axis of the single valid slot -> mean Huber 0.125 / 3.
    assert torch.isclose(losses["grounding_xyz_loss"], torch.tensor(0.125 / 3))
    # uv: slot 1 off by 1/64 = 0.5 units of 1/32 on one axis -> mean over 4 values = 0.125 / 4.
    assert torch.isclose(losses["grounding_uv_loss"], torch.tensor(0.125 / 4))
    assert torch.isclose(losses["grounding_loss"], 0.5 * losses["grounding_xyz_loss"] + 2.0 * losses["grounding_uv_loss"])
    losses["grounding_loss"].backward()
    assert prediction["xyz"].grad[0, 1].abs().sum() == 0  # invalid goal slot gets no gradient


def test_augment_moves_uv_with_the_image():
    torch.manual_seed(3)
    image = np.zeros((64, 64, 3), np.uint8)
    image[40:44, 18:22] = 255  # bright dot centred at col 20, row 42 (pixel centres 18..21 / 40..43)
    uv = np.asarray([[(19.5 + 0.5) / 64, (41.5 + 0.5) / 64]], np.float32)
    augment = GroundingAugment({"enabled": True, "brightness": 0, "contrast": 0, "saturation": 0, "blur_p": 0, "noise_std": 0,
                                "secondary_dropout": 1.0, "scale": [0.9, 1.1], "rotate_deg": 10, "shift": 0.1})
    for _ in range(5):
        (primary, secondary), moved, valid = augment([Image.fromarray(image), Image.fromarray(image)], uv, np.asarray([True]))
        out = np.asarray(primary)[..., 0].astype(float)
        rows, cols = np.nonzero(out > 60)
        centre = np.array([(cols * out[rows, cols]).sum(), (rows * out[rows, cols]).sum()]) / out[rows, cols].sum()
        assert valid[0] and np.allclose(moved[0] * 64 - 0.5, centre, atol=0.6)
        assert np.asarray(secondary).max() == 0  # second view dropped


def test_augment_disabled_is_identity():
    images = [Image.new("RGB", (8, 8), (10, 20, 30))]
    out, uv, valid = GroundingAugment({"enabled": False})(images, np.zeros((1, 2)), np.ones(1, bool))
    assert out is images


def _write_v3(root, count=2, k=4):
    split = root / "train"
    split.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"format": "starvla_dexycb_grounding_sidecar_v3", "tau_slots": [1, 2, 4, 8]}))
    Image.new("RGB", (8, 8)).save(root / "a.jpg")
    rows = [{"image": str(root / "a.jpg"), "image2": str(root / "a.jpg"), "sequence_id": "s", "frame_index": i,
             "objects": ["banana", "mug", "bowl"], "grasp_index": 1, "lang": "pick up the mug"} for i in range(count)]
    (split / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    objects = np.zeros((count, 5, 3), np.float32)
    objects[:, :3, 0] = [0.1, 0.2, 0.3]
    arrays = {
        "current_pose": np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
        "relative_waypoints": np.tile(np.eye(4, dtype=np.float32), (count, k, 1, 1)),
        "future_valid": np.ones((count, k), bool), "future_closure": np.ones((count, k), np.float32),
        "future_closure_valid": np.ones((count, k), bool), "object_wrist": objects,
        "object_valid": np.asarray([[True, True, True, False, False]] * count),
        "object_uv": np.full((count, 5, 2), 0.5, np.float32), "object_uv_valid": np.asarray([[True, True, True, False, False]] * count),
    }
    for key, value in arrays.items():
        np.save(split / f"{key}.npy", value)


def test_human_grounding_pairs_real_and_counterfactual(tmp_path):
    _write_v3(tmp_path)
    torch.manual_seed(0)
    batch = collate_human_grounding_batch([HumanGroundingDataset(tmp_path, image_size=8)[i] for i in range(2)])
    assert len(batch) == 4
    real, counter = batch[0], batch[1]
    assert real["lang"] == "pick up the mug" and not real["counterfactual"]
    assert np.isclose(real["geometry_labels"]["object_xyz"][0, 0], 0.2) and real["geometry_labels"]["future_valid"].all()
    assert counter["counterfactual"] and counter["lang"] in ("pick up the banana", "pick up the bowl")
    other = {"pick up the banana": 0.1, "pick up the bowl": 0.3}[counter["lang"]]
    assert np.isclose(counter["geometry_labels"]["object_xyz"][0, 0], other)
    # Counterfactual: grounding only; no waypoint or closure supervision; no goal slot.
    assert not counter["geometry_labels"]["future_valid"].any() and not counter["geometry_labels"]["future_gripper_valid"].any()
    assert counter["geometry_labels"]["object_xyz_valid"].tolist() == [True, False]
    assert counter["image"] is real["image"]


def test_libero_hold_points():
    gripper = np.asarray([1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0], np.float64)  # one-frame closure at 7 is noise
    segments = prep.hold_segments(gripper, min_hold=3, min_gap=3)
    assert segments == [(2, 5), (9, 12)]
    eef = np.arange(12, dtype=np.float64)[:, None].repeat(3, 1)
    target, goal, valid, holding = prep.episode_points(eef, segments)
    assert target[0, 0] == 2 and goal[0, 0] == 5  # before the first hold: its grasp and release points
    assert target[3, 0] == 3 and goal[3, 0] == 5  # while holding: the TCP itself, goal = release
    assert target[6, 0] == 9 and goal[6, 0] == 11  # next hold; demo ends holding -> last frame is the goal
    assert valid.all() and holding.tolist() == [False, False, True, True, True, False, False, False, False, True, True, True]


def test_training_uv_matches_rot180_camera_pixels():
    matrix = np.eye(4)
    uv, inside = prep.to_training_uv(matrix, np.asarray([[10.0, 20.0, 1.0]]), size=256)
    # Camera pixel (u=10, v=20) -> training image col 245, row 20.
    assert np.allclose(uv[0] * 256 - 0.5, [245, 20]) and inside[0]
