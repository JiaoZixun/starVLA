import numpy as np
import pytest
import torch

from starVLA.dataloader.memory_v2 import (
    LiberoExecutionEventBuffer,
    QWEN_VISUAL_FEATURE_DIM,
    build_execution_event,
)
from starVLA.model.framework.VLM4A.execution_event_memory import ExecutionEventEncoder, MemoryConfig
from starVLA.model.framework.VLM4A.proprio_layer_bridge import (
    PROPRIO_DIM,
    TOKEN_PRIMARY,
    TOKEN_TEXT,
    TOKEN_WRIST,
    BridgeConfig,
    ProprioLayerBridge,
    bridge_token_types,
    proprio_features,
)

IMAGE = 99


def test_token_types_split_each_sample_into_primary_then_wrist():
    ids = torch.tensor(
        [
            [0, 1, IMAGE, IMAGE, IMAGE, IMAGE, 5, 6],
            [0, 0, 0, IMAGE, IMAGE, IMAGE, IMAGE, 6],  # left-padded sample
        ]
    )
    types = bridge_token_types(ids, IMAGE)
    assert types[0].tolist() == [TOKEN_TEXT] * 2 + [TOKEN_PRIMARY] * 2 + [TOKEN_WRIST] * 2 + [TOKEN_TEXT] * 2
    assert types[1].tolist() == [TOKEN_TEXT] * 3 + [TOKEN_PRIMARY] * 2 + [TOKEN_WRIST] * 2 + [TOKEN_TEXT]
    with pytest.raises(ValueError):
        bridge_token_types(torch.tensor([[IMAGE, IMAGE, IMAGE, 1]]), IMAGE)


def test_proprio_features_use_pose_rotation_and_gripper_qpos():
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, :3, 3] = torch.tensor([0.1, -0.2, 0.9])
    state = torch.zeros(2, 8)
    state[:, 6:] = torch.tensor([0.03, -0.03])
    features = proprio_features(pose, state)
    assert features.shape == (2, PROPRIO_DIM)
    assert torch.allclose(features[0, :3], torch.tensor([0.1, -0.2, 0.9]))
    assert torch.allclose(features[0, 3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(features[0, 9:], torch.tensor([0.03, -0.03]))


def _bridge_inputs(batch=2, tokens=12, width=32, layers=2):
    torch.manual_seed(0)
    ids = torch.zeros(batch, tokens, dtype=torch.long)
    ids[:, 2:10] = IMAGE
    mask = torch.ones(batch, tokens, dtype=torch.long)
    mask[1, :2] = 0
    return (
        torch.randn(batch, 8, width),
        torch.randn(batch, layers, tokens, width),
        bridge_token_types(ids, IMAGE),
        mask,
        torch.randn(batch, PROPRIO_DIM),
    )


def test_bridge_is_identity_at_init_and_learns_through_proprio():
    config = BridgeConfig.from_dict({"hidden_layers": [1, 2], "width": 16, "heads": 2, "ffn_width": 32, "layers": 1})
    bridge = ProprioLayerBridge(32, config)
    queries, layers, types, mask, proprio = _bridge_inputs()
    conditioned, residual = bridge(queries, layers, types, mask, proprio)
    assert torch.equal(conditioned, queries)
    assert torch.count_nonzero(residual) == 0

    torch.nn.init.normal_(bridge.output_projection.weight, std=0.1)
    conditioned, _ = bridge(queries, layers, types, mask, proprio)
    changed, _ = bridge(queries, layers, types, mask, proprio + 1.0)
    assert not torch.allclose(conditioned, changed)
    conditioned.sum().backward()
    assert bridge.proprio_projection[0].weight.grad.abs().sum() > 0


def test_bridge_ignores_padded_positions():
    config = BridgeConfig.from_dict({"hidden_layers": [1, 2], "width": 16, "heads": 2, "ffn_width": 32, "layers": 1})
    bridge = ProprioLayerBridge(32, config).eval()
    torch.nn.init.normal_(bridge.output_projection.weight, std=0.1)
    queries, layers, types, mask, proprio = _bridge_inputs()
    before, _ = bridge(queries, layers, types, mask, proprio)
    perturbed = layers.clone()
    perturbed[1, :, :2] += 100.0  # padded tokens of sample 1
    after, _ = bridge(queries, perturbed, types, mask, proprio)
    assert torch.allclose(before, after, atol=1e-5)


def test_bridge_config_rejects_bad_proprio_statistics():
    with pytest.raises(ValueError):
        BridgeConfig.from_dict({"proprio_std": [1.0] * (PROPRIO_DIM - 1)})
    with pytest.raises(ValueError):
        BridgeConfig.from_dict({"proprio_std": [0.0] * PROPRIO_DIM})


def _event_values():
    return {
        "state_start": np.zeros(8, dtype=np.float32),
        "state_end": np.ones(8, dtype=np.float32),
        "executed_actions": np.ones((8, 7), dtype=np.float32),
        "pose_start_in_episode": np.eye(4, dtype=np.float32),
        "pose_end_in_episode": np.eye(4, dtype=np.float32),
    }


def test_execution_event_without_visuals_for_qwen_memory():
    event = build_execution_event(start_step=0, end_step=8, visual_start=None, visual_end=None, **_event_values())
    assert event["visual_start"] is None and event["visual_end"] is None
    with pytest.raises(ValueError):
        build_execution_event(
            start_step=0, end_step=8, visual_start=np.zeros((2, 16, 3)), visual_end=None, **_event_values()
        )


def test_eval_buffer_can_omit_rgb_summaries():
    buffer = LiberoExecutionEventBuffer(capacity=4, include_visual=False)
    observation = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8)] * 2,
        "memory_state": np.zeros(8, dtype=np.float32),
        "geometry_current": {"eef_pose": np.eye(4, dtype=np.float32)},
    }
    assert buffer.begin_query(observation, 0) == []
    for step in range(8):
        buffer.record_executed_action(np.zeros(7, dtype=np.float32), step)
    events = buffer.begin_query(observation, 8)
    assert len(events) == 1 and events[0]["visual_start"] is None and events[0]["end_step"] == 8


def test_event_encoder_normalizes_qwen_visual_features():
    config = MemoryConfig(width=32, heads=2, ffn_width=64)
    encoder = ExecutionEventEncoder(QWEN_VISUAL_FEATURE_DIM, config, visual_input_norm=True)
    assert encoder.visual_input_norm is not None
    batch = 3
    visual = torch.randn(batch, 2, 16, QWEN_VISUAL_FEATURE_DIM)
    args = (
        torch.zeros(batch, 8),
        torch.zeros(batch, 8),
        torch.zeros(batch, 8, 7),
        torch.eye(4).repeat(batch, 1, 1),
        torch.eye(4).repeat(batch, 1, 1),
        torch.full((batch,), 8.0),
    )
    tokens = encoder(visual, visual * 50.0 + 3.0, *args)
    # Per-frame normalization makes the encoding invariant to feature scale/offset.
    same = encoder(visual, visual, *args)
    assert torch.allclose(tokens, same, atol=1e-4)
    assert tokens.shape == (batch, config.tokens_per_event, 32)
    # RGB checkpoints keep the original state-dict layout.
    assert ExecutionEventEncoder(3, config).visual_input_norm is None
