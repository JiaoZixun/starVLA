import numpy as np
import torch

from starVLA.dataloader.memory_v2 import (
    EpisodeEventHistory,
    LiberoExecutionEventBuffer,
    build_execution_event,
)
from starVLA.model.framework.VLM4A.execution_event_memory import (
    CausalMemoryReader,
    ExecutionEventEncoder,
    MemoryConfig,
    se3_exp,
)


def _event(start: int, end: int) -> dict:
    return {
        "start_step": start,
        "end_step": end,
        "event_tokens": np.zeros((9, 16), dtype=np.float32),
    }


def test_event_history_enforces_order_capacity_and_causal_visibility():
    prefix = EpisodeEventHistory(capacity=2)
    prefix.append(_event(0, 8))
    prefix.append(_event(8, 16))
    assert [event["start_step"] for event in prefix.available(16)] == [0, 8]

    history = EpisodeEventHistory(capacity=2)
    history.append(_event(0, 8))
    history.append(_event(8, 16))
    history.append(_event(16, 24))

    assert [event["start_step"] for event in history.available(24)] == [8, 16]
    assert [event["start_step"] for event in history.available(23)] == [8]

    try:
        history.append(_event(32, 40))
    except ValueError as exc:
        assert "without gaps" in str(exc)
    else:
        raise AssertionError("event gaps must be rejected")


def test_execution_event_rejects_wrong_interval_and_packs_real_actions():
    values = {
        "visual_start": np.zeros((2, 16, 3), dtype=np.float32),
        "visual_end": np.ones((2, 16, 3), dtype=np.float32),
        "state_start": np.zeros(8, dtype=np.float32),
        "state_end": np.ones(8, dtype=np.float32),
        "executed_actions": np.ones((8, 7), dtype=np.float32),
        "pose_start_in_episode": np.eye(4, dtype=np.float32),
        "pose_end_in_episode": np.eye(4, dtype=np.float32),
    }
    event = build_execution_event(start_step=8, end_step=16, **values)
    assert event["elapsed_steps"] == 8
    assert event["executed_actions"].shape == (8, 7)

    try:
        build_execution_event(start_step=8, end_step=15, **values)
    except ValueError as exc:
        assert "exactly 8 steps" in str(exc)
    else:
        raise AssertionError("partial events must not be committed")


def test_eval_buffer_emits_training_format_only_after_eight_accepted_actions():
    buffer = LiberoExecutionEventBuffer(capacity=2)
    observation = {
        "image": [np.zeros((224, 224, 3), dtype=np.uint8)] * 2,
        "state": np.zeros((1, 8), dtype=np.float32),
        "geometry_current": {"eef_pose": np.eye(4, dtype=np.float32)},
    }
    assert buffer.begin_query(observation, 0) == []
    for step in range(8):
        action = np.asarray([step, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        buffer.record_executed_action(action, step)

    later = {**observation, "state": np.ones((1, 8), dtype=np.float32)}
    events = buffer.begin_query(later, 8)
    assert len(events) == 1
    assert events[0]["start_step"] == 0 and events[0]["end_step"] == 8
    assert events[0]["visual_start"].shape == (2, 16, 3)
    assert events[0]["executed_actions"].shape == (8, 7)
    assert np.array_equal(events[0]["executed_actions"][:, 0], np.arange(8, dtype=np.float32))


def test_event_encoder_batches_events_and_backpropagates():
    config = MemoryConfig(width=16, heads=4, ffn_width=32, event_tokens=2, summary_tokens=1)
    encoder = ExecutionEventEncoder(visual_dim=3, config=config)
    visual_start = torch.rand((3, 2, 16, 3))
    visual_end = torch.rand((3, 2, 16, 3))
    output = encoder(
        visual_start,
        visual_end,
        torch.zeros((3, 8)),
        torch.ones((3, 8)),
        torch.zeros((3, 8, 7)),
        torch.eye(4).repeat(3, 1, 1),
        torch.eye(4).repeat(3, 1, 1),
        torch.full((3,), 8.0),
    )
    assert output.shape == (3, config.tokens_per_event, config.width)
    output.square().mean().backward()
    assert encoder.visual_projection.weight.grad is not None


def test_causal_reader_masks_future_events_and_keeps_recent_capacity():
    config = MemoryConfig(
        width=16,
        heads=4,
        ffn_width=32,
        event_tokens=1,
        summary_tokens=1,
        capacity_events=2,
        reader_layers=1,
    )
    reader = CausalMemoryReader(config)
    queries = torch.zeros((1, 3, 16))
    events = torch.zeros((1, 3, config.tokens_per_event, 16))
    valid = torch.ones((1, 3), dtype=torch.bool)
    query_steps = torch.tensor([[8, 16, 24]])
    starts = torch.tensor([[0, 8, 16]])
    ends = torch.tensor([[8, 16, 24]])

    _, allowed = reader(queries, events, valid, query_steps, starts, ends)
    assert torch.equal(
        allowed,
        torch.tensor([[[True, False, False], [True, True, False], [False, True, True]]]),
    )


def test_se3_exponential_handles_identity_translation_and_rotation_with_gradients():
    zero = torch.zeros((1, 6), requires_grad=True)
    identity = se3_exp(zero)
    assert torch.allclose(identity, torch.eye(4).unsqueeze(0), atol=1e-7)

    translation = se3_exp(torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(translation[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))

    half_turn = se3_exp(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]]))
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(half_turn[0, :3, :3], expected, atol=1e-6)
    identity.sum().backward()
    assert zero.grad is not None and torch.isfinite(zero.grad).all()
