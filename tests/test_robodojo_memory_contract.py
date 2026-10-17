"""Causal parity, fused recurrence, and server feedback regression coverage."""
import numpy as np
import pandas as pd
import pytest
import torch

from deployment.model_server.robodojo_memory_adapter import RoboDojoMemorySessions
from starVLA.dataloader.robodojo_memory import RoboDojoMemorySingleDataset
from starVLA.model.framework.VLM4A.QwenSpatialMemOFT import CausalExecutionMemory, QwenSpatialMemOFT
from starVLA.model.framework.VLM4A.QwenSpatialOFT import SpatialAdapter, WaypointDecoder
from starVLA.utils.robodojo_memory import RoboDojoMemoryBuffer, RoboDojoMemorySpec


def episode_data(length=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"frame_index": np.arange(length),
                         "observation.state": list(rng.normal(size=(length, 14)).astype(np.float32)),
                         "action": list(rng.normal(size=(length, 14)).astype(np.float32))})


@pytest.mark.parametrize("use_pose", [False, True])
def test_offline_online_match_every_frame_through_padding_rollover_reset(use_pose):
    spec = RoboDojoMemorySpec(3, 4, use_pose)
    data = episode_data()
    dataset = object.__new__(RoboDojoMemorySingleDataset)
    dataset.memory_spec = spec
    dataset.curr_traj_data = data
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (len(data), 2, 4, 4)).copy()
    dataset._spatial_episode_cache = {5: {"pose": poses, "source_frame_index": np.arange(len(data))}}
    buffer = RoboDojoMemoryBuffer(3, 4, use_pose)
    for step in range(len(data)):
        offline = dataset._memory_records(5, step)
        online = buffer.build(step)
        for key in offline:
            np.testing.assert_array_equal(offline[key], online[key])
        spec.validate(online, spec.contract(step))
        buffer.append(data.iloc[step]["observation.state"], data.iloc[step]["action"],
                      source_step=step, pose=poses[step] if use_pose else None)
    buffer.reset()
    assert not buffer.build(0)["valid_mask"].any()


def test_future_rows_never_enter_dataset_history():
    dataset = object.__new__(RoboDojoMemorySingleDataset)
    dataset.memory_spec = RoboDojoMemorySpec(3, 4)
    dataset.curr_traj_data = episode_data()
    before = dataset._memory_records(0, 20)
    for index in range(20, len(dataset.curr_traj_data)):
        for key in ("observation.state", "action"):
            dataset.curr_traj_data.at[index, key] = np.full(14, 999, np.float32)
    after = dataset._memory_records(0, 20)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def tiny_model(spec):
    model = QwenSpatialMemOFT.__new__(QwenSpatialMemOFT)
    torch.nn.Module.__init__(model)
    model.memory_required = True
    model.memory_spec = spec
    return model


@pytest.mark.parametrize("tamper", ["current", "future", "elapsed", "mask", "nonfinite", "burnin", "grid", "pose"])
def test_reject_invalid_or_leaking_records(tamper):
    spec = RoboDojoMemorySpec(3, 4)
    buffer = RoboDojoMemoryBuffer(3, 4)
    for step in range(12):
        buffer.append(np.zeros(14), np.ones(14), source_step=step)
    record = buffer.build(12)
    example = {"memory_records": record, "memory_contract": spec.contract(12)}
    if tamper in {"current", "future"}:
        record["source_frame_index"][-1] = 12 if tamper == "current" else 13
    elif tamper == "elapsed":
        record["elapsed_env_steps"][-1] = 0
    elif tamper == "mask":
        record["valid_mask"][-1] = False
    elif tamper == "nonfinite":
        record["state"][-1, 0] = np.nan
    elif tamper == "burnin":
        example["memory_burnin_records"] = {"state": np.ones((1, 14))}
    elif tamper == "grid":
        record["source_frame_index"][-1] = 9
    else:
        record["eef_pose"] = np.zeros((3, 2, 4, 4))
    with pytest.raises(ValueError):
        tiny_model(spec)._batch_memory_records([example])


def test_fused_gru_matches_masked_cell_outputs_and_gradients():
    torch.manual_seed(4)
    memory = CausalExecutionMemory(32, width=16, heads=2, layers=1)
    cell = torch.nn.GRUCell(16, 16)
    cell.load_state_dict({k.removesuffix("_l0"): v for k, v in memory.slot_update.state_dict().items()})
    valid = torch.tensor([[False]*5, [False, False, True, True, True], [True]*5])
    inputs = torch.randn(3, 5, 16, requires_grad=True)
    initial = memory.slot_seed[None].expand(3, -1, -1)
    fused = memory._update_slots(initial, inputs, valid)
    expected = initial
    for step in range(5):
        updated = cell(inputs[:, step, None].expand(-1, 8, -1).reshape(-1, 16), expected.reshape(-1, 16)).view(3, 8, 16)
        expected = torch.where(valid[:, step, None, None], updated, expected)
    torch.testing.assert_close(fused, expected, atol=2e-6, rtol=2e-5)
    fused_grad = torch.autograd.grad(fused.square().sum(), inputs, retain_graph=True)[0]
    cell_grad = torch.autograd.grad(expected.square().sum(), inputs)[0]
    torch.testing.assert_close(fused_grad, cell_grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("use_pose", [False, True])
def test_dual_arm_memory_zero_init_and_backward(use_pose):
    spec = RoboDojoMemorySpec(4, 2, use_pose)
    memory = CausalExecutionMemory(32, pose_dim=32 if use_pose else 0, width=16, heads=2, layers=1)
    record = spec.empty()
    records = {k: np.stack([v, v]) for k, v in record.items()}
    queries = torch.randn(2, 5, 32, requires_grad=True)
    out, summary, gate = memory(queries, records)
    torch.testing.assert_close(out, queries, atol=0, rtol=0)
    (out.square().mean() + summary.square().mean()).backward()
    assert torch.isfinite(out).all() and 0 <= gate <= 1
    assert memory.output_projection.weight.grad.abs().sum() > 0


def request(env="a", episode="1", step=0, feedback=None):
    return {"env_id": env, "episode_id": episode, "env_step": step,
            "raw_state": np.arange(14, dtype=np.float32), "memory_feedback": feedback or [],
            "image": [], "lang": "move"}


def feedback(start, stop):
    return [{"source_step": step, "state": np.full(14, step, np.float32),
             "executed_action": np.full(14, step + 100, np.float32)} for step in range(start, stop)]


def test_server_tracks_env_id_reset_feedback_and_transaction():
    sessions = RoboDojoMemorySessions(RoboDojoMemorySpec(3, 4))
    normalize = lambda x: x[None] / 10
    _, update = sessions.prepare([request("a"), request("b")], normalize)
    sessions.commit(update)
    requests = [request("b", step=16, feedback=feedback(0, 16)), request("a", step=8, feedback=feedback(0, 8))]
    examples, update = sessions.prepare(requests, normalize)
    np.testing.assert_array_equal(examples[0]["memory_records"]["executed_actions"][-1], np.full(14, 112))
    assert examples[1]["memory_contract"]["source_frame_index"] == 8
    np.testing.assert_array_equal(examples[0]["state"], np.arange(14, dtype=np.float32)[None] / 10)
    # Failed/uncommitted inference is retriable with exactly the same feedback.
    sessions.prepare(requests, normalize)
    sessions.commit(update)
    with pytest.raises(ValueError, match="strictly before"):
        sessions.prepare([request("a", step=16)], normalize)
    with pytest.raises(ValueError, match="every step"):
        sessions.prepare([request("a", step=10, feedback=feedback(9, 10))], normalize)
    reset, update = sessions.prepare([request("a", episode="2")], normalize)
    assert not reset[0]["memory_records"]["valid_mask"].any()
    sessions.commit(update)
    sessions.reset()
    assert not sessions.sessions
    with pytest.raises(ValueError, match="legacy stateless"):
        sessions.prepare([{"image": [], "lang": "move"}], normalize)


def test_spatial_prediction_independent_of_geometry_and_action_labels():
    # Real spatial and memory modules, small dimensions; omit only the VLM.
    torch.manual_seed(7)
    spec = RoboDojoMemorySpec(3, 4)
    model = tiny_model(spec)
    model.execution_memory = CausalExecutionMemory(32, width=16, heads=2, layers=1)
    model.memory_waypoint_projection = torch.nn.Linear(16, 32)
    model.spatial_enabled = True
    model.waypoint_model = WaypointDecoder(32, 16, 1, 2, 32, [1, 2])
    model.spatial_adapter = SpatialAdapter(32, 16, 2)
    # Nonzero adapters ensure the test is sensitive to accidental label inputs.
    torch.nn.init.normal_(model.execution_memory.output_projection.weight, std=.1)
    torch.nn.init.normal_(model.spatial_adapter.up.weight, std=.1)
    model.eval()
    base = {"memory_records": spec.empty(), "memory_contract": spec.contract(0)}
    geometry = {"current_pose": np.tile(np.eye(4), (2, 1, 1)),
                "future_pose": np.tile(np.eye(4), (2, 2, 1, 1)), "future_valid": np.ones((2, 2), bool)}
    labeled = {**base, "geometry": geometry, "action": np.ones((50, 14))}
    hidden, queries = torch.randn(1, 8, 32), torch.randn(1, 5, 32)
    with torch.no_grad():
        unlabeled = model._spatial_forward(hidden, queries, [base])[0]
        training_path = model._spatial_forward(hidden, queries, [labeled])[0]
        geometry["future_pose"][..., :3, 3] = 999
        labeled["action"] *= 999
        poisoned = model._spatial_forward(hidden, queries, [labeled])[0]
    torch.testing.assert_close(unlabeled, training_path, atol=0, rtol=0)
    torch.testing.assert_close(unlabeled, poisoned, atol=0, rtol=0)


def test_saved_state_transform_matches_training_pipeline(monkeypatch):
    from deployment.model_server import policy_norm_processor as serving
    from examples.simBenchmarks.RoboDojo.train_files.data_registry.data_config import RoboDojoArxX5H50Q99DataConfig
    config = RoboDojoArxX5H50Q99DataConfig()
    low = np.linspace(-2, 0, 14, dtype=np.float32)
    high = np.linspace(1, 4, 14, dtype=np.float32)
    high[6] = low[6]  # Exercise constant-channel handling too.
    stats = {name: {"q01": low.tolist(), "q99": high.tolist(),
                    "min": low.tolist(), "max": high.tolist(),
                    "mean": ((low + high) / 2).tolist(), "std": np.ones(14).tolist()}
             for name in ("state", "action")}
    monkeypatch.setattr(serving, "read_mode_config", lambda _: (
        {"datasets": {"vla_data": {"data_mix": "robodojo_v30_all_h50_q99"}}}, {"robot": stats}))
    proc = serving.PolicyNormProcessor("unused.pt")
    training_transform = config.transform()
    metadata = serving._build_dataset_metadata(stats, config.embodiment_tag,
        config.action_keys, config.state_keys, config.action_key_dims, config.state_key_dims)
    training_transform.set_metadata(metadata)
    raw = np.linspace(-3, 5, 14, dtype=np.float32)
    data, cursor = {}, 0
    for keys, dims in ((config.state_keys, config.state_key_dims), (config.action_keys, config.action_key_dims)):
        cursor = 0
        for key in keys:
            data[key] = raw[None, cursor:cursor + dims[key]].copy()
            cursor += dims[key]
    transformed = training_transform(data)
    expected = torch.cat([transformed[key] for key in config.state_keys], -1).numpy()
    np.testing.assert_array_equal(proc.apply_state(raw), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cast_parameters", [False, True])
def test_cuda_autocast_fused_memory_backward(cast_parameters):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    memory = CausalExecutionMemory(32, width=16, heads=2, layers=1).cuda()
    if cast_parameters:
        memory.to(dtype)
    buffer = RoboDojoMemoryBuffer(4, 2)
    for step in range(8):
        buffer.append(np.full(14, step / 10), np.ones(14), source_step=step)
    record = buffer.build(8)
    records = {k: np.stack([v, v]) for k, v in record.items()}
    queries = torch.randn(2, 5, 32, device="cuda", dtype=dtype if cast_parameters else torch.float32, requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        out, summary, _ = memory(queries, records)
        loss = out.float().square().mean() + summary.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in memory.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("step", [True, 0.5, 0.0, "0", -1])
def test_reject_noninteger_feedback_steps(step):
    buffer = RoboDojoMemoryBuffer()
    with pytest.raises(ValueError):
        buffer.append(np.zeros(14), np.zeros(14), source_step=step)
    with pytest.raises(ValueError):
        RoboDojoMemorySessions(buffer.spec).prepare([request(step=step)], lambda x: x[None])
