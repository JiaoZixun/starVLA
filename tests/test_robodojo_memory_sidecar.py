"""Linked preprocessing must preserve causal online/offline semantics."""
import json

import numpy as np
import pandas as pd
import pytest

from starVLA.utils.robodojo_memory import RoboDojoMemoryBuffer, RoboDojoMemorySpec
from starVLA.utils.robodojo_memory_sidecar import (
    CAMERAS, RoboDojoMemorySidecar, build_sidecar, sha256, validate_sidecar,
)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "original"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"features": {key: {"shape": [14]} for key in ("observation.state", "action")},
            "total_episodes": 2, "total_frames": 201}
    modality = {name: {key: {"start": start, "end": end, "original_key": original}
                      for key, start, end in (("left_joints", 0, 6), ("left_gripper", 6, 7),
                                              ("right_joints", 7, 13), ("right_gripper", 13, 14))}
                for name, original in (("state", "observation.state"), ("action", "action"))}
    modality["video"] = {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/modality.json").write_text(json.dumps(modality))
    rows = []
    for episode, length in ((5, 180), (8, 21)):
        for step in range(length):
            rows.append({"episode_index": episode, "frame_index": step,
                         "observation.state": np.full(14, step / 10, np.float32),
                         "action": np.arange(14, dtype=np.float32) + step})
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    return root


def test_linked_indices_match_online_every_frame_without_copying_original(source, tmp_path):
    spec = RoboDojoMemorySpec()
    output = tmp_path / "memory/v1"
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    build_sidecar(source, output, spec, workers=2)
    validate_sidecar(output, source, spec)
    assert (output / "source_dataset").is_symlink()
    assert (output / "source_dataset").resolve() == source
    # Explicit file list: only derived indices, manifest and one source symlink.
    assert sorted(p.name for p in output.iterdir()) == ["episodes", "manifest.json", "source_dataset"]
    assert sorted(p.suffix for p in (output / "episodes").iterdir()) == [".npy", ".npy"]
    assert before == {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    reader = RoboDojoMemorySidecar(output, source, spec)
    data = pd.read_parquet(source / "data/chunk-000/file-000.parquet")
    for episode, group in data.groupby("episode_index"):
        group = group.reset_index(drop=True)
        indices = reader.indices(episode, len(group))
        buffer = RoboDojoMemoryBuffer()
        for step in range(len(group)):
            online = buffer.build(step)
            np.testing.assert_array_equal(indices[step], online["source_frame_index"])
            valid = indices[step] >= 0
            for key, column in (("state", "observation.state"), ("executed_actions", "action")):
                np.testing.assert_array_equal(online[key][valid], np.stack(group.iloc[indices[step][valid]][column])
                                              if valid.any() else np.empty((0, 14)))
            buffer.append(group.iloc[step]["observation.state"], group.iloc[step]["action"], source_step=step)
    with pytest.raises(FileExistsError):
        build_sidecar(source, output, spec)


@pytest.mark.parametrize("tamper", ["frame_gap", "nonfinite", "wrong_dim", "channel_order", "counts"])
def test_bad_source_is_rejected_without_publishing_partial_output(source, tmp_path, tamper):
    path = source / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    if tamper == "frame_gap":
        data.loc[5, "frame_index"] = 99
    elif tamper in {"nonfinite", "wrong_dim"}:
        data.at[5, "action"] = (np.full(14, np.nan, dtype=np.float32) if tamper == "nonfinite"
                                else np.ones(7, dtype=np.float32))
    elif tamper == "counts":
        path_info = source / "meta/info.json"
        info = json.loads(path_info.read_text())
        info["total_frames"] += 1
        path_info.write_text(json.dumps(info))
    else:
        path_modality = source / "meta/modality.json"
        modality = json.loads(path_modality.read_text())
        modality["action"]["right_joints"]["start"] = 0
        path_modality.write_text(json.dumps(modality))
    data.to_parquet(path, index=False)
    output = tmp_path / "memory/v1"
    with pytest.raises(ValueError):
        build_sidecar(source, output)
    assert not output.exists()
    assert not list(output.parent.glob("*.building-*"))


@pytest.mark.parametrize("tamper", ["source", "metadata", "indices", "future", "row_link", "spec", "link"])
def test_changed_source_and_invalid_links_or_future_indices_fail(source, tmp_path, tamper):
    output = tmp_path / "memory/v1"
    manifest = build_sidecar(source, output)
    if tamper == "source":
        with (source / "data/chunk-000/file-000.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "metadata":
        with (source / "meta/info.json").open("a") as stream:
            stream.write(" ")
    elif tamper in {"indices", "future"}:
        path = output / manifest["episodes"]["5"]["file"]
        values = np.load(path)
        values[16, -1] = 16  # Current command must never be a history input.
        np.save(path, values)
        if tamper == "future":
            manifest["episodes"]["5"]["sha256"] = sha256(path)
    elif tamper == "row_link":
        manifest["episodes"]["5"]["source_row_start"] += 1
        manifest["episodes"]["5"]["source_row_stop"] += 1
    elif tamper == "spec":
        manifest["decision_stride_env_steps"] = 4
    else:
        (output / "source_dataset").unlink()
        (output / "source_dataset").symlink_to(tmp_path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        validate_sidecar(output, source)


def test_output_inside_original_is_rejected(source):
    with pytest.raises(ValueError, match="outside"):
        build_sidecar(source, source / "memory")


def test_source_adapter_can_link_original_data_and_metadata(source, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in ("data", "meta"):
        (adapter / name).symlink_to(source / name, target_is_directory=True)
    output = tmp_path / "memory/v1"
    build_sidecar(adapter, output)
    manifest = validate_sidecar(output, adapter)
    assert manifest["total_frames"] == 201
    assert (output / "source_dataset").resolve() == adapter


def test_training_dataset_uses_linked_raw_rows_and_matches_online(source, tmp_path):
    from starVLA.dataloader.robodojo_memory import RoboDojoMemorySingleDataset

    output = tmp_path / "memory/v1"
    spec = RoboDojoMemorySpec()
    build_sidecar(source, output, spec)
    dataset = object.__new__(RoboDojoMemorySingleDataset)
    dataset.memory_spec = spec
    dataset._memory_sidecar = RoboDojoMemorySidecar(output, source, spec)
    data = pd.read_parquet(source / "data/chunk-000/file-000.parquet")
    for episode, group in data.groupby("episode_index"):
        dataset.curr_traj_data = group.reset_index(drop=True)
        buffer = RoboDojoMemoryBuffer()
        for step in range(len(group)):
            expected = buffer.build(step)
            actual = dataset._memory_records(episode, step)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])
            row = dataset.curr_traj_data.iloc[step]
            buffer.append(row["observation.state"], row["action"], source_step=step)
