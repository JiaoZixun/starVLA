"""Derived causal indices linked to immutable RoboDojo source parquet.

Only history frame indices are materialized. Images, state, commands and
normalization statistics always come from the original dataset.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from starVLA.utils.robodojo_memory import RoboDojoMemorySpec


CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHANNELS = ["left_joints_0:6", "left_gripper", "right_joints_0:6", "right_gripper"]
REPRESENTATION = "raw_dual_arm_abs_qpos_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def history_indices(length, spec):
    indices = np.arange(length, dtype=np.int64)[:, None] - spec.decision_stride * np.arange(
        spec.recent_records, 0, -1, dtype=np.int64)[None]
    if length > np.iinfo(np.int32).max:
        raise ValueError("Episode too long for int32 frame indices")
    return np.maximum(indices, -1).astype(np.int32)


def _inside(root, relative, allow_symlinks=False):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path escapes sidecar/source root: {relative}")
    path = root / relative
    if not allow_symlinks and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes sidecar/source root: {relative}")
    return path


def _identity(manifest):
    content = {key: manifest[key] for key in ("metadata_sha256", "source_files")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _source_metadata(source):
    info = json.loads((source / "meta/info.json").read_text())
    modality = json.loads((source / "meta/modality.json").read_text())
    for feature in ("observation.state", "action"):
        if info["features"][feature]["shape"] != [14]:
            raise ValueError(f"Expected raw 14D {feature}")
    for name, original in (("state", "observation.state"), ("action", "action")):
        for key, start, end in (("left_joints", 0, 6), ("left_gripper", 6, 7),
                                ("right_joints", 7, 13), ("right_gripper", 13, 14)):
            entry = modality[name][key]
            if (entry["start"], entry["end"], entry["original_key"]) != (start, end, original):
                raise ValueError("Source channel order differs from the online dual-arm contract")
    for camera in CAMERAS:
        if modality["video"][camera]["original_key"] != f"observation.images.{camera}":
            raise ValueError("Source camera mapping differs from the online contract")
    return info


def _build_file(source, path, target, spec):
    import pandas as pd

    before = path.stat()
    data = pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state", "action"])
    relative = str(path.relative_to(source))
    episodes = {}
    for episode, group in data.groupby("episode_index", sort=False):
        if not isinstance(episode, (int, np.integer)) or episode < 0:
            raise ValueError("Episode IDs must be nonnegative integers")
        episode = int(episode)
        length = len(group)
        frames = group["frame_index"].to_numpy()
        if not np.issubdtype(frames.dtype, np.integer) or not np.array_equal(frames, np.arange(length)):
            raise ValueError(f"Episode {episode} must have contiguous ordered frame indices starting at zero")
        rows = data.index.get_indexer(group.index)
        if not np.array_equal(rows, np.arange(rows[0], rows[0] + length)):
            raise ValueError(f"Episode {episode} must occupy contiguous source rows")
        for key in ("observation.state", "action"):
            values = np.stack(group[key].to_numpy())
            if values.shape != (length, 14) or not np.isfinite(values).all():
                raise ValueError(f"Episode {episode}: {key} must be finite raw [N,14]")
        name = f"episodes/{episode:06d}.npy"
        destination = target / name
        # Fail on duplicates, including episodes split between source shards.
        with destination.open("xb") as stream:
            np.save(stream, history_indices(length, spec), allow_pickle=False)
        episodes[str(episode)] = {
            "length": length, "file": name, "sha256": sha256(destination),
            "source_file": relative, "source_row_start": int(rows[0]),
            "source_row_stop": int(rows[-1] + 1),
        }
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during preprocessing: {path}")
    return relative, {"sha256": digest, "size_bytes": after.st_size, "rows": len(data)}, episodes


def build_sidecar(source, output, spec=RoboDojoMemorySpec(), workers=3):
    """Publish a complete version atomically; never edit/duplicate source data."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("Memory sidecar must be outside the original dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; validate it or choose a new version: {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    info = _source_metadata(source)
    files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError("No source parquet shards found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        (temporary / "episodes").mkdir()
        metadata_names = ["meta/info.json", "meta/modality.json"]
        metadata_names += [name for name in ("meta/stats.json", "meta/stats_gr00t.json", "meta/tasks.parquet")
                           if (source / name).is_file()]
        metadata_names += [str(p.relative_to(source)) for p in sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))]
        manifest = {
            "format": "robodojo_causal_memory_indices", "format_version": 1,
            "source": str(source), "source_link": "source_dataset",
            "representation": REPRESENTATION, "channel_order": CHANNELS, "camera_order": CAMERAS,
            "recent_records": spec.recent_records, "decision_stride_env_steps": spec.decision_stride,
            "use_pose": spec.use_pose, "padding": -1,
            "record_semantics": "pre_step_raw_state_and_recorded_abs_qpos_command_after_step_completed",
            "normalization": "none_for_history; current_state_and_action_use_original_training_q99",
            "metadata_sha256": {name: sha256(source / name) for name in metadata_names},
            "source_files": {}, "episodes": {},
        }
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
            results = pool.map(lambda path: _build_file(source, path, temporary, spec), files)
            for relative, meta, episodes in results:
                if manifest["episodes"].keys() & episodes.keys():
                    raise ValueError("Episode spans multiple source shards")
                manifest["source_files"][relative] = meta
                manifest["episodes"].update(episodes)
                print(f"Indexed {relative}: {len(episodes)} episodes, {meta['rows']} frames", flush=True)
        manifest["total_episodes"] = len(manifest["episodes"])
        manifest["total_frames"] = sum(e["length"] for e in manifest["episodes"].values())
        if (manifest["total_episodes"], manifest["total_frames"]) != (info["total_episodes"], info["total_frames"]):
            raise ValueError("Source counts differ from meta/info.json")
        if any(sha256(source / name) != digest for name, digest in manifest["metadata_sha256"].items()):
            raise ValueError("Source metadata changed during preprocessing")
        manifest["source_identity_sha256"] = _identity(manifest)
        (temporary / "source_dataset").symlink_to(os.path.relpath(source, output), target_is_directory=True)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


class RoboDojoMemorySidecar:
    """Validate source identity once, then map only the active episode's indices."""

    def __init__(self, root, source, spec):
        self.root, self.source, self.spec = Path(root).resolve(), Path(source).resolve(), spec
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (m.get("format"), m.get("format_version"), m.get("representation"), m.get("padding")) != (
                "robodojo_causal_memory_indices", 1, REPRESENTATION, -1):
            raise ValueError("Unsupported memory sidecar format")
        if (m.get("recent_records"), m.get("decision_stride_env_steps"), m.get("use_pose")) != (
                spec.recent_records, spec.decision_stride, spec.use_pose):
            raise ValueError("Memory sidecar and model/dataset contracts differ")
        if m.get("camera_order") != CAMERAS or m.get("channel_order") != CHANNELS:
            raise ValueError("Memory sidecar channel/camera order differs")
        if m.get("source_link") != "source_dataset" or (self.root / "source_dataset").resolve() != self.source:
            raise ValueError("Memory sidecar must link to the configured original dataset")
        if m.get("source_identity_sha256") != _identity(m):
            raise ValueError("Invalid source identity in memory manifest")
        for name, digest in m["metadata_sha256"].items():
            if sha256(_inside(self.source, name, allow_symlinks=True)) != digest:
                raise ValueError(f"Source metadata changed: {name}")
        self._verified_sources = set()
        self._episode = None
        self._indices = None

    def indices(self, episode, length):
        if self._episode == episode:
            if self._indices.shape[0] != length:
                raise ValueError("Memory sidecar episode length differs from source")
            return self._indices
        entry = self.manifest["episodes"][str(int(episode))]
        if entry["length"] != length or entry["source_row_stop"] - entry["source_row_start"] != length:
            raise ValueError("Memory sidecar episode length differs from source")
        relative = entry["source_file"]
        if relative not in self._verified_sources:
            path, meta = _inside(self.source, relative, allow_symlinks=True), self.manifest["source_files"][relative]
            if path.stat().st_size != meta["size_bytes"] or sha256(path) != meta["sha256"]:
                raise ValueError(f"Source parquet changed: {relative}")
            self._verified_sources.add(relative)
        path = _inside(self.root, entry["file"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Memory indices changed: {path}")
        indices = np.load(path, mmap_mode="r", allow_pickle=False)
        if indices.dtype != np.int32 or not np.array_equal(indices, history_indices(length, self.spec)):
            raise ValueError("Memory indices violate the strictly-past stride grid")
        self._episode, self._indices = episode, indices
        return indices


def validate_sidecar(root, source, spec=RoboDojoMemorySpec()):
    """Check every derived index, original shard checksum and episode row link."""
    import pandas as pd

    reader = RoboDojoMemorySidecar(root, source, spec)
    manifest = reader.manifest
    total = 0
    for relative, meta in manifest["source_files"].items():
        rows = pd.read_parquet(_inside(reader.source, relative, allow_symlinks=True), columns=["episode_index", "frame_index"])
        if len(rows) != meta["rows"]:
            raise ValueError("Source row count changed")
        covered = np.zeros(len(rows), bool)
        for episode, entry in manifest["episodes"].items():
            if entry["source_file"] != relative:
                continue
            start, stop = entry["source_row_start"], entry["source_row_stop"]
            if not 0 <= start < stop <= len(rows) or covered[start:stop].any():
                raise ValueError("Overlapping or invalid source row links")
            group = rows.iloc[start:stop]
            if not (group["episode_index"] == int(episode)).all() or not np.array_equal(
                    group["frame_index"].to_numpy(), np.arange(entry["length"])):
                raise ValueError("Episode/frame links do not match original source rows")
            reader.indices(int(episode), entry["length"])
            covered[start:stop] = True
            total += entry["length"]
        if not covered.all():
            raise ValueError("Memory manifest does not cover every original source row")
    if total != manifest["total_frames"] or len(manifest["episodes"]) != manifest["total_episodes"]:
        raise ValueError("Memory manifest totals differ")
    print(f"Validated {total} frames, {len(manifest['episodes'])} episodes; original data unchanged", flush=True)
    return manifest
