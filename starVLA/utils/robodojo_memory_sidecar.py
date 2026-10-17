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
from functools import lru_cache

import numpy as np

from starVLA.utils.robodojo_memory import MEMORY_VISUAL_SHAPE, image_to_memory_tokens, RoboDojoMemorySpec


CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHANNELS = ["left_joints_0:6", "left_gripper", "right_joints_0:6", "right_gripper"]
REPRESENTATION = "raw_dual_arm_abs_qpos_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1024)
def _cached_sha256(path):
    """Hash each immutable source video at most once per preprocessing run."""
    return sha256(Path(path))


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


def _episode_metadata(source):
    import pandas as pd
    files = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    if not files:
        return {}
    table = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return {int(row["episode_index"]): row.to_dict() for _, row in table.iterrows()}


def _video_path(source, camera, chunk, file_index):
    return source / "videos" / f"observation.images.{camera}" / f"chunk-{int(chunk):03d}" / f"file-{int(file_index):03d}.mp4"


def _decode_video_at_timestamps(path, timestamps, episode, camera):
    """Select each nearest source frame using the training loader's timestamp contract."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to preprocess RoboDojo visual memory sidecars") from exc
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise ValueError("Video timestamps must be a nonempty finite vector")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Video timestamps must be ordered")
    output = np.empty((len(timestamps), MEMORY_VISUAL_SHAPE[1], *MEMORY_VISUAL_SHAPE[2:]), dtype=np.uint8)
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        # Start one second early so timestamp rounding at an episode boundary
        # cannot skip the closest frame. Seek lands on an earlier keyframe.
        target_pts = max(0, int((timestamps[0] - 1.0) / time_base))
        try:
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            seeked = True
        except (av.error.FFmpegError, ValueError, TypeError):
            seeked = False
        iterator = iter(container.decode(stream))
        previous = None

        def next_frame():
            for frame in iterator:
                if frame.pts is None:
                    continue
                if seeked and frame.pts < target_pts:
                    continue
                return frame
            return None

        current = next_frame()
        for index, timestamp in enumerate(timestamps):
            while current is not None and float(current.pts) * time_base < timestamp:
                previous, current = current, next_frame()
            candidates = [frame for frame in (previous, current) if frame is not None]
            if not candidates:
                raise ValueError(f"No video frame for episode {episode}/{camera} at {timestamp:.6f}s")
            closest = min(candidates, key=lambda frame: abs(float(frame.pts) * time_base - timestamp))
            output[index] = image_to_memory_tokens(closest.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return output


def _decode_episode_visual(source, metadata, episode, length, episode_timestamps):
    """Decode three AV1 streams into patches aligned to parquet observation timestamps."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to preprocess RoboDojo visual memory sidecars") from exc
    episode_timestamps = np.asarray(episode_timestamps, dtype=np.float64)
    if episode_timestamps.shape != (length,) or not np.isfinite(episode_timestamps).all():
        raise ValueError(f"Episode {episode} must have {length} finite timestamps")
    output = np.empty((length, *MEMORY_VISUAL_SHAPE), dtype=np.uint8)
    hashes = {}
    for camera in CAMERAS:
        path = _video_path(source, camera, metadata[f"videos/observation.images.{camera}/chunk_index"],
                           metadata[f"videos/observation.images.{camera}/file_index"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing source video for episode {episode}: {path}")
        # RoboDojo stores many episodes in one long video file.  Cache the
        # immutable-file hash so adjacent episodes do not rescan the same
        # multi-gigabyte source over and over.
        hashes[camera] = _cached_sha256(str(path))
        start_ts = float(metadata[f"videos/observation.images.{camera}/from_timestamp"])
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            rate = float(stream.average_rate or 25.0)
            start = max(0, int(round(start_ts * rate)))
            # Seek to the keyframe before this episode instead of decoding
            # the complete preceding history in the same source video.
            # The timestamp is expressed in the stream's own time base.
            target_pts = int(round(start_ts / float(stream.time_base)))
            try:
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                seeked = True
            except (av.error.FFmpegError, ValueError, TypeError):
                seeked = False
            decoded = 0
            seen = 0
            for index, frame in enumerate(container.decode(stream)):
                if seeked:
                    # Seek lands on a preceding keyframe.  Drop frames before
                    # the requested episode boundary using exact PTS when
                    # available; frame-count fallback covers unusual codecs.
                    if frame.pts is not None:
                        if frame.pts < target_pts:
                            continue
                    elif seen < start:
                        seen += 1
                        continue
                elif index < start:
                    continue
                if decoded >= length:
                    break
                rgb = frame.to_ndarray(format="rgb24")
                output[decoded, CAMERAS.index(camera)] = image_to_memory_tokens(rgb)
                decoded += 1
                seen += 1
        finally:
            container.close()
        if decoded != length:
            # LeRobot maps every parquet timestamp to the nearest video frame;
            # episode video segments can differ from parquet row counts by one.
            # Reuse that exact rule instead of rejecting or shifting the history.
            print(f"Timestamp-aligning episode {episode}/{camera}: sequential frames {decoded}/{length}", flush=True)
            targets = episode_timestamps + start_ts
            output[:, CAMERAS.index(camera)] = _decode_video_at_timestamps(path, targets, episode, camera)
    return output, hashes


def _build_file(source, path, target, spec, episode_meta=None, visual=False, resume=False):
    import pandas as pd

    before = path.stat()
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    if visual:
        columns.append("timestamp")
    data = pd.read_parquet(path, columns=columns)
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
        expected_indices = history_indices(length, spec)
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Duplicate episode sidecar: {destination}")
            existing_indices = np.load(destination, mmap_mode="r", allow_pickle=False)
            if existing_indices.dtype != np.int32 or not np.array_equal(existing_indices, expected_indices):
                raise ValueError(f"Invalid resumable memory indices: {destination}")
        else:
            temporary_index = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            with temporary_index.open("xb") as stream:
                np.save(stream, expected_indices, allow_pickle=False)
            temporary_index.replace(destination)
        visual_meta = None
        if visual:
            if episode_meta is None or episode not in episode_meta:
                raise ValueError(f"Missing episode metadata for visual episode {episode}")
            visual_destination = target / f"episodes/{episode:06d}.visual.npy"
            visual_values = None
            if visual_destination.exists() and resume:
                try:
                    existing_visual = np.load(visual_destination, mmap_mode="r", allow_pickle=False)
                    if existing_visual.dtype == np.uint8 and existing_visual.shape == (length, *MEMORY_VISUAL_SHAPE):
                        visual_values = existing_visual
                    else:
                        visual_destination.unlink()
                except (OSError, ValueError):
                    visual_destination.unlink(missing_ok=True)
            elif visual_destination.exists():
                raise FileExistsError(f"Duplicate episode visual sidecar: {visual_destination}")
            video_hashes = {}
            if visual_values is None:
                visual_values, video_hashes = _decode_episode_visual(
                    source, episode_meta[episode], episode, length, group["timestamp"].to_numpy()
                )
                temporary_visual = visual_destination.with_name(f".{visual_destination.name}.{uuid.uuid4().hex}.tmp")
                with temporary_visual.open("xb") as stream:
                    np.save(stream, visual_values, allow_pickle=False)
                temporary_visual.replace(visual_destination)
            else:
                for camera in CAMERAS:
                    video_path = _video_path(source, camera,
                        episode_meta[episode][f"videos/observation.images.{camera}/chunk_index"],
                        episode_meta[episode][f"videos/observation.images.{camera}/file_index"])
                    video_hashes[camera] = _cached_sha256(str(video_path))
            visual_meta = {"file": str(visual_destination.relative_to(target)), "sha256": sha256(visual_destination),
                           "visual_shape": list(visual_values.shape[1:]), "visual_dtype": "uint8",
                           "video_sha256": video_hashes}
        episodes[str(episode)] = {
            "length": length, "file": name, "sha256": sha256(destination),
            "source_file": relative, "source_row_start": int(rows[0]),
            "source_row_stop": int(rows[-1] + 1),
            **({"visual": visual_meta} if visual_meta else {}),
        }
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during preprocessing: {path}")
    return relative, {"sha256": digest, "size_bytes": after.st_size, "rows": len(data)}, episodes


def build_sidecar(source, output, spec=RoboDojoMemorySpec(), workers=3, visual=True, resume_from=None):
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
    resuming = resume_from is not None
    if resuming:
        temporary = Path(resume_from).resolve()
        if not temporary.is_dir() or not (temporary / "episodes").is_dir():
            raise ValueError(f"Resume directory must contain episodes/: {temporary}")
        if (temporary / "manifest.json").exists() or (temporary / "source_dataset").exists():
            raise ValueError("Resume directory already contains published metadata")
        for stale in (temporary / "episodes").glob(".*.tmp"):
            stale.unlink()
    else:
        temporary = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
        temporary.mkdir()
    try:
        (temporary / "episodes").mkdir(exist_ok=True)
        metadata_names = ["meta/info.json", "meta/modality.json"]
        metadata_names += [name for name in ("meta/stats.json", "meta/stats_gr00t.json", "meta/tasks.parquet")
                           if (source / name).is_file()]
        metadata_names += [str(p.relative_to(source)) for p in sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))]
        episode_meta = _episode_metadata(source) if visual else {}
        has_visual = visual and bool(episode_meta) and all((_video_path(source, camera,
            next(iter(episode_meta.values()))[f"videos/observation.images.{camera}/chunk_index"],
            next(iter(episode_meta.values()))[f"videos/observation.images.{camera}/file_index"])).exists() for camera in CAMERAS)
        # Small synthetic/index-only fixtures have no videos; retain v1 behavior for them.
        visual = bool(has_visual)
        manifest = {
            "format": "robodojo_causal_memory_indices", "format_version": 2 if visual else 1,
            # Keep the published artifact relocatable with its original data.
            "source": os.path.relpath(source, output), "source_link": "source_dataset",
            "representation": REPRESENTATION, "channel_order": CHANNELS, "camera_order": CAMERAS,
            "recent_records": spec.recent_records, "decision_stride_env_steps": spec.decision_stride,
            "use_pose": spec.use_pose, "padding": -1,
            "record_semantics": "pre_step_raw_state_and_recorded_abs_qpos_command_after_step_completed",
            "normalization": "none_for_history; current_state_and_action_use_original_training_q99",
            "metadata_sha256": {name: sha256(source / name) for name in metadata_names},
            "source_files": {}, "episodes": {},
        }
        if visual:
            manifest.update({"visual_shape": list(MEMORY_VISUAL_SHAPE), "visual_dtype": "uint8",
                             "visual_encoding": "ordered_rgb_2x4_box_8x8", "visual_sidecar": True})
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
            results = pool.map(lambda path: _build_file(source, path, temporary, spec, episode_meta, visual, resuming), files)
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
        if resuming:
            print(f"Resume failed; preserving partial sidecar at {temporary}", flush=True)
        else:
            shutil.rmtree(temporary)
        raise
    return manifest


class RoboDojoMemorySidecar:
    """Validate source identity once, then map only the active episode's indices."""

    def __init__(self, root, source, spec):
        self.root, self.source, self.spec = Path(root).resolve(), Path(source).resolve(), spec
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (m.get("format"), m.get("format_version"), m.get("representation"), m.get("padding")) not in (
                ("robodojo_causal_memory_indices", 1, REPRESENTATION, -1),
                ("robodojo_causal_memory_indices", 2, REPRESENTATION, -1)):
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
        self._verified_visuals = set()
        self._episode = None
        self._indices = None
        self._visual_episode = None
        self._visual_values = None
        self.visual = bool(m.get("format_version") == 2 and m.get("visual_sidecar"))
        if self.visual:
            if m.get("camera_order") != CAMERAS or m.get("visual_shape") != list(MEMORY_VISUAL_SHAPE):
                raise ValueError("Visual memory manifest shape/camera contract differs")
            if m.get("visual_dtype") != "uint8" or m.get("visual_encoding") != "ordered_rgb_2x4_box_8x8":
                raise ValueError("Unsupported visual memory encoding")

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

    def visual_tokens(self, episode, length):
        if not self.visual:
            return np.zeros((length, *MEMORY_VISUAL_SHAPE), dtype=np.uint8)
        episode = int(episode)
        if self._visual_episode == episode:
            if self._visual_values.shape[0] != length:
                raise ValueError("Memory visual episode length differs from source")
            return self._visual_values
        entry = self.manifest["episodes"][str(episode)]
        if int(entry["length"]) != int(length):
            raise ValueError("Memory visual episode length differs from source")
        visual_entry = entry.get("visual")
        if not visual_entry:
            raise ValueError("Visual memory entry missing")
        path = _inside(self.root, visual_entry["file"])
        if episode not in self._verified_visuals:
            if sha256(path) != visual_entry["sha256"]:
                raise ValueError(f"Memory visual sidecar changed: {path}")
            self._verified_visuals.add(episode)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint8 or tuple(values.shape) != (length, *MEMORY_VISUAL_SHAPE):
            raise ValueError(f"Invalid memory visual sidecar shape/dtype: {values.shape}/{values.dtype}")
        self._visual_episode, self._visual_values = episode, values
        return values


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
            if reader.visual:
                visual_entry = entry.get("visual")
                if not visual_entry:
                    raise ValueError(f"Episode {episode} is missing visual memory metadata")
                visual_path = _inside(reader.root, visual_entry["file"])
                if sha256(visual_path) != visual_entry["sha256"]:
                    raise ValueError(f"Memory visual sidecar changed: {visual_path}")
                values = np.load(visual_path, mmap_mode="r", allow_pickle=False)
                if values.dtype != np.uint8 or tuple(values.shape) != (entry["length"], *MEMORY_VISUAL_SHAPE):
                    raise ValueError(f"Invalid memory visual sidecar shape/dtype for episode {episode}: {values.shape}/{values.dtype}")
            covered[start:stop] = True
            total += entry["length"]
        if not covered.all():
            raise ValueError("Memory manifest does not cover every original source row")
    if total != manifest["total_frames"] or len(manifest["episodes"]) != manifest["total_episodes"]:
        raise ValueError("Memory manifest totals differ")
    print(f"Validated {total} frames, {len(manifest['episodes'])} episodes; original data unchanged", flush=True)
    return manifest
