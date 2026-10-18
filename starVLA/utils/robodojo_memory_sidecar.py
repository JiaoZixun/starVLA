"""Build and read linked causal visual-memory sidecars.

Sidecars contain only derived history indices and compact RGB patches.  The
original LeRobot dataset remains the source of state, action, metadata, and
full-resolution videos through a symbolic link.
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

from starVLA.dataloader.memory_visual import image_to_memory_tokens
from starVLA.utils.robodojo_memory import RoboDojoMemorySpec


CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CHANNELS = ["left_joints_0:6", "left_gripper", "right_joints_0:6", "right_gripper"]
REPRESENTATION = "raw_dual_arm_abs_qpos_v1"
VISUAL_SHAPE = (len(CAMERAS), 8, 8, 8, 3)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def history_indices(length, spec):
    indices = np.arange(length, dtype=np.int64)[:, None] - spec.decision_stride * np.arange(
        spec.recent_records, 0, -1, dtype=np.int64
    )[None]
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
    keys = ("metadata_sha256", "source_files", "source_inventory", "video_sources")
    return hashlib.sha256(json.dumps({key: manifest[key] for key in keys}, sort_keys=True).encode()).hexdigest()


def _indices_v2_identity(manifest):
    # Format v2 produced by the RoboDojo preprocessing bundle predates the
    # visual-directory format below. Keep its published identity contract.
    content = {key: manifest[key] for key in ("metadata_sha256", "source_files")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _source_inventory(source):
    return {
        "parquet": sorted(p.relative_to(source).as_posix() for p in (source / "data").glob("chunk-*/*.parquet")),
        "videos": sorted(p.relative_to(source).as_posix() for p in (source / "videos").rglob("*.mp4")),
    }


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


def _decode_visuals(source, info, episode, timestamps):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for Robotwin visual memory preprocessing") from exc
    length = len(timestamps)
    chunk = int(episode) // int(info["chunks_size"])
    result = np.empty((length, *VISUAL_SHAPE), dtype=np.uint8)
    video_sources = {}
    for camera_index, camera in enumerate(CAMERAS):
        path = source / info["video_path"].format(
            episode_chunk=chunk, episode_index=int(episode), video_key=f"observation.images.{camera}"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing source video: {path}")
        before = path.stat()
        digest = sha256(path)
        frame_times = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base) if stream.time_base is not None else 0.0
            if not np.isfinite(time_base) or time_base <= 0:
                raise ValueError(f"Video has no valid time base: {path}")
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError(f"Video frame is missing PTS: {path} frame {index}")
                frame_times.append(float(frame.pts * frame.time_base))
                if index < length:
                    result[index, camera_index] = image_to_memory_tokens(
                        frame.to_ndarray(format="rgb24"), resize_size=256
                    )
        if len(frame_times) < length:
            raise ValueError(f"Video frame count is shorter than parquet for {path}: {len(frame_times)} < {length}")
        frame_times = np.asarray(frame_times, dtype=np.float64)
        expected = np.asarray(timestamps, dtype=np.float64)
        tolerance = max(1e-4, 2.0 * time_base)
        if not np.all(np.diff(frame_times) > 0) or not np.allclose(
            frame_times[:length] - frame_times[0], expected - expected[0], rtol=0, atol=tolerance
        ):
            raise ValueError(f"Video PTS do not align with parquet timestamps: {path}")
        if len(frame_times) > length and np.any(frame_times[length:] <= expected[-1] + tolerance):
            raise ValueError(f"Extra video frames occur before the parquet episode end: {path}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source video changed during preprocessing: {path}")
        video_sources[path.relative_to(source).as_posix()] = {
            "sha256": digest, "size_bytes": after.st_size,
        }
    return result, video_sources


def _build_file(source, path, target, spec, info):
    import pandas as pd
    before = path.stat()
    data = pd.read_parquet(path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
    relative = path.relative_to(source).as_posix()
    episodes = {}
    for episode, group in data.groupby("episode_index", sort=False):
        episode, length = int(episode), len(group)
        frames = group["frame_index"].to_numpy()
        if not np.issubdtype(frames.dtype, np.integer) or not np.array_equal(frames, np.arange(length)):
            raise ValueError(f"Episode {episode} frame indices are not contiguous")
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        if not np.isfinite(timestamps).all() or (length > 1 and not np.all(np.diff(timestamps) > 0)):
            raise ValueError(f"Episode {episode} timestamps are not strictly increasing")
        rows = data.index.get_indexer(group.index)
        if not np.array_equal(rows, np.arange(rows[0], rows[0] + length)):
            raise ValueError(f"Episode {episode} rows are not contiguous")
        for key in ("observation.state", "action"):
            values = np.stack(group[key].to_numpy())
            if values.shape != (length, 14) or not np.isfinite(values).all():
                raise ValueError(f"Episode {episode}: {key} must be finite raw [N,14]")
        index_name = f"episodes/{episode:06d}.npy"
        with (target / index_name).open("xb") as stream:
            np.save(stream, history_indices(length, spec), allow_pickle=False)
        visual_name = f"visual/{episode:06d}.npy"
        visual_values, videos = _decode_visuals(source, info, episode, timestamps)
        with (target / visual_name).open("xb") as stream:
            np.save(stream, visual_values, allow_pickle=False)
        episodes[str(episode)] = {
            "length": length, "file": index_name, "sha256": sha256(target / index_name),
            "visual_file": visual_name, "visual_sha256": sha256(target / visual_name),
            "video_sources": videos, "source_file": relative,
            "source_row_start": int(rows[0]), "source_row_stop": int(rows[-1] + 1),
        }
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source parquet changed during preprocessing: {path}")
    return relative, {"sha256": digest, "size_bytes": after.st_size, "rows": len(data)}, episodes


def _check_layout(root, manifest):
    if {p.name for p in root.iterdir()} != {"manifest.json", "source_dataset", "episodes", "visual"}:
        raise ValueError("Sidecar contains unexpected files")
    if not (root / "source_dataset").is_symlink():
        raise ValueError("source_dataset must be a symlink")
    expected = {"episodes": set(), "visual": set()}
    for entry in manifest["episodes"].values():
        expected["episodes"].add(entry["file"])
        expected["visual"].add(entry["visual_file"])
    for directory, names in expected.items():
        actual = {p.relative_to(root).as_posix() for p in (root / directory).rglob("*") if p.is_file()}
        if actual != names:
            raise ValueError(f"Sidecar {directory} file set differs from manifest")


def _check_indices_v2_layout(root, manifest):
    if {path.name for path in root.iterdir()} != {"manifest.json", "source_dataset", "episodes"}:
        raise ValueError("Indices v2 sidecar contains unexpected files")
    if not (root / "source_dataset").is_symlink():
        raise ValueError("source_dataset must be a symlink")
    expected = set()
    for entry in manifest["episodes"].values():
        expected.add(entry["file"])
        visual = entry.get("visual")
        if visual:
            expected.add(visual["file"])
    actual = {path.relative_to(root).as_posix() for path in (root / "episodes").rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Indices v2 sidecar episode files differ from manifest")


def build_sidecar(source, output, spec=RoboDojoMemorySpec(), workers=3):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("Sidecar must be outside the original dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    info = _source_metadata(source)
    files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise ValueError("No source parquet shards found")
    temporary = output.with_name(f".{output.name}.building-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        (temporary / "episodes").mkdir()
        (temporary / "visual").mkdir()
        metadata_names = ["meta/info.json", "meta/modality.json"]
        metadata_names += [p.relative_to(source).as_posix() for p in sorted((source / "meta").glob("*.json"))
                           if p.name not in {"info.json", "modality.json"}]
        manifest = {
            "format": "robodojo_causal_visual_memory", "format_version": 2,
            "source": str(source), "source_link": "source_dataset", "no_raw_source_copies": True,
            "representation": REPRESENTATION, "channel_order": CHANNELS, "camera_order": CAMERAS,
            "recent_records": spec.recent_records, "decision_stride_env_steps": spec.decision_stride,
            "use_pose": spec.use_pose, "padding": -1,
            "visual_shape": list(VISUAL_SHAPE), "visual_dtype": "uint8",
            "visual_encoding": "ordered_rgb_2x4_box_8x8_resize256",
            "metadata_sha256": {name: sha256(source / name) for name in metadata_names},
            "source_inventory": _source_inventory(source), "source_files": {}, "video_sources": {}, "episodes": {},
        }
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(files)))) as pool:
            results = pool.map(lambda p: _build_file(source, p, temporary, spec, info), files)
            for relative, meta, episodes in results:
                if manifest["episodes"].keys() & episodes.keys():
                    raise ValueError("Episode spans multiple source shards")
                manifest["source_files"][relative] = meta
                manifest["episodes"].update(episodes)
                for entry in episodes.values():
                    for path, identity in entry["video_sources"].items():
                        prior = manifest["video_sources"].setdefault(path, identity)
                        if prior != identity:
                            raise ValueError(f"Conflicting source video identity: {path}")
                print(f"Indexed {relative}: {len(episodes)} episodes, {meta['rows']} frames", flush=True)
        manifest["total_episodes"] = len(manifest["episodes"])
        manifest["total_frames"] = sum(e["length"] for e in manifest["episodes"].values())
        if (manifest["total_episodes"], manifest["total_frames"]) != (info["total_episodes"], info["total_frames"]):
            raise ValueError("Source counts differ from meta/info.json")
        if manifest["source_inventory"] != _source_inventory(source):
            raise ValueError("Source inventory changed during preprocessing")
        manifest["source_identity_sha256"] = _identity(manifest)
        (temporary / "source_dataset").symlink_to(os.path.relpath(source, output), target_is_directory=True)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


class RoboDojoMemorySidecar:
    def __init__(self, root, source, spec):
        self.root, self.source, self.spec = Path(root).resolve(), Path(source).resolve(), spec
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        contract = (m.get("format"), m.get("format_version"), m.get("representation"), m.get("padding"))
        self._indices_v2 = contract == ("robodojo_causal_memory_indices", 2, REPRESENTATION, -1)
        self._visual_v2 = contract == ("robodojo_causal_visual_memory", 2, REPRESENTATION, -1)
        if not (self._indices_v2 or self._visual_v2):
            raise ValueError(f"Unsupported memory sidecar format: {contract}")
        if (m.get("recent_records"), m.get("decision_stride_env_steps"), m.get("use_pose")) != (
            spec.recent_records, spec.decision_stride, spec.use_pose
        ):
            raise ValueError("Memory sidecar and model contracts differ")
        if m.get("camera_order") != CAMERAS or m.get("channel_order") != CHANNELS:
            raise ValueError("Memory sidecar channel/camera order differs")
        if m.get("source_link") != "source_dataset" or (self.root / "source_dataset").resolve() != self.source:
            raise ValueError("Memory sidecar must link to the configured source dataset")
        if self._indices_v2:
            if not m.get("visual_sidecar"):
                raise ValueError("RoboDojo memory v2 sidecar must include visual history")
            if (m.get("visual_shape"), m.get("visual_dtype"), m.get("visual_encoding")) != (
                list(VISUAL_SHAPE), "uint8", "ordered_rgb_2x4_box_8x8"
            ):
                raise ValueError("Unsupported visual memory shape or encoding")
            _check_indices_v2_layout(self.root, m)
            source_files = set(m["source_files"])
            actual_source_files = {path.relative_to(self.source).as_posix()
                                   for path in (self.source / "data").glob("chunk-*/*.parquet")}
            if source_files != actual_source_files or m.get("source_identity_sha256") != _indices_v2_identity(m):
                raise ValueError("Source files or identity differ from indices v2 manifest")
        else:
            _check_layout(self.root, m)
            if m.get("source_inventory") != _source_inventory(self.source) or m.get("source_identity_sha256") != _identity(m):
                raise ValueError("Source inventory or identity differs from visual memory manifest")
            if (m.get("visual_shape"), m.get("visual_dtype"), m.get("visual_encoding")) != (
                list(VISUAL_SHAPE), "uint8", "ordered_rgb_2x4_box_8x8_resize256"
            ):
                raise ValueError("Unsupported visual memory shape or encoding")
        for name, digest in m["metadata_sha256"].items():
            if sha256(_inside(self.source, name, allow_symlinks=True)) != digest:
                raise ValueError(f"Source metadata changed: {name}")
        if self._visual_v2 and set(m["video_sources"]) != set(m["source_inventory"]["videos"]):
            raise ValueError("Memory manifest does not cover source videos")
        self._verified_sources, self._verified_videos = set(), set()
        self._episode, self._indices = None, None
        self._visual_episode, self._visual_tokens = None, None

    def indices(self, episode, length):
        episode = int(episode)
        if self._episode == episode:
            if self._indices.shape[0] != length:
                raise ValueError("Memory sidecar episode length differs from source")
            return self._indices
        entry = self.manifest["episodes"][str(episode)]
        if entry["length"] != length:
            raise ValueError("Memory sidecar episode length differs from source")
        relative = entry["source_file"]
        if relative not in self._verified_sources:
            path, meta = _inside(self.source, relative, True), self.manifest["source_files"][relative]
            if path.stat().st_size != meta["size_bytes"] or sha256(path) != meta["sha256"]:
                raise ValueError(f"Source parquet changed: {relative}")
            self._verified_sources.add(relative)
        if self._visual_v2:
            for video_path, identity in entry["video_sources"].items():
                if video_path not in self._verified_videos:
                    path = _inside(self.source, video_path, True)
                    if path.stat().st_size != identity["size_bytes"] or sha256(path) != identity["sha256"]:
                        raise ValueError(f"Source video changed: {video_path}")
                    self._verified_videos.add(video_path)
        path = _inside(self.root, entry["file"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Memory indices changed: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.int32 or not np.array_equal(values, history_indices(length, self.spec)):
            raise ValueError("Memory indices violate the strictly-past grid")
        self._episode, self._indices = episode, values
        return values

    def visual_tokens(self, episode, length):
        episode = int(episode)
        if self._visual_episode == episode:
            if self._visual_tokens.shape[0] != length:
                raise ValueError("Memory visual episode length differs from source")
            return self._visual_tokens
        entry = self.manifest["episodes"][str(episode)]
        if entry["length"] != length:
            raise ValueError("Memory visual episode length differs from source")
        # The legacy indices-v2 format stores visual metadata under a nested
        # ``visual`` entry, while the visual-memory-v2 format stores it beside
        # the history index as ``visual_file``/``visual_sha256``.  Both must
        # resolve to the compact visual token array, never the index array.
        if self._indices_v2:
            visual = entry["visual"]
            relative, digest = visual["file"], visual["sha256"]
        else:
            relative, digest = entry["visual_file"], entry["visual_sha256"]
        path = _inside(self.root, relative)
        if sha256(path) != digest:
            raise ValueError(f"Memory visual patches changed: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint8 or tuple(values.shape) != (length, *VISUAL_SHAPE):
            raise ValueError(f"Invalid visual memory array: {values.shape}/{values.dtype}")
        self._visual_episode, self._visual_tokens = episode, values
        return values


def validate_sidecar(root, source, spec=RoboDojoMemorySpec()):
    import pandas as pd
    reader = RoboDojoMemorySidecar(root, source, spec)
    total = 0
    for relative, meta in reader.manifest["source_files"].items():
        rows = pd.read_parquet(_inside(reader.source, relative, True), columns=["episode_index", "frame_index", "timestamp"])
        if len(rows) != meta["rows"]:
            raise ValueError("Source row count changed")
        covered = np.zeros(len(rows), dtype=bool)
        for episode, entry in reader.manifest["episodes"].items():
            if entry["source_file"] != relative:
                continue
            start, stop = entry["source_row_start"], entry["source_row_stop"]
            group = rows.iloc[start:stop]
            if not 0 <= start < stop <= len(rows) or covered[start:stop].any():
                raise ValueError("Invalid or overlapping source row links")
            if not (group["episode_index"] == int(episode)).all() or not np.array_equal(
                group["frame_index"].to_numpy(), np.arange(entry["length"])
            ):
                raise ValueError("Episode/frame links do not match source rows")
            reader.indices(int(episode), entry["length"])
            reader.visual_tokens(int(episode), entry["length"])
            covered[start:stop] = True
            total += entry["length"]
        if not covered.all():
            raise ValueError("Memory manifest does not cover every source row")
    manifest = reader.manifest
    if total != manifest["total_frames"] or len(manifest["episodes"]) != manifest["total_episodes"]:
        raise ValueError("Memory manifest totals differ")
    print(f"Validated {total} frames, {len(manifest['episodes'])} episodes; original data unchanged", flush=True)
    return manifest
