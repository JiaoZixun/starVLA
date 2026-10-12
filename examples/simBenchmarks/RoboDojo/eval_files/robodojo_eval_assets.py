"""Inspect official RoboDojo layout assets before starting an evaluation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

OBJECT_TYPES = {"Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"}

# These files are consumed outside the layout JSON itself.  In particular,
# the three competition tasks load support-arm trajectories at reset time;
# checking only the object references in the layout lets an evaluation run for
# hours before failing when it reaches one of those tasks.
_COMMON_STATIC_PREFIXES = (
    "Assets/Background/",
    "Assets/Room/Simple_Room/",
    "Assets/Room/Simple_Room_nolight/",
    "Assets/Material/material_0122/",
    "Assets/Material/material_0564/",
    "Assets/Sensor/Camera/",
    "Assets/Robots/x5/",
)
_FRANKA_STATIC_PREFIX = "Assets/Robots/franka/"
_TRAJECTORY_PREFIXES = {
    "imitate_sorting_sequence": "Assets/Traj/RoboDojo/imitate_sorting_sequence/0/",
    "make_kong": "Assets/Traj/RoboDojo/make_kong/",
    "play_tic_tac_toe": "Assets/Traj/RoboDojo/play_tic_tac_toe/",
}


def format_bytes(size: int) -> str:
    if size >= 1024 ** 2:
        return f"{size / (1024 ** 2):.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def _layout_files(layout_dir: Path, task: str, episodes: int) -> list[Path]:
    pattern = re.compile(rf"{re.escape(task)}_(\d+)\.json")
    files = []
    for path in layout_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if len(files) < episodes:
        raise ValueError(f"{task}: need {episodes} layouts, found {len(files)} in {layout_dir}")
    return [path for _, path in files[:episodes]]


def _tracked_asset_files(asset_repo: Path) -> set[str]:
    if not (asset_repo / ".git").exists():
        return set()
    result = subprocess.run(
        ["git", "-C", str(asset_repo), "ls-files", "Assets"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode:
        return set()
    return set(result.stdout.splitlines())


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _lfs_size(asset_repo: Path, relative_path: str) -> int:
    try:
        result = subprocess.run(
            ["git", "-C", str(asset_repo), "show", f"HEAD:{relative_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    match = re.search(rb"(?m)^size (\d+)$", result.stdout[:512])
    return int(match.group(1)) if match else 0


def _add_tracked_prefix(
    assets_root: Path,
    tracked: set[str],
    prefix: str,
    needed: set[str],
    issues: list[str],
) -> None:
    """Add absent files under a tracked asset prefix to the download set."""
    candidates = sorted(path for path in tracked if path.startswith(prefix))
    if not candidates:
        issues.append(f"asset repository has no tracked files under {prefix}")
        return
    for relative_path in candidates:
        local_path = assets_root / relative_path.removeprefix("Assets/")
        if not local_path.is_file() or _is_lfs_pointer(local_path):
            needed.add(relative_path)


def inspect_eval_assets(bench: Path, tasks: list[str], episodes: int, seed: int) -> dict:
    """Return required absent/corrupt LFS files and their declared total size."""
    bench = Path(bench).resolve()
    assets_root = bench / "Assets"
    asset_repo = bench / ".cache/robodojo_assets_repo"
    tracked = _tracked_asset_files(asset_repo)
    layout_dir = assets_root / "Eval_Layout/RoboDojo/arx_x5" / str(seed)
    needed: set[str] = set()
    issues: list[str] = []

    # Validate the scene, camera, X5 robot and materials used by arx_x5.  The
    # competition tasks add Franka and their support-arm trajectory bundles.
    for prefix in _COMMON_STATIC_PREFIXES:
        _add_tracked_prefix(assets_root, tracked, prefix, needed, issues)
    if any(task in _TRAJECTORY_PREFIXES for task in tasks):
        _add_tracked_prefix(assets_root, tracked, _FRANKA_STATIC_PREFIX, needed, issues)
    for task, prefix in _TRAJECTORY_PREFIXES.items():
        if task in tasks:
            _add_tracked_prefix(assets_root, tracked, prefix, needed, issues)

    for task in tasks:
        for layout_path in _layout_files(layout_dir, task, episodes):
            try:
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{layout_path}: cannot parse layout ({exc})")
                continue
            if not isinstance(layout, dict):
                issues.append(f"{layout_path}: layout root is not an object")
                continue

            for object_type, categories in layout.items():
                if object_type not in OBJECT_TYPES or not isinstance(categories, dict):
                    continue
                for category, instances in categories.items():
                    if not isinstance(instances, list):
                        continue
                    for instance in instances:
                        if not isinstance(instance, dict):
                            continue
                        category_idx = instance.get("category_idx")
                        if not isinstance(category_idx, int):
                            issues.append(
                                f"{layout_path.name}: {object_type}/{category} has invalid category_idx={category_idx!r}"
                            )
                            continue
                        model_type = "Clutter" if instance.get("type") == "cluttered" else object_type
                        model_rel = f"Assets/Object/RoboDojo/{model_type}/{category}/{category_idx:05d}"
                        model_dir = assets_root / model_rel.removeprefix("Assets/")

                        metadata = model_dir / "metadata.json"
                        metadata_rel = f"{model_rel}/metadata.json"
                        valid_metadata = False
                        if metadata.is_file() and not _is_lfs_pointer(metadata):
                            try:
                                valid_metadata = bool(json.loads(metadata.read_text(encoding="utf-8")).get("geometry"))
                            except (OSError, json.JSONDecodeError, AttributeError):
                                valid_metadata = False
                        if not valid_metadata:
                            if metadata_rel in tracked:
                                needed.add(metadata_rel)
                            else:
                                issues.append(f"{layout_path.name}: missing/untracked or invalid {metadata_rel}")

                        description = model_dir / "description.json"
                        description_rel = f"{model_rel}/description.json"
                        if description_rel in tracked:
                            valid_description = False
                            if description.is_file() and not _is_lfs_pointer(description):
                                try:
                                    json.loads(description.read_text(encoding="utf-8"))
                                    valid_description = True
                                except (OSError, json.JSONDecodeError):
                                    valid_description = False
                            if not valid_description:
                                needed.add(description_rel)

                        usdz_path = model_dir / "object.usdz"
                        usd_path = model_dir / "object.usd"
                        usdz_exists = usdz_path.is_file()
                        preferred_valid = (
                            usdz_exists and not _is_lfs_pointer(usdz_path)
                        ) or (
                            not usdz_exists and usd_path.is_file() and not _is_lfs_pointer(usd_path)
                        )
                        if not preferred_valid:
                            tracked_objects = [
                                f"{model_rel}/object.usdz",
                                f"{model_rel}/object.usd",
                            ]
                            if usdz_exists:
                                available = [tracked_objects[0]] if tracked_objects[0] in tracked else []
                            else:
                                available = [path for path in tracked_objects if path in tracked]
                            if available:
                                needed.add(available[0])
                            else:
                                issues.append(f"{layout_path.name}: missing/untracked object file under {model_rel}")

    missing = sorted(needed)
    total_bytes = sum(_lfs_size(asset_repo, path) for path in missing)
    return {"missing": missing, "issues": sorted(set(issues)), "estimated_bytes": total_bytes}

