"""Read-only per-episode tracing of LIBERO rollouts for failure analysis.

Enabled with ``LIBERO_FAILURE_TRACE=1``. After every simulator step the tracer evaluates each
BDDL goal predicate separately and checks which objects the gripper is grasping, then
classifies the failed episode (wrong object, dropped target, partial goal completion, ...).
It only reads simulator state: no env.step/reset, no RNG, no rendering, so rollouts are
identical with the tracer on or off. The goal predicates repeat exactly what
``env._check_success()`` already evaluated on the same state.

Output per worker: ``<video_out_path>/failure_traces/episodes.jsonl`` (one line per episode,
successes included as the contrast set) and, for the first ``LIBERO_TRACE_KEYFRAMES_PER_TASK``
failed episodes of each task, one keyframe strip ``<task>_ep<idx>.png`` (start, grasp/release
and predicate events, final frame). ``summarize_failure_traces.py`` merges them per run.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

MAX_KEYFRAMES = 8
STALL_WINDOW = 60  # policy steps
STALL_DISTANCE_M = 0.02
PATH_STRIDE = 8  # store the eef path every 8 steps (one action chunk)
DROP_GRACE = 20  # steps after a release in which the object's goals may still settle


def failure_trace_enabled() -> bool:
    return os.getenv("LIBERO_FAILURE_TRACE", "0").lower() in {"1", "true", "yes", "on"}


def _predicate_text(state) -> str:
    return " ".join(str(item) for item in state)


class EpisodeTracer:
    """Collect goal-predicate, grasp and end-effector state for one episode."""

    def __init__(self, env, task_id: int, episode_idx: int, task_description: str, keep_frames: bool):
        self._env = env.env  # LIBERO problem env (BDDLBaseDomain) under the ControlEnv wrapper
        self._gripper = self._env.robots[0].gripper
        self.goals = list(self._env.parsed_problem["goal_state"])
        # First argument of a binary goal is the object that has to be moved there.
        self.targets = sorted({str(goal[1]) for goal in self.goals if len(goal) == 3})
        self.objects = list(self._env.objects_dict.keys())
        self.task_id = int(task_id)
        self.episode_idx = int(episode_idx)
        self.task_description = task_description
        self.keep_frames = bool(keep_frames)
        self.predicates: list[list[bool]] = []
        self.grasped: list[list[str]] = []
        self.eef: list[np.ndarray] = []
        self.gripper_cmd: list[float] = []
        self.frames: list[tuple[int, str, np.ndarray]] = []
        self._last_frame = None

    def _keyframe(self, step: int, label: str, image) -> None:
        if self.keep_frames and image is not None and len(self.frames) < MAX_KEYFRAMES - 1:
            self.frames.append((step, label, np.asarray(image).copy()))

    def record(self, obs: dict, gripper_cmd: float | None, image=None) -> None:
        """Call once before the first action and after every policy env.step."""
        step = len(self.predicates)
        predicates = [bool(self._env._eval_predicate(goal)) for goal in self.goals]
        grasped = [name for name, obj in self._env.objects_dict.items() if self._env._check_grasp(self._gripper, obj)]
        if step == 0:
            self._keyframe(step, "start", image)
        else:
            for goal, before, now in zip(self.goals, self.predicates[-1], predicates):
                if before != now:
                    self._keyframe(step, ("+" if now else "-") + _predicate_text(goal), image)
            for name in sorted(set(grasped) ^ set(self.grasped[-1])):
                self._keyframe(step, ("grasp " if name in grasped else "release ") + name, image)
        self.predicates.append(predicates)
        self.grasped.append(grasped)
        self.eef.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy())
        self.gripper_cmd.append(float("nan") if gripper_cmd is None else float(gripper_cmd))
        if self.keep_frames and image is not None:
            self._last_frame = np.asarray(image).copy()

    def summary(self, success: bool) -> dict:
        predicates = np.asarray(self.predicates, dtype=bool).reshape(len(self.predicates), len(self.goals))
        satisfied = predicates.sum(axis=1)
        n_goals = len(self.goals)
        first_satisfied = [int(np.argmax(col)) if col.any() else None for col in predicates.T]
        grasp_spans = _grasp_spans(self.grasped)
        first_grasp = min(grasp_spans, key=lambda span: span["start"]) if grasp_spans else None
        drops = []
        for span in grasp_spans:
            # Released a goal object and its goals did not all hold within DROP_GRACE steps
            # (contact predicates such as "on" settle a few steps after release).
            if span["object"] not in self.targets or span["end"] is None:
                continue
            goal_ids = [i for i, goal in enumerate(self.goals) if len(goal) == 3 and goal[1] == span["object"]]
            window = predicates[span["end"]:span["end"] + DROP_GRACE][:, goal_ids]
            if not window.all(axis=1).any():
                drops.append(span)
        eef = np.asarray(self.eef)
        tail = eef[-STALL_WINDOW:] if len(eef) else eef
        tail_motion = float(np.linalg.norm(tail.max(axis=0) - tail.min(axis=0))) if len(tail) else 0.0
        commands = np.asarray(self.gripper_cmd[1:])
        commands = commands[np.isfinite(commands)]
        progress_steps = np.nonzero(np.diff(satisfied) > 0)[0] + 1 if len(satisfied) > 1 else np.asarray([], dtype=int)
        record = {
            "task_id": self.task_id,
            "episode_idx": self.episode_idx,
            "task_description": self.task_description,
            "success": bool(success),
            "steps": len(self.predicates) - 1,
            "goals": [_predicate_text(goal) for goal in self.goals],
            "targets": self.targets,
            "final_satisfied": predicates[-1].tolist() if len(predicates) else [],
            "max_satisfied": int(satisfied.max()) if len(satisfied) else 0,
            "first_satisfied_step": first_satisfied,
            "last_progress_step": int(progress_steps[-1]) if len(progress_steps) else None,
            "grasps": grasp_spans,
            "first_grasp_object": first_grasp["object"] if first_grasp else None,
            "first_grasp_step": first_grasp["start"] if first_grasp else None,
            "wrong_object_grasps": sorted({span["object"] for span in grasp_spans if span["object"] not in self.targets}),
            "drops": [{"object": span["object"], "step": span["end"]} for span in drops],
            "gripper_toggles": int(np.count_nonzero(np.diff(commands) != 0)) if len(commands) > 1 else 0,
            "tail_motion_m": round(tail_motion, 4),
            "stalled": bool(not success and len(eef) >= STALL_WINDOW and tail_motion < STALL_DISTANCE_M),
            "eef_path": eef[::PATH_STRIDE].round(4).tolist(),
        }
        record["category"] = classify(record, n_goals)
        return record

    def write(self, out_dir: pathlib.Path, success: bool) -> dict:
        record = self.summary(success)
        trace_dir = pathlib.Path(out_dir) / "failure_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        if not success and self.keep_frames and self._last_frame is not None:
            frames = self.frames + [(record["steps"], "final", self._last_frame)]
            name = f"task{self.task_id:02d}_ep{self.episode_idx:02d}.png"
            _write_strip(trace_dir / name, frames)
            record["keyframes"] = str(trace_dir / name)
            record["keyframe_labels"] = [f"{step}:{label}" for step, label, _ in frames]
        with (trace_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def _grasp_spans(grasped: list[list[str]]) -> list[dict]:
    spans, open_spans = [], {}
    for step, names in enumerate(grasped):
        for name in names:
            open_spans.setdefault(name, step)
        for name in [n for n in open_spans if n not in names]:
            spans.append({"object": name, "start": open_spans.pop(name), "end": step})
    spans += [{"object": name, "start": start, "end": None} for name, start in open_spans.items()]
    return sorted(spans, key=lambda span: span["start"])


def classify(record: dict, n_goals: int) -> str:
    """Primary failure mode; the first matching rule wins."""
    if record["success"]:
        return "success"
    final = sum(record["final_satisfied"])
    first = record["first_grasp_object"]
    if first is not None and record["targets"] and first not in record["targets"]:
        return "wrong_object_first"  # grounding error: went for a non-goal object first
    if record["max_satisfied"] > final:
        return "regressed"  # a satisfied goal was undone
    if 0 < final < n_goals:
        return "partial"  # some conjuncts done; see final_satisfied for which subgoal is missing
    if record["drops"]:
        return "dropped_target"
    if record["targets"] and any(span["object"] in record["targets"] for span in record["grasps"]):
        return "grasped_not_placed"
    if not record["grasps"]:
        return "no_grasp_stalled" if record["stalled"] else "no_grasp"
    return "other"


def _write_strip(path: pathlib.Path, frames: list[tuple[int, str, np.ndarray]]) -> None:
    import imageio
    from PIL import Image, ImageDraw

    tiles = []
    for step, label, image in frames:
        tile = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile.width, 14), fill=(0, 0, 0))
        draw.text((2, 1), f"{step} {label}"[:48], fill=(255, 255, 255))
        tiles.append(np.asarray(tile))
    imageio.imwrite(path, np.concatenate(tiles, axis=1))
