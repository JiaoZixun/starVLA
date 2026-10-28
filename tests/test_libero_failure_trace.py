"""Failure-trace classification, and (when LIBERO + EGL are available) a check that tracing is read-only."""

import json
import os
import pathlib

import numpy as np
import pytest

from examples.simBenchmarks.LIBERO.eval_files.failure_trace import EpisodeTracer, _grasp_spans, classify


def _record(**overrides):
    record = {
        "success": False, "targets": ["a_1", "b_1"], "final_satisfied": [False, False], "max_satisfied": 0,
        "first_grasp_object": None, "grasps": [], "drops": [], "stalled": False,
    }
    record.update(overrides)
    return record


def test_grasp_spans_open_and_closed():
    spans = _grasp_spans([[], ["a"], ["a"], [], ["b"]])
    assert spans == [{"object": "a", "start": 1, "end": 3}, {"object": "b", "start": 4, "end": None}]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"success": True}, "success"),
        ({"first_grasp_object": "c_1", "grasps": [{"object": "c_1", "start": 3, "end": None}]}, "wrong_object_first"),
        ({"max_satisfied": 1, "final_satisfied": [False, False]}, "regressed"),
        ({"max_satisfied": 1, "final_satisfied": [True, False]}, "partial"),
        ({"first_grasp_object": "a_1", "grasps": [{"object": "a_1", "start": 3, "end": 9}],
          "drops": [{"object": "a_1", "step": 9}]}, "dropped_target"),
        ({"first_grasp_object": "a_1", "grasps": [{"object": "a_1", "start": 3, "end": None}]}, "grasped_not_placed"),
        ({}, "no_grasp"),
        ({"stalled": True}, "no_grasp_stalled"),
    ],
)
def test_classify(overrides, expected):
    assert classify(_record(**overrides), n_goals=2) == expected


def _offline_tracer(predicates, grasped):
    tracer = object.__new__(EpisodeTracer)
    tracer.goals = [("on", "a_1", "plate_1")]
    tracer.targets = ["a_1"]
    tracer.task_id, tracer.episode_idx, tracer.task_description = 0, 0, "probe"
    tracer.predicates = [[p] for p in predicates]
    tracer.grasped = grasped
    tracer.eef = [np.zeros(3, dtype=np.float32) for _ in predicates]
    tracer.gripper_cmd = [float("nan")] * len(predicates)
    return tracer


def test_release_before_on_predicate_settles_is_not_a_drop():
    # Released at step 3; "on" becomes true one step later, as in a real LIBERO placement.
    placed = _offline_tracer([False, False, False, False, True], [[], ["a_1"], ["a_1"], [], []]).summary(success=True)
    assert placed["drops"] == []
    dropped = _offline_tracer([False] * 30, [[], ["a_1"], ["a_1"]] + [[]] * 27).summary(success=False)
    assert dropped["drops"] == [{"object": "a_1", "step": 3}] and dropped["category"] == "dropped_target"


def _make_env():
    pytest.importorskip("libero.libero")
    if os.getenv("MUJOCO_GL") != "egl":
        pytest.skip("needs NVIDIA EGL rendering (MUJOCO_GL=egl)")
    from examples.simBenchmarks.LIBERO.eval_files.eval_libero import LIBERO_DUMMY_ACTION, _get_libero_env
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env, _ = _get_libero_env(suite.get_task(0), 256, seed=7)
    return env, suite.get_task_init_states(0)[0], LIBERO_DUMMY_ACTION


def _rollout(env, init_state, wait_action, trace: bool, steps: int = 40):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, *_ = env.step(wait_action)
    rng = np.random.default_rng(0)
    tracer = EpisodeTracer(env, 0, 0, "probe", keep_frames=True) if trace else None
    if tracer is not None:
        tracer.record(obs, None, obs["agentview_image"])
    states = []
    for _ in range(steps):
        action = np.concatenate([rng.uniform(-0.5, 0.5, 6), [rng.choice([-1.0, 1.0])]])
        obs, _, done, _ = env.step(action.tolist())
        if tracer is not None:
            tracer.record(obs, action[-1], obs["agentview_image"])
        states.append(env.env.sim.get_state().flatten().copy())
        states.append(obs["agentview_image"].copy())
    return states, tracer


def test_tracer_does_not_change_rollout(tmp_path):
    # Fresh env per run: two consecutive resets of one LIBERO env already differ at ~1e-14
    # (solver warm start), with or without the tracer.
    results = []
    for trace in (False, True):
        env, init_state, wait_action = _make_env()
        try:
            results.append(_rollout(env, init_state, wait_action, trace=trace))
        finally:
            env.close()
    (baseline, _), (traced, tracer) = results
    for a, b in zip(baseline, traced):
        np.testing.assert_array_equal(a, b)
    record = tracer.write(tmp_path, success=False)
    assert record["goals"] and len(record["final_satisfied"]) == len(record["goals"])
    assert record["category"] != "success"
    assert pathlib.Path(record["keyframes"]).is_file() and record["keyframe_labels"][-1].endswith("final")
    lines = (tmp_path / "failure_traces" / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["category"] == record["category"]
