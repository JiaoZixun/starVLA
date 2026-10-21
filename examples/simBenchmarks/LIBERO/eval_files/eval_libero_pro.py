"""Zero-shot LIBERO-PRO evaluator (Object / Position / Semantic / Task).

The episode loop is the standard LIBERO one from ``eval_libero.py``: same
seeding, init states, wait steps, observation preprocessing, websocket policy
protocol, and memory bookkeeping.  Only the suite (pre-generated LIBERO-PRO
BDDL + init files) and, for Semantic/Task, the instruction source change.
"""

import dataclasses
import json
import logging
import os
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.bddl_utils as BDDLUtils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.eval_libero import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _binarize_gripper_open,
    _quat2axisangle,
)
from examples.simBenchmarks.LIBERO.eval_files.libero_pro_utils import (
    BASE_SUITE_MAX_STEPS,
    PERTURBATIONS,
    pro_suite_name,
    resolve_instruction,
)
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient
from starVLA.model.framework.VLM4A.spatial_geometry import pose_from_position_axis_angle


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    task_suite_name: str = "libero_spatial"  # Base suite: libero_spatial, libero_object, libero_goal, libero_10
    perturbation: str = "object"  # object (Object), swap (Position), language (Semantic), task (Task)
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = -1  # If > 0, only the first N tasks (smoke test).
    task_id_start: int = 0
    task_id_end: int = -1
    max_steps: int = -1  # If > 0, cap the episode horizon (smoke test only; not a benchmark result).

    video_out_path: str = "results/libero_pro"
    seed: int = 7
    unnorm_key: str | None = None
    save_videos: bool = False
    max_videos_per_outcome: int = 1


def _bddl_path(problem_folder: str, bddl_file: str) -> pathlib.Path:
    return pathlib.Path(get_libero_path("bddl_files")) / problem_folder / bddl_file


def eval_libero_pro(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    suite_name = pro_suite_name(args.task_suite_name, args.perturbation)
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    base_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    base_tasks = {task.name: task for task in base_suite.tasks}
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = BASE_SUITE_MAX_STEPS[args.task_suite_name]
    if args.max_steps > 0:
        logging.warning("max_steps=%d overrides the benchmark horizon %d (smoke test only).", args.max_steps, max_steps)
        max_steps = args.max_steps
    if args.max_videos_per_outcome < 1:
        raise ValueError("max_videos_per_outcome must be >= 1")
    logging.info(
        "LIBERO-PRO %s (%s): suite %s, horizon %d",
        PERTURBATIONS[args.perturbation]["dimension"], args.perturbation, suite_name, max_steps,
    )

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    client_model = ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)

    n_available_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    task_start = max(0, args.task_id_start)
    task_end = n_available_tasks if args.task_id_end < 0 else min(args.task_id_end, n_available_tasks)
    if task_start >= task_end:
        raise ValueError(f"Empty task range [{task_start}, {task_end}) for {suite_name}")

    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(range(task_start, task_end)):
        task = task_suite.get_task(task_id)
        if task.name not in base_tasks:
            raise ValueError(f"{suite_name} task {task.name!r} has no counterpart in {args.task_suite_name}")
        base_task = base_tasks[task.name]
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < args.num_trials_per_task:
            raise ValueError(
                f"{suite_name} task {task_id} has {len(initial_states)} init states, "
                f"fewer than num_trials_per_task={args.num_trials_per_task}"
            )

        bddl_file = _bddl_path(task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)  # Same per-task seeding as eval_libero.py.
        base_bddl_language = BDDLUtils.get_problem_info(
            _bddl_path(base_task.problem_folder, base_task.bddl_file)
        )["language_instruction"]
        task_description = resolve_instruction(
            perturbation=args.perturbation,
            registry_language=task.language,
            pro_bddl_language=env.language_instruction,
            base_bddl_language=base_bddl_language,
        )
        logging.info("Task %d: %s | instruction: %s", task_id, task.name, task_description)

        task_episodes, task_successes = 0, 0
        saved_task_videos = {"success": 0, "failure": 0}
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            step = 0
            done = False
            replay_images = (
                []
                if args.save_videos and any(c < args.max_videos_per_outcome for c in saved_task_videos.values())
                else None
            )
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Rotate 180 degrees to match train preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if replay_images is not None:
                    replay_images.append(img)
                state = np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                )
                example_dict = {
                    "image": [img, wrist_img],
                    "lang": str(task_description),
                    # Memory-only proprioception; never sent as ``state`` (see eval_libero.py).
                    "memory_state": state.astype(np.float32),
                    "geometry_current": {
                        "eef_pose": pose_from_position_axis_angle(state[:3], state[3:6]),
                        "arm_mask": np.asarray([True], dtype=np.bool_),
                    },
                }
                raw_action = client_model.step(example=example_dict, step=step)["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={open_gripper.shape}"
                    )
                delta_action = np.concatenate(
                    [world_vector_delta, rotation_delta, _binarize_gripper_open(open_gripper)], axis=0
                )

                obs, reward, done, info = env.step(delta_action.tolist())
                # Commit the accepted command; the next 8-step query closes the memory event.
                client_model.record_executed_action(delta_action, step)
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            suffix = "success" if done else "failure"
            if replay_images is not None and saved_task_videos[suffix] < args.max_videos_per_outcome:
                saved_task_videos[suffix] += 1
                video_name = f"task{task_id:02d}_episode{episode_idx:02d}_{suffix}.mp4"
                imageio.mimwrite(out_dir / video_name, [np.asarray(x) for x in replay_images], fps=10)
            logging.info(
                "Success: %s | episodes %d | successes %d (%.1f%%)",
                done, total_episodes, total_successes, 100.0 * total_successes / total_episodes,
            )

        env.close()
        task_results.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task_description,
                "bddl_file": str(bddl_file),
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_successes / task_episodes,
            }
        )
        logging.info("Task %d success rate: %.4f", task_id, task_successes / task_episodes)

    summary = {
        "benchmark": "LIBERO-PRO",
        "task_suite": args.task_suite_name,
        "perturbation": args.perturbation,
        "dimension": PERTURBATIONS[args.perturbation]["dimension"],
        "pro_suite": suite_name,
        "instruction_source": PERTURBATIONS[args.perturbation]["instruction"],
        "seed": args.seed,
        "num_trials_per_task": args.num_trials_per_task,
        "max_steps": max_steps,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "task_results": task_results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logging.info("Total success rate: %.4f (%d/%d)", summary["success_rate"], total_successes, total_episodes)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_libero_pro)
