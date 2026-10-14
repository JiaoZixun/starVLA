#!/usr/bin/env python3
"""Aggregate LIBERO suite SR and request-latency summaries into JSON/Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BASELINE = {
    "libero_spatial": {"successes": 461, "episodes": 500, "mean_request_latency_ms": 176.53},
    "libero_object": {"successes": 489, "episodes": 500, "mean_request_latency_ms": 180.96},
    "libero_goal": {"successes": 471, "episodes": 500, "mean_request_latency_ms": 184.31},
    "libero_10": {"successes": 381, "episodes": 500, "mean_request_latency_ms": 143.42},
    "overall": {"successes": 1802, "episodes": 2000, "mean_request_latency_ms": 163.84},
}


def compare_to_baseline(current: dict, baseline: dict) -> dict:
    baseline_success_rate = baseline["successes"] / baseline["episodes"]
    current_latency = float(current["mean_request_latency_ms"])
    baseline_latency = float(baseline["mean_request_latency_ms"])
    current_chunks_per_second = 1000.0 / current_latency
    baseline_chunks_per_second = 1000.0 / baseline_latency
    return {
        "baseline_successes": baseline["successes"],
        "baseline_episodes": baseline["episodes"],
        "baseline_success_rate": baseline_success_rate,
        "success_rate_delta_percentage_points": 100.0 * (current["success_rate"] - baseline_success_rate),
        "baseline_mean_request_latency_ms": baseline_latency,
        "request_latency_delta_ms": current_latency - baseline_latency,
        "request_latency_change_percent": 100.0 * (current_latency / baseline_latency - 1.0),
        "baseline_mean_action_chunks_per_second": baseline_chunks_per_second,
        "action_chunks_per_second_change_percent": 100.0 * (
            current_chunks_per_second / baseline_chunks_per_second - 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    args = parser.parse_args()
    run_root = args.run_root.resolve()

    suites: dict[str, dict] = {}
    global_successes = global_episodes = global_requests = 0
    global_latency_sum = 0.0
    for suite in SUITES:
        suite_dir = run_root / suite
        summary_path = suite_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        seed = int(summary.get("seed", 7))
        task_results = summary.get("task_results", [])
        expected_episodes = args.tasks_per_suite * args.trials_per_task
        if len(task_results) != args.tasks_per_suite or summary.get("episodes") != expected_episodes:
            raise RuntimeError(
                f"Incomplete {suite}: tasks={len(task_results)}/{args.tasks_per_suite}, "
                f"episodes={summary.get('episodes')}/{expected_episodes}"
            )

        timing_by_task: dict[int, dict] = {}
        for shard_dir in sorted(suite_dir.glob("shard_*")):
            timing_path = shard_dir / "inference_timing.json"
            if not timing_path.is_file():
                raise FileNotFoundError(f"Missing request timing summary: {timing_path}")
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            for item in timing.get("tasks", []):
                timing_by_task[int(item["task_id"])] = item

        if len(timing_by_task) != args.tasks_per_suite:
            raise RuntimeError(f"Incomplete per-task request timing for {suite}: {len(timing_by_task)} tasks")
        requests = sum(int(item["requests"]) for item in timing_by_task.values())
        latency_sum = sum(float(item["mean_ms"]) * int(item["requests"]) for item in timing_by_task.values())
        if requests <= 0 or latency_sum <= 0:
            raise RuntimeError(f"No valid inference timing requests for {suite}")
        mean_latency_ms = latency_sum / requests
        successes = int(summary["successes"])
        episodes = int(summary["episodes"])
        suites[suite] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "task_results": task_results,
            "inference_requests": requests,
            "mean_request_latency_ms": mean_latency_ms,
            "mean_action_chunks_per_second": 1000.0 / mean_latency_ms,
            "task_timing": {str(task_id): item for task_id, item in sorted(timing_by_task.items())},
        }
        global_successes += successes
        global_episodes += episodes
        global_requests += requests
        global_latency_sum += latency_sum

    overall_latency_ms = global_latency_sum / global_requests
    overall = {
        "successes": global_successes,
        "episodes": global_episodes,
        "success_rate": global_successes / global_episodes,
        "inference_requests": global_requests,
        "mean_request_latency_ms": overall_latency_ms,
        "mean_action_chunks_per_second": 1000.0 / overall_latency_ms,
    }
    baseline_comparison = {
        suite: compare_to_baseline(suites[suite], BASELINE[suite]) for suite in SUITES
    }
    baseline_comparison["overall"] = compare_to_baseline(overall, BASELINE["overall"])
    video_manifest_path = run_root / "video_manifest.json"
    video_manifest = json.loads(video_manifest_path.read_text(encoding="utf-8")) if video_manifest_path.is_file() else None
    complete_videos = bool(video_manifest and video_manifest.get("complete"))
    video_collection_status = "not_collected" if video_manifest is None else (
        "complete" if complete_videos else "incomplete"
    )
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": seed,
        "trials_per_task": args.trials_per_task,
        "suite_count": len(suites),
        "overall": overall,
        "suites": suites,
        "baseline_comparison": baseline_comparison,
        "video_coverage_complete": complete_videos,
        "video_collection_status": video_collection_status,
        "video_manifest": str(video_manifest_path) if video_manifest_path.is_file() else None,
        "timing_boundary": (
            "Client-observed WebSocket action-chunk request latency; includes client serialization, batching wait, "
            "network round trip and inference, and excludes simulator stepping and observation preprocessing."
        ),
    }
    metrics_path = run_root / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen3.5-0.8B Spatial OFT：LIBERO 评估结果",
        "",
        f"- Checkpoint：`{args.checkpoint.resolve()}`",
        f"- Seed：`{metrics['seed']}`；每任务 `{args.trials_per_task}` episodes；共 4 suites、40 tasks、{global_episodes} episodes。",
        "- 推理速度为 client-observed 单次 8-action chunk 请求往返延迟；包含序列化、server batching 等待、通信和策略推理，不包含仿真 step 与观测预处理。chunks/s 是该延迟的倒数。",
        "",
        "| Suite | SR | 成功/总数 | 平均请求延迟 (ms/chunk) | 平均速度 (chunks/s) | 请求数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for suite in SUITES:
        item = suites[suite]
        lines.append(
            f"| {suite} | {100 * item['success_rate']:.1f}% | {item['successes']}/{item['episodes']} "
            f"| {item['mean_request_latency_ms']:.2f} | {item['mean_action_chunks_per_second']:.2f} "
            f"| {item['inference_requests']} |"
        )
    lines.extend([
        f"| **Overall** | **{100 * metrics['overall']['success_rate']:.1f}%** "
        f"| **{global_successes}/{global_episodes}** | **{overall_latency_ms:.2f}** "
        f"| **{metrics['overall']['mean_action_chunks_per_second']:.2f}** | **{global_requests}** |",
        "",
        "## 与 Qwen3.5-0.8B OFT 50K 基线对比",
        "",
        "| Suite | 本次 SR | 基线 SR | Δ SR (pp) | 本次延迟 (ms/chunk) | 基线延迟 | Δ 延迟 (ms) | 本次速度 (chunks/s) | 基线速度 | 速度变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for suite in SUITES:
        item = suites[suite]
        comparison = baseline_comparison[suite]
        lines.append(
            f"| {suite} | {100 * item['success_rate']:.1f}% | "
            f"{100 * comparison['baseline_success_rate']:.1f}% | "
            f"{comparison['success_rate_delta_percentage_points']:+.1f} | "
            f"{item['mean_request_latency_ms']:.2f} | "
            f"{comparison['baseline_mean_request_latency_ms']:.2f} | "
            f"{comparison['request_latency_delta_ms']:+.2f} | "
            f"{item['mean_action_chunks_per_second']:.2f} | "
            f"{comparison['baseline_mean_action_chunks_per_second']:.2f} | "
            f"{comparison['action_chunks_per_second_change_percent']:+.1f}% |"
        )
    overall_comparison = baseline_comparison["overall"]
    lines.extend([
        f"| **Overall** | **{100 * overall['success_rate']:.1f}%** | "
        f"**{100 * overall_comparison['baseline_success_rate']:.1f}%** | "
        f"**{overall_comparison['success_rate_delta_percentage_points']:+.1f}** | "
        f"**{overall['mean_request_latency_ms']:.2f}** | "
        f"**{overall_comparison['baseline_mean_request_latency_ms']:.2f}** | "
        f"**{overall_comparison['request_latency_delta_ms']:+.2f}** | "
        f"**{overall['mean_action_chunks_per_second']:.2f}** | "
        f"**{overall_comparison['baseline_mean_action_chunks_per_second']:.2f}** | "
        f"**{overall_comparison['action_chunks_per_second_change_percent']:+.1f}%** |",
        "",
        "基线成功率来自四套件各 500 episodes；基线时延为每套件 Task ID 0–4、每任务 10 episodes 的抽样计时。本次时延按全量评估的全部请求加权平均。两者采用相同的 client 请求计时边界、8-worker 并行、batch 4 和 12 ms batching window，但计时样本构成不同。",
        "",
        f"视频收集：{'四套件 40 个任务均有成功与失败视频。' if complete_videos else ('本次评估关闭视频保存。' if video_manifest is None else '存在未收集到两类视频的任务；详见 `video_manifest.json`。')}",
        "",
        "逐任务 SR 和请求时延保存在各 suite 的 `summary.json` 与 shard `inference_timing.json`；全部聚合值见 `evaluation_metrics.json`。",
        "",
    ])
    report_path = run_root / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote evaluation metrics: {metrics_path}")
    print(f"Wrote evaluation report: {report_path}")


if __name__ == "__main__":
    main()
