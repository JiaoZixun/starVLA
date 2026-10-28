"""Merge per-worker failure traces of a LIBERO run into common failure modes.

Usage: python summarize_failure_traces.py --run-root results/libero_eval/<run_id> [--suite libero_10]
Reads ``<run-root>/<suite>/shard_*/failure_traces/episodes.jsonl`` (written with
LIBERO_FAILURE_TRACE=1) and writes ``<run-root>/<suite>/failure_summary.{json,md}``.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

EXAMPLES_PER_CLUSTER = 3


def load_episodes(suite_dir: pathlib.Path) -> list[dict]:
    episodes = []
    for path in sorted(suite_dir.glob("shard_*/failure_traces/episodes.jsonl")):
        episodes += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(episodes, key=lambda e: (e["task_id"], e["episode_idx"]))


def missing_goals(episode: dict) -> tuple[str, ...]:
    return tuple(goal for goal, done in zip(episode["goals"], episode["final_satisfied"]) if not done)


def summarize(episodes: list[dict]) -> dict:
    failures = [e for e in episodes if not e["success"]]
    # Cluster = (task, failure mode, which goal conjuncts are still missing).
    clusters = collections.defaultdict(list)
    for e in failures:
        clusters[(e["task_id"], e["category"], missing_goals(e))].append(e)
    tasks = {}
    for e in episodes:
        t = tasks.setdefault(e["task_id"], {"task_description": e["task_description"], "episodes": 0, "successes": 0,
                                            "categories": collections.Counter(), "wrong_objects": collections.Counter()})
        t["episodes"] += 1
        t["successes"] += int(e["success"])
        if not e["success"]:
            t["categories"][e["category"]] += 1
            t["wrong_objects"].update(e["wrong_object_grasps"])
    cluster_rows = []
    for (task_id, category, missing), members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        steps = [m["last_progress_step"] for m in members if m["last_progress_step"] is not None]
        cluster_rows.append({
            "task_id": task_id,
            "task_description": members[0]["task_description"],
            "category": category,
            "missing_goals": list(missing),
            "count": len(members),
            "stalled": sum(m["stalled"] for m in members),
            "median_last_progress_step": sorted(steps)[len(steps) // 2] if steps else None,
            "wrong_objects": dict(collections.Counter(o for m in members for o in m["wrong_object_grasps"])),
            "examples": [{"episode_idx": m["episode_idx"], "keyframes": m.get("keyframes")} for m in members[:EXAMPLES_PER_CLUSTER]],
        })
    return {
        "episodes": len(episodes),
        "failures": len(failures),
        "categories": dict(collections.Counter(e["category"] for e in failures).most_common()),
        "tasks": {k: {**v, "categories": dict(v["categories"]), "wrong_objects": dict(v["wrong_objects"])} for k, v in sorted(tasks.items())},
        "clusters": cluster_rows,
    }


def to_markdown(suite: str, summary: dict) -> str:
    lines = [f"# {suite} failure modes", "",
             f"{summary['failures']} failed of {summary['episodes']} traced episodes.", "",
             "| Failure mode | Count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in summary["categories"].items()]
    lines += ["", "## Per task", "", "| Task | Success | Failure modes | Wrong objects grasped |", "|---|---:|---|---|"]
    for task_id, t in summary["tasks"].items():
        modes = ", ".join(f"{k} {v}" for k, v in sorted(t["categories"].items(), key=lambda kv: -kv[1])) or "-"
        wrong = ", ".join(f"{k} {v}" for k, v in t["wrong_objects"].items()) or "-"
        lines.append(f"| {task_id} {t['task_description']} | {t['successes']}/{t['episodes']} | {modes} | {wrong} |")
    lines += ["", "## Largest clusters (task, mode, missing goals)", "",
              "| Task | Mode | Missing goals | Count | Stalled | Median last progress step | Examples |", "|---|---|---|---:|---:|---:|---|"]
    for c in summary["clusters"][:20]:
        examples = ", ".join(f"ep{x['episode_idx']}" + (f" [{pathlib.Path(x['keyframes']).name}]" if x["keyframes"] else "") for x in c["examples"])
        lines.append(f"| {c['task_id']} | {c['category']} | {'; '.join(c['missing_goals']) or '-'} | {c['count']} | {c['stalled']} | "
                     f"{c['median_last_progress_step']} | {examples} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--suite", action="append", help="Suite directory name(s); default: every suite with traces")
    args = parser.parse_args()
    suites = args.suite or sorted(p.name for p in args.run_root.iterdir() if p.is_dir() and any(p.glob("shard_*/failure_traces")))
    for suite in suites:
        episodes = load_episodes(args.run_root / suite)
        if not episodes:
            print(f"{suite}: no failure traces")
            continue
        summary = summarize(episodes)
        (args.run_root / suite / "failure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (args.run_root / suite / "failure_summary.md").write_text(to_markdown(suite, summary), encoding="utf-8")
        print(f"{suite}: {summary['failures']}/{summary['episodes']} failed; modes {summary['categories']}")


if __name__ == "__main__":
    main()
