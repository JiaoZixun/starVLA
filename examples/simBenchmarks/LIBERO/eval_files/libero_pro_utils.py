"""LIBERO-PRO suite selection and checkpoint-loading helpers.

Only the four file-based LIBERO-PRO generalization dimensions are supported.
Each one is a pre-generated suite shipped with LIBERO-PRO (registered BDDL +
init files), so evaluation only changes which suite is loaded and which
instruction the policy receives; the episode loop stays the standard LIBERO one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

# Same horizons as the standard LIBERO evaluator and the LIBERO-PRO README.
BASE_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

# Paper dimension -> (CLI name, registered suite suffix, instruction source).
# "registry": the filename-derived ``task.language`` used by standard LIBERO eval.
# "bddl": the perturbed ``(:language ...)`` written by LIBERO-PRO into the BDDL.
PERTURBATIONS = {
    "object": {"dimension": "Object", "suffix": "object", "instruction": "registry"},
    "swap": {"dimension": "Position", "suffix": "swap", "instruction": "registry"},
    "language": {"dimension": "Semantic", "suffix": "lan", "instruction": "bddl"},
    "task": {"dimension": "Task", "suffix": "task", "instruction": "bddl"},
}


def pro_suite_name(base_suite: str, perturbation: str) -> str:
    if base_suite not in BASE_SUITE_MAX_STEPS:
        raise ValueError(f"Unsupported base suite {base_suite!r}; choose one of {sorted(BASE_SUITE_MAX_STEPS)}.")
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unsupported perturbation {perturbation!r}; choose one of {sorted(PERTURBATIONS)}.")
    return f"{base_suite}_{PERTURBATIONS[perturbation]['suffix']}"


def resolve_instruction(
    *,
    perturbation: str,
    registry_language: str,
    pro_bddl_language: str,
    base_bddl_language: str,
) -> str:
    """Pick the instruction the policy sees for one LIBERO-PRO task.

    Semantic/Task perturb only the BDDL ``(:language ...)`` field, so the
    filename-derived registry text would silently drop the perturbation.
    Object/Position keep the base instruction; that is verified rather than
    assumed, so a perturbed instruction can never be ignored.
    """
    source = PERTURBATIONS[perturbation]["instruction"]
    pro = " ".join(str(pro_bddl_language).split())
    base = " ".join(str(base_bddl_language).split())
    if source == "bddl":
        if not pro:
            raise ValueError("LIBERO-PRO BDDL has an empty (:language ...) field.")
        return pro
    if pro != base:
        raise ValueError(
            f"{perturbation!r} is expected to keep the base instruction, but the BDDL changed it: "
            f"{base!r} -> {pro!r}"
        )
    return registry_language


def spatial_adapter_version(state_keys: set[str]) -> str:
    """Infer the QwenSpatialOFT adapter layout from checkpoint weights.

    v2 adds ``spatial_adapter.spatial_gate`` and ``waypoint_model.gripper_head``;
    v1 checkpoints (e.g. qwen35_0p8b_spatial_oft_libero_50k) have neither.
    """
    has_gate = any(key.startswith("spatial_adapter.spatial_gate.") for key in state_keys)
    has_gripper = any(key.startswith("waypoint_model.gripper_head.") for key in state_keys)
    if has_gate != has_gripper:
        raise ValueError("Checkpoint has a partial v2 spatial adapter (gate/gripper head mismatch).")
    if not any(key.startswith("spatial_adapter.") for key in state_keys):
        raise ValueError("Checkpoint has no spatial_adapter weights.")
    return "v2" if has_gate else "v1"


SPATIAL_FRAMEWORKS = {"QwenSpatialOFT", "QwenSpatialMemOFT"}


def checkpoint_config_overrides(config: dict[str, Any], load_state_keys: Callable[[], set[str]]) -> list[str]:
    """Return the server ``--config_override`` items needed to rebuild a checkpoint.

    Only Spatial frameworks need one: their ``adapter_version`` default changed
    to v2 after older checkpoints were saved without the field.  The version is
    taken from the weights; an explicit config value must agree with them.
    """
    framework = config.get("framework", {}) or {}
    name = framework.get("name")
    if name not in SPATIAL_FRAMEWORKS:
        return []
    detected = spatial_adapter_version(load_state_keys())
    declared = (framework.get("spatial", {}) or {}).get("adapter_version")
    if declared is not None:
        if str(declared) != detected:
            raise ValueError(f"config adapter_version={declared} but checkpoint weights are {detected}.")
        return []
    return [f"framework.spatial.adapter_version={detected}"]


def checkpoint_run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.parent.name != "checkpoints":
        raise ValueError(f"Expected <run>/checkpoints/<file>.pt, got {path}")
    return path.parents[1]


# Ablation label -> framework the checkpoint must have been trained with.
MODEL_FRAMEWORKS = {
    "baseline": "QwenOFT",
    "spatial": "QwenSpatialOFT",
    "mem": "QwenSpatialMemOFT",
}


def _main() -> None:
    """Print one ``--config_override`` value per line for a checkpoint.

    Usage: python libero_pro_utils.py <model: baseline|spatial|mem> <ckpt.pt>
    Fails if the checkpoint framework does not match the requested ablation.
    """
    import sys

    import torch
    import yaml

    model, checkpoint = sys.argv[1], sys.argv[2]
    if model not in MODEL_FRAMEWORKS:
        raise SystemExit(f"Unknown model {model!r}; choose one of {sorted(MODEL_FRAMEWORKS)}.")
    config_path = checkpoint_run_dir(checkpoint) / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = (config.get("framework", {}) or {}).get("name")
    if name != MODEL_FRAMEWORKS[model]:
        raise SystemExit(f"{model} expects framework {MODEL_FRAMEWORKS[model]}, but {config_path} has {name}.")

    def load_state_keys() -> set[str]:
        return set(torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True).keys())

    for item in checkpoint_config_overrides(config, load_state_keys):
        print(item)


if __name__ == "__main__":
    _main()
