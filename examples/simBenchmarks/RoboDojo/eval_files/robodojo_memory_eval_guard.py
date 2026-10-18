"""Detect memory checkpoints before they reach RoboDojo's legacy eval client."""

from pathlib import Path

import yaml


def checkpoint_requires_execution_memory(checkpoint: str | Path) -> bool:
    """Read the saved training config and identify causal-memory checkpoints."""
    run_dir = Path(checkpoint).resolve().parent.parent
    for name in ("config.full.yaml", "config.yaml"):
        config_path = run_dir / name
        if not config_path.is_file():
            continue
        with config_path.open() as stream:
            config = yaml.safe_load(stream) or {}
        framework = config.get("framework") or {}
        memory = framework.get("memory") or {}
        if isinstance(memory, dict) and memory.get("required") is True:
            return True
    return False


def ensure_legacy_robodojo_eval_compatible(checkpoint: str | Path) -> None:
    """Fail closed while the delegated simulator client lacks memory feedback."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint_requires_execution_memory(checkpoint):
        return
    raise RuntimeError(
        "Refusing legacy RoboDojo evaluation for this memory checkpoint. Its saved config "
        "sets framework.memory.required=true, but the delegated XPolicyLab eval client does "
        "not send env_id, episode_id, env_step, raw_state, or per-step executed-action and "
        "three-camera feedback. A memory-aware simulator client must report every completed "
        "environment step before this checkpoint can be evaluated consistently with training."
    )
