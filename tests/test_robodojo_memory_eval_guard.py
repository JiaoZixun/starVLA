from pathlib import Path
import sys

import pytest


EVAL_FILES = Path(__file__).resolve().parents[1] / "examples/simBenchmarks/RoboDojo/eval_files"
sys.path.insert(0, str(EVAL_FILES))
from robodojo_memory_eval_guard import (  # noqa: E402
    checkpoint_requires_execution_memory,
    ensure_legacy_robodojo_eval_compatible,
)


def _checkpoint(tmp_path, config_name, memory_required):
    run_dir = tmp_path / "run"
    final_model = run_dir / "final_model"
    final_model.mkdir(parents=True)
    checkpoint = final_model / "pytorch_model.pt"
    checkpoint.touch()
    (run_dir / config_name).write_text(
        "framework:\n  memory:\n    required: "
        + ("true" if memory_required else "false")
        + "\n"
    )
    return checkpoint


@pytest.mark.parametrize("config_name", ["config.full.yaml", "config.yaml"])
def test_memory_checkpoint_is_rejected_by_legacy_eval_client(tmp_path, config_name):
    checkpoint = _checkpoint(tmp_path, config_name, True)

    assert checkpoint_requires_execution_memory(checkpoint)
    with pytest.raises(RuntimeError, match="memory-aware simulator client"):
        ensure_legacy_robodojo_eval_compatible(checkpoint)


def test_non_memory_checkpoint_remains_eligible_for_legacy_eval(tmp_path):
    checkpoint = _checkpoint(tmp_path, "config.full.yaml", False)

    assert not checkpoint_requires_execution_memory(checkpoint)
    ensure_legacy_robodojo_eval_compatible(checkpoint)
