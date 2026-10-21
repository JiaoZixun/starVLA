import re
from pathlib import Path

import pytest

from examples.simBenchmarks.LIBERO.eval_files.libero_pro_utils import (
    PERTURBATIONS,
    checkpoint_config_overrides,
    pro_suite_name,
    resolve_instruction,
    spatial_adapter_version,
)

REPO = Path(__file__).resolve().parents[1]
PRO_BDDL = REPO / "playground/LIBERO-PRO/libero/libero/bddl_files"
V1_KEYS = {"spatial_adapter.spatial_projection.0.weight", "waypoint_model.anchor_projection.0.weight"}
V2_KEYS = V1_KEYS | {"spatial_adapter.spatial_gate.0.weight", "waypoint_model.gripper_head.weight"}


def test_only_the_four_paper_dimensions_are_supported():
    assert {v["dimension"] for v in PERTURBATIONS.values()} == {"Object", "Position", "Semantic", "Task"}
    assert pro_suite_name("libero_goal", "object") == "libero_goal_object"
    assert pro_suite_name("libero_10", "swap") == "libero_10_swap"
    assert pro_suite_name("libero_spatial", "language") == "libero_spatial_lan"
    assert pro_suite_name("libero_object", "task") == "libero_object_task"
    for bad in ("environment", "standard", "lan"):
        with pytest.raises(ValueError):
            pro_suite_name("libero_goal", bad)
    with pytest.raises(ValueError):
        pro_suite_name("libero_90", "object")


def test_semantic_and_task_use_the_perturbed_bddl_instruction():
    for pert in ("language", "task"):
        assert resolve_instruction(
            perturbation=pert,
            registry_language="open the middle drawer of the cabinet",
            pro_bddl_language="open the  bottom drawer of the cabinet",
            base_bddl_language="open the middle layer of the drawer",
        ) == "open the bottom drawer of the cabinet"


def test_object_and_position_keep_registry_instruction_and_reject_changed_language():
    for pert in ("object", "swap"):
        assert resolve_instruction(
            perturbation=pert,
            registry_language="put the bowl on the plate",
            pro_bddl_language="put the bowl on the plate",
            base_bddl_language="put the bowl on the plate",
        ) == "put the bowl on the plate"
        with pytest.raises(ValueError):
            resolve_instruction(
                perturbation=pert,
                registry_language="put the bowl on the plate",
                pro_bddl_language="put the bowl on the stove",
                base_bddl_language="put the bowl on the plate",
            )


def test_spatial_adapter_version_is_read_from_weights():
    assert spatial_adapter_version(V1_KEYS) == "v1"
    assert spatial_adapter_version(V2_KEYS) == "v2"
    with pytest.raises(ValueError):
        spatial_adapter_version(V1_KEYS | {"spatial_adapter.spatial_gate.0.weight"})


def test_overrides_only_fill_a_missing_spatial_adapter_version():
    assert checkpoint_config_overrides({"framework": {"name": "QwenOFT"}}, lambda: pytest.fail("unused")) == []
    legacy = {"framework": {"name": "QwenSpatialOFT", "spatial": {"tau": [1, 2, 4, 8]}}}
    assert checkpoint_config_overrides(legacy, lambda: V1_KEYS) == ["framework.spatial.adapter_version=v1"]
    explicit = {"framework": {"name": "QwenSpatialMemOFT", "spatial": {"adapter_version": "v1"}}}
    assert checkpoint_config_overrides(explicit, lambda: V1_KEYS) == []
    with pytest.raises(ValueError):
        checkpoint_config_overrides(explicit, lambda: V2_KEYS)


def _bddl_language(path: Path) -> str:
    return " ".join(re.search(r"\(:language\s*(.*?)\)", path.read_text(encoding="utf-8"), re.S).group(1).split())


@pytest.mark.skipif(not PRO_BDDL.is_dir(), reason="LIBERO-PRO data is not available")
def test_shipped_pro_suites_match_the_instruction_rule():
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for pert, spec in PERTURBATIONS.items():
            pro_dir = PRO_BDDL / pro_suite_name(suite, pert)
            files = sorted(pro_dir.glob("*.bddl"))
            assert len(files) == 10, pro_dir
            changed = sum(_bddl_language(f).lower() != _bddl_language(PRO_BDDL / suite / f.name).lower() for f in files)
            if spec["instruction"] == "registry":
                assert changed == 0, pro_dir
            else:
                assert changed == 10, pro_dir
