import json
from pathlib import Path

import numpy as np
import pytest

from robot_data_generation.evaluation_plan import (
    build_evaluation_plan,
    build_training_evaluation_plan,
    compare_evaluation_reports,
    diagnose_evaluation_splits,
    load_evaluation_plan,
)
from robot_data_generation.generation_plan import (
    build_generation_plan,
    new_batch_report,
    write_json_atomic,
)
from robot_data_generation.models import (
    AttemptResultV1,
    ObjectXYRandomizationV1,
    RigidCubeLiftConfigV1,
    SourceEpisodeRefV1,
)


def _generation_plan(tmp_path: Path) -> Path:
    config = RigidCubeLiftConfigV1(
        spec_id="m2_test",
        attempts_per_source=20,
        randomization_seed=7,
        transition_frames=5,
        object_xy=ObjectXYRandomizationV1(range_m=0.02),
        sources=(SourceEpisodeRefV1(path=tmp_path / "source", episode_uuid="a"),),
    )
    path = tmp_path / "m2.json"
    write_json_atomic(path, build_generation_plan(config))
    return path


def test_evaluation_plan_is_deterministic_bounded_and_leakage_free(tmp_path: Path) -> None:
    source = _generation_plan(tmp_path)
    first = build_evaluation_plan(source, plan_id="m4", seed=101)
    second = build_evaluation_plan(source, plan_id="m4", seed=101)

    assert first == second
    assert len(first.conditions) == 20
    training = np.asarray(
        [attempt.object_xy_offset_m for attempt in build_generation_plan(
            RigidCubeLiftConfigV1(
                spec_id="m2_test", attempts_per_source=20, randomization_seed=7,
                transition_frames=5, object_xy=ObjectXYRandomizationV1(range_m=0.02),
                sources=(SourceEpisodeRefV1(path=tmp_path / "source", episode_uuid="a"),),
            )
        ).attempts]
    )
    for condition in first.conditions:
        offset = np.asarray(condition.object_xy_offset_m)
        assert np.max(np.abs(offset)) <= 0.02
        assert np.min(np.linalg.norm(training - offset, axis=1)) >= 0.001

    output = tmp_path / "evaluation.json"
    write_json_atomic(output, first)
    assert load_evaluation_plan(output) == first


def test_training_evaluation_plan_uses_only_distinct_successful_m2_positions(
    tmp_path: Path,
) -> None:
    source = _generation_plan(tmp_path)
    generation = build_generation_plan(
        RigidCubeLiftConfigV1(
            spec_id="m2_test", attempts_per_source=20, randomization_seed=7,
            transition_frames=5, object_xy=ObjectXYRandomizationV1(range_m=0.02),
            sources=(SourceEpisodeRefV1(path=tmp_path / "source", episode_uuid="a"),),
        )
    )
    report = new_batch_report(generation, source).model_copy(update={
        "status": "completed",
        "results": tuple(
            AttemptResultV1(
                attempt_id=attempt.attempt_id,
                status="succeeded" if index < 12 else "failed",
                episode_path=tmp_path / f"episode_{index}" if index < 12 else None,
                error=None if index < 12 else "failed",
            )
            for index, attempt in enumerate(generation.attempts)
        ),
    })
    report_path = tmp_path / "batch.json"
    write_json_atomic(report_path, report)

    plan = build_training_evaluation_plan(
        source, report_path, plan_id="m5a", count=10, seed=303
    )
    assert plan.position_split == "m2_train"
    assert plan.sampling == "m2_successful_subset"
    assert len(plan.conditions) == 10
    successful_offsets = {
        attempt.object_xy_offset_m for attempt in generation.attempts[:12]
    }
    assert all(
        condition.object_xy_offset_m in successful_offsets
        for condition in plan.conditions
    )
    output = tmp_path / "training_evaluation.json"
    write_json_atomic(output, plan)
    assert load_evaluation_plan(output) == plan


def _report(path: Path, checkpoint: str, values: list[bool]) -> None:
    rollouts = []
    for index, value in enumerate(values):
        rollouts.append({
            "condition_id": f"heldout-{index:03d}",
            "object_xy_offset_m": [index / 1000, 0.0],
            "frames": 400,
            "success": value,
            "metrics": {stage: value for stage in ("grasp", "lift", "hold", "release", "success")},
        })
    path.write_text(json.dumps({
        "status": "completed", "evaluation_plan_sha256": "a" * 64,
        "requested_rollouts": len(values), "control_fps": 10.0,
        "replan_steps": 10, "full_horizon": True,
        "gripper_filter": {"mode": "hysteresis"},
        "policy_checkpoint": checkpoint, "rollouts": rollouts,
    }))


def test_paired_comparison_keeps_all_conditions_and_discordance(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    augmented = tmp_path / "augmented.json"
    _report(baseline, "base", [True, False, False, True])
    _report(augmented, "aug", [True, True, False, False])

    result = compare_evaluation_reports(
        baseline, augmented,
        output_json=tmp_path / "paired.json", output_csv=tmp_path / "paired.csv",
    )
    assert result["paired_conditions"] == 4
    assert result["summary"]["success"]["baseline_only"] == 1
    assert result["summary"]["success"]["augmented_only"] == 1
    assert result["summary"]["success"]["delta_rate"] == 0.0
    assert result["claim_c2_seed0"] == "no_difference"


def test_comparison_rejects_unpaired_settings(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    augmented = tmp_path / "augmented.json"
    _report(baseline, "base", [True])
    _report(augmented, "aug", [True])
    payload = json.loads(augmented.read_text())
    payload["replan_steps"] = 5
    augmented.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="replan_steps"):
        compare_evaluation_reports(
            baseline, augmented,
            output_json=tmp_path / "paired.json", output_csv=tmp_path / "paired.csv",
        )


def _diagnostic_report(tmp_path: Path, split: str, successes: list[bool]) -> Path:
    rollouts = []
    for index, success in enumerate(successes):
        episode = tmp_path / split / f"episode_{index:03d}"
        episode.mkdir(parents=True)
        np.savez(
            episode / "trajectory.npz",
            ee_pose=np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.4]]),
            object_pose=np.asarray([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]]),
            policy_gripper_filtered_closed=np.asarray([[False], [True]]),
            policy_replan_id=np.asarray([[0], [1]]),
            requested_action=np.asarray([[0.0] * 9, [0.1] * 9]),
        )
        rollouts.append({
            "condition_id": f"{split}-{index:03d}",
            "object_xy_offset_m": [0.001 * index, 0.0],
            "episode_path": str(episode),
            "success": success,
            "metrics": {stage: success for stage in ("grasp", "lift", "hold", "release", "success")},
            "gripper_diagnostics": {"raw_switches": 1, "applied_switches": 1},
        })
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps({
        "status": "completed", "scene": "rigid_cube_lift",
        "policy_checkpoint": "checkpoint", "policy_config": "config",
        "control_fps": 10.0, "replan_steps": 10, "inference_steps": 2,
        "full_horizon": True, "gripper_filter": {"mode": "hysteresis"},
        "rollouts": rollouts,
    }))
    return path


def test_m5a_diagnosis_identifies_training_or_executor_failure(tmp_path: Path) -> None:
    reports = {
        "nominal": _diagnostic_report(tmp_path, "nominal", [False] * 5),
        "m2_train": _diagnostic_report(tmp_path, "m2_train", [True, True] + [False] * 8),
        "heldout": _diagnostic_report(tmp_path, "heldout", [True] + [False] * 19),
    }
    result = diagnose_evaluation_splits(
        reports,
        output_json=tmp_path / "diagnosis.json",
        output_csv=tmp_path / "diagnosis.csv",
    )
    assert result["diagnosis"] == "training_action_or_executor_failure"
    assert result["summary"]["m2_train"]["stages"]["success"]["successes"] == 2
    assert result["rows"][0]["first_close_hand_object_distance_m"] == pytest.approx(0.1)
    assert result["rows"][0]["max_replan_boundary_joint_jump_rad"] == pytest.approx(0.1)
