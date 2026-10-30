from pathlib import Path

import numpy as np
import pytest

from robot_data_generation.generation_plan import (
    build_generation_plan,
    build_nominal_plan,
    load_batch_report,
    new_batch_report,
    pending_attempts,
    record_attempt_result,
    write_json_atomic,
)
from robot_data_generation.models import (
    AttemptResultV1,
    ObjectXYRandomizationV1,
    RigidCubeLiftConfigV1,
    SourceEpisodeRefV1,
)


def _config() -> RigidCubeLiftConfigV1:
    return RigidCubeLiftConfigV1(
        spec_id="test_v1",
        attempts_per_source=4,
        randomization_seed=7,
        transition_frames=5,
        object_xy=ObjectXYRandomizationV1(range_m=0.02),
        sources=(
            SourceEpisodeRefV1(path=Path("/tmp/a"), episode_uuid="a"),
            SourceEpisodeRefV1(path=Path("/tmp/b"), episode_uuid="b"),
        ),
    )


def test_generation_plan_is_deterministic_paired_and_bounded() -> None:
    first = build_generation_plan(_config())
    second = build_generation_plan(_config())

    assert first == second
    assert len(first.attempts) == 8
    assert first.execution.control_hz == 20
    assert first.execution.success_mode == "lift"
    first_offsets = [attempt.object_xy_offset_m for attempt in first.attempts[:4]]
    second_offsets = [attempt.object_xy_offset_m for attempt in first.attempts[4:]]
    np.testing.assert_allclose(first_offsets, second_offsets)
    assert np.max(np.abs(first_offsets)) <= 0.02


def test_nominal_plan_and_resume_bookkeeping() -> None:
    plan = build_nominal_plan(_config())
    report = new_batch_report(plan, "/tmp/plan.json")
    assert len(plan.attempts) == 2
    assert all(attempt.object_xy_offset_m == (0.0, 0.0) for attempt in plan.attempts)

    result = AttemptResultV1(
        attempt_id=plan.attempts[0].attempt_id,
        status="succeeded",
        episode_path=Path("/tmp/episode_000000"),
    )
    updated = record_attempt_result(report, result)
    assert [attempt.attempt_id for attempt in pending_attempts(plan, updated)] == [
        plan.attempts[1].attempt_id
    ]
    with pytest.raises(ValueError, match="already has a result"):
        record_attempt_result(updated, result)


def test_resume_rejects_a_changed_plan(tmp_path) -> None:
    plan = build_nominal_plan(_config())
    report_path = tmp_path / "report.json"
    write_json_atomic(report_path, new_batch_report(plan, tmp_path / "plan.json"))
    changed_attempt = plan.attempts[0].model_copy(
        update={"object_xy_offset_m": (0.001, 0.0)}
    )
    changed = plan.model_copy(update={"attempts": (changed_attempt, *plan.attempts[1:])})

    with pytest.raises(ValueError, match="digest"):
        load_batch_report(report_path, plan=changed)
