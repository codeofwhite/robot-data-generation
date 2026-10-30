import json
from pathlib import Path

import pytest

from robot_data_generation.generation_plan import (
    build_generation_plan,
    new_batch_report,
    record_attempt_result,
)
from robot_data_generation.models import (
    AttemptResultV1,
    ObjectXYRandomizationV1,
    RigidCubeLiftConfigV1,
    SourceEpisodeRefV1,
)
from robot_data_generation.training_selection import prepare_training_selections


def _episode(path: Path, uuid: str, *, success: bool, parent: str | None = None) -> None:
    path.mkdir()
    (path / "payload.bin").write_bytes(b"episode")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "episode_uuid": uuid,
                "num_frames": 10,
                "fps": 10,
                "camera_names": ["base", "top", "side", "wrist"],
                "control_source": "retargeted_trajectory" if parent else "teleoperation",
                "evaluator": {"success": success},
                "lineage": {"parent_episode_uuid": parent},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, generated_success: bool = True):
    sources = []
    for index in range(2):
        path = tmp_path / f"source_{index}"
        _episode(path, f"source-{index}", success=True)
        sources.append(SourceEpisodeRefV1(path=path, episode_uuid=f"source-{index}"))
    config = RigidCubeLiftConfigV1(
        spec_id="test_v1",
        attempts_per_source=1,
        randomization_seed=7,
        transition_frames=5,
        object_xy=ObjectXYRandomizationV1(range_m=0.02),
        sources=tuple(sources),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    plan = build_generation_plan(config)
    report = new_batch_report(plan, tmp_path / "plan.json")
    generated = tmp_path / "generated_0"
    _episode(generated, "generated-0", success=generated_success, parent="source-0")
    report = record_attempt_result(
        report,
        AttemptResultV1(
            attempt_id=plan.attempts[0].attempt_id,
            status="succeeded",
            episode_path=generated,
            metrics={"success": True},
        ),
    )
    report = record_attempt_result(
        report,
        AttemptResultV1(
            attempt_id=plan.attempts[1].attempt_id,
            status="failed",
            error="automatic_criterion_not_met",
            metrics={"success": False},
        ),
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    return config_path, report_path


def test_prepare_training_selections_excludes_failures_and_stages_views(tmp_path) -> None:
    config, report = _inputs(tmp_path)
    result = prepare_training_selections(
        config_path=config,
        batch_report_path=report,
        output=tmp_path / "selection",
        minimum_generated_successes=1,
    )

    payload = json.loads(result.read_text())
    assert payload["generated_successes"] == 1
    assert len(payload["excluded_attempts"]) == 1
    assert payload["variants"]["human5"]["episode_count"] == 2
    assert payload["variants"]["human5_generated_success"]["episode_count"] == 3
    assert len(list((tmp_path / "selection" / "human5").glob("*/manifest.json"))) == 2
    staged = tmp_path / "selection" / "human5_generated_success" / "episode_000002"
    assert (staged / "manifest.json").is_symlink()
    assert (staged / "payload.bin").read_bytes() == b"episode"


def test_prepare_training_selections_rejects_evaluator_disagreement(tmp_path) -> None:
    config, report = _inputs(tmp_path, generated_success=False)
    with pytest.raises(ValueError, match="not an automatic success"):
        prepare_training_selections(
            config_path=config,
            batch_report_path=report,
            output=tmp_path / "selection",
            minimum_generated_successes=1,
        )
