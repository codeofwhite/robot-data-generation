from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .models import (
    AttemptResultV1,
    GenerationAttemptV1,
    GenerationBatchReportV1,
    GenerationPlanV1,
    RigidCubeLiftConfigV1,
)


def _plan_sha256(plan: GenerationPlanV1) -> str:
    return hashlib.sha256(
        plan.model_dump_json(exclude_none=True).encode("utf-8")
    ).hexdigest()


def _paired_latin_hypercube(count: int, *, seed: int, range_m: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((count, 2), dtype=np.float64)
    for axis in range(2):
        unit = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        values[:, axis] = unit[rng.permutation(count)]
    return (2.0 * values - 1.0) * range_m


def build_generation_plan(config: RigidCubeLiftConfigV1) -> GenerationPlanV1:
    offsets = _paired_latin_hypercube(
        config.attempts_per_source,
        seed=config.randomization_seed,
        range_m=config.object_xy.range_m,
    )
    attempts: list[GenerationAttemptV1] = []
    for source_rank, source in enumerate(config.sources):
        for offset_rank, offset in enumerate(offsets):
            attempt_seed = config.randomization_seed + source_rank * config.attempts_per_source + offset_rank
            attempts.append(
                GenerationAttemptV1(
                    attempt_id=f"source-{source_rank:02d}-attempt-{offset_rank:03d}",
                    source_rank=source_rank,
                    source_episode_path=source.path.expanduser().resolve(),
                    source_episode_uuid=source.episode_uuid,
                    randomization_seed=attempt_seed,
                    object_xy_offset_m=(float(offset[0]), float(offset[1])),
                    transition_frames=config.transition_frames,
                )
            )
    return GenerationPlanV1(
        spec_id=config.spec_id,
        nominal_only=False,
        execution=config.execution,
        attempts=tuple(attempts),
    )


def build_nominal_plan(config: RigidCubeLiftConfigV1) -> GenerationPlanV1:
    attempts = tuple(
        GenerationAttemptV1(
            attempt_id=f"source-{source_rank:02d}-nominal",
            source_rank=source_rank,
            source_episode_path=source.path.expanduser().resolve(),
            source_episode_uuid=source.episode_uuid,
            randomization_seed=config.randomization_seed + source_rank,
            object_xy_offset_m=(0.0, 0.0),
            transition_frames=config.transition_frames,
        )
        for source_rank, source in enumerate(config.sources)
    )
    return GenerationPlanV1(
        spec_id=config.spec_id,
        nominal_only=True,
        execution=config.execution,
        attempts=attempts,
    )


def load_generation_plan(path: str | Path) -> GenerationPlanV1:
    plan_path = Path(path).expanduser().resolve()
    return GenerationPlanV1.model_validate_json(plan_path.read_text(encoding="utf-8"))


def load_batch_report(
    path: str | Path,
    *,
    plan: GenerationPlanV1,
) -> GenerationBatchReportV1:
    report_path = Path(path).expanduser().resolve()
    report = GenerationBatchReportV1.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.spec_id != plan.spec_id or report.requested_attempts != len(plan.attempts):
        raise ValueError("batch report does not match the generation plan")
    if report.plan_sha256 != _plan_sha256(plan):
        raise ValueError("batch report generation-plan digest does not match")
    known_ids = {attempt.attempt_id for attempt in plan.attempts}
    unknown = {result.attempt_id for result in report.results} - known_ids
    if unknown:
        raise ValueError(f"batch report contains unknown attempt IDs: {sorted(unknown)}")
    return report


def new_batch_report(plan: GenerationPlanV1, plan_path: str | Path) -> GenerationBatchReportV1:
    return GenerationBatchReportV1(
        spec_id=plan.spec_id,
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=_plan_sha256(plan),
        requested_attempts=len(plan.attempts),
    )


def pending_attempts(
    plan: GenerationPlanV1,
    report: GenerationBatchReportV1,
) -> tuple[GenerationAttemptV1, ...]:
    completed = {result.attempt_id for result in report.results}
    return tuple(attempt for attempt in plan.attempts if attempt.attempt_id not in completed)


def record_attempt_result(
    report: GenerationBatchReportV1,
    result: AttemptResultV1,
) -> GenerationBatchReportV1:
    if result.attempt_id in {existing.attempt_id for existing in report.results}:
        raise ValueError(f"attempt {result.attempt_id} already has a result")
    results = (*report.results, result)
    return report.model_copy(
        update={
            "results": results,
            "status": "completed" if len(results) == report.requested_attempts else "running",
            "updated_at": datetime.now(timezone.utc),
        }
    )


def write_json_atomic(path: str | Path, payload: BaseModel) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def load_config(path: str | Path) -> RigidCubeLiftConfigV1:
    config_path = Path(path).expanduser().resolve()
    return RigidCubeLiftConfigV1.model_validate_json(config_path.read_text(encoding="utf-8"))
