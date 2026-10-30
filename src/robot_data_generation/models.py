from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceEpisodeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    episode_uuid: str


class ObjectXYRandomizationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    range_m: float = Field(gt=0.0, le=0.10)
    sampling: Literal["paired_latin_hypercube"] = "paired_latin_hypercube"


class TrajectoryExecutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control_hz: int = Field(default=20, gt=0, le=240)
    pregrasp_position_tolerance_m: float = Field(default=0.003, gt=0.0, le=0.05)
    pregrasp_rotation_tolerance_rad: float = Field(default=0.0872664626, gt=0.0, le=1.0)
    pregrasp_stable_ticks: int = Field(default=3, gt=0, le=100)
    pregrasp_timeout_s: float = Field(default=2.0, gt=0.0, le=30.0)
    grasp_lift_threshold_m: float = Field(default=0.02, gt=0.0, le=0.20)
    task_lift_threshold_m: float = Field(default=0.08, gt=0.0, le=0.50)
    success_mode: Literal["lift"] = "lift"

    @model_validator(mode="after")
    def validate_lift_thresholds(self) -> "TrajectoryExecutionV1":
        if self.grasp_lift_threshold_m >= self.task_lift_threshold_m:
            raise ValueError("grasp lift threshold must be below task lift threshold")
        return self


class RigidCubeLiftConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["robot_data_generation_config"] = "robot_data_generation_config"
    format_version: Literal[1] = 1
    spec_id: str
    scene_id: Literal["rigid_cube_lift"] = "rigid_cube_lift"
    attempts_per_source: int = Field(gt=0, le=1000)
    randomization_seed: int = Field(ge=0)
    transition_frames: int = Field(ge=2, le=100)
    object_xy: ObjectXYRandomizationV1
    execution: TrajectoryExecutionV1 = Field(default_factory=TrajectoryExecutionV1)
    sources: tuple[SourceEpisodeRefV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "RigidCubeLiftConfigV1":
        uuids = [source.episode_uuid for source in self.sources]
        paths = [source.path.expanduser().resolve() for source in self.sources]
        if len(set(uuids)) != len(uuids) or len(set(paths)) != len(paths):
            raise ValueError("source episodes must have unique UUIDs and paths")
        return self


class GenerationAttemptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    source_rank: int = Field(ge=0)
    source_episode_path: Path
    source_episode_uuid: str
    randomization_seed: int = Field(ge=0)
    object_xy_offset_m: tuple[float, float]
    transition_frames: int = Field(ge=2)


class GenerationPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["robot_data_generation_plan"] = "robot_data_generation_plan"
    format_version: Literal[1] = 1
    spec_id: str
    scene_id: Literal["rigid_cube_lift"] = "rigid_cube_lift"
    nominal_only: bool
    execution: TrajectoryExecutionV1 | None = None
    attempts: tuple[GenerationAttemptV1, ...]

    @model_validator(mode="after")
    def validate_attempt_ids(self) -> "GenerationPlanV1":
        ids = [attempt.attempt_id for attempt in self.attempts]
        if len(set(ids)) != len(ids):
            raise ValueError("generation attempt IDs must be unique")
        return self


class AttemptResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    status: Literal["succeeded", "failed"]
    episode_path: Path | None = None
    error: str | None = None
    metrics: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> "AttemptResultV1":
        if self.status == "succeeded" and self.episode_path is None:
            raise ValueError("successful attempt requires an episode path")
        if self.status == "failed" and not self.error and self.episode_path is None:
            raise ValueError("failed attempt requires an error or a retained episode")
        return self


class GenerationBatchReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["robot_data_generation_batch_report"] = "robot_data_generation_batch_report"
    format_version: Literal[1] = 1
    spec_id: str
    plan_path: Path
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["running", "completed"] = "running"
    requested_attempts: int = Field(gt=0)
    results: tuple[AttemptResultV1, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_results(self) -> "GenerationBatchReportV1":
        ids = [result.attempt_id for result in self.results]
        if len(set(ids)) != len(ids):
            raise ValueError("batch report contains duplicate attempt results")
        if len(ids) > self.requested_attempts:
            raise ValueError("batch report contains more results than requested attempts")
        if self.status == "completed" and len(ids) != self.requested_attempts:
            raise ValueError("completed batch report must contain every attempt result")
        return self


class EvaluationConditionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition_id: str
    rollout_seed: int = Field(ge=0)
    object_xy_offset_m: tuple[float, float]


class PolicyEvaluationPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["robot_data_policy_evaluation_plan"] = (
        "robot_data_policy_evaluation_plan"
    )
    format_version: Literal[1] = 1
    plan_id: str
    scene_id: Literal["rigid_cube_lift"] = "rigid_cube_lift"
    source_generation_plan: Path
    source_generation_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_generation_batch_report: Path | None = None
    source_generation_batch_report_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    position_split: Literal["heldout", "m2_train"] = "heldout"
    sampling: Literal["heldout_latin_hypercube", "m2_successful_subset"] = (
        "heldout_latin_hypercube"
    )
    randomization_seed: int = Field(ge=0)
    object_xy_range_m: float = Field(gt=0.0, le=0.10)
    minimum_m2_distance_m: float = Field(gt=0.0, le=0.10)
    conditions: tuple[EvaluationConditionV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conditions(self) -> "PolicyEvaluationPlanV1":
        ids = [condition.condition_id for condition in self.conditions]
        seeds = [condition.rollout_seed for condition in self.conditions]
        offsets = [condition.object_xy_offset_m for condition in self.conditions]
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation condition IDs must be unique")
        if len(set(seeds)) != len(seeds):
            raise ValueError("evaluation rollout seeds must be unique")
        if len(set(offsets)) != len(offsets):
            raise ValueError("evaluation object offsets must be unique")
        if any(max(abs(value) for value in offset) > self.object_xy_range_m for offset in offsets):
            raise ValueError("evaluation object offset exceeds configured range")
        if self.position_split == "heldout" and self.sampling != "heldout_latin_hypercube":
            raise ValueError("heldout split requires heldout Latin hypercube sampling")
        if self.position_split == "m2_train" and self.sampling != "m2_successful_subset":
            raise ValueError("M2 train split requires successful-subset sampling")
        if (self.source_generation_batch_report is None) != (
            self.source_generation_batch_report_sha256 is None
        ):
            raise ValueError("generation batch report path and digest must be provided together")
        if self.position_split == "m2_train" and self.source_generation_batch_report is None:
            raise ValueError("M2 train split requires a generation batch report")
        return self
