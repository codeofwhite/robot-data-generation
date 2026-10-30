from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_data_contracts import CanonicalEpisode, open_canonical_episode

from .models import SourceEpisodeRefV1


@dataclass(frozen=True)
class GripperPhaseBoundaries:
    close_frame: int
    grasp_frame: int
    lift_frame: int
    release_frame: int


@dataclass(frozen=True)
class SourceTrajectory:
    path: Path
    episode_uuid: str
    task: str
    fps: int
    ee_pose: np.ndarray
    object_pose: np.ndarray
    applied_action: np.ndarray
    phases: GripperPhaseBoundaries

    @property
    def length(self) -> int:
        return int(self.ee_pose.shape[0])


def segment_gripper_phases(
    applied_action: np.ndarray,
    object_pose: np.ndarray,
    *,
    closed_width_threshold_m: float = 0.04,
    grasp_lift_threshold_m: float = 0.02,
    task_lift_threshold_m: float = 0.08,
) -> GripperPhaseBoundaries:
    """Select the unique close interval that physically lifts the cube.

    Close intervals that do not move the object by ``grasp_lift_threshold_m``
    are treated as harmless operator mis-clicks. More than one physically valid
    interval is ambiguous and is rejected instead of silently choosing one.
    """

    actions = np.asarray(applied_action)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError("applied_action must have shape [frames, 8]")
    if len(actions) < 3 or not np.isfinite(actions).all():
        raise ValueError("applied_action must contain at least three finite frames")
    objects = np.asarray(object_pose)
    if objects.shape != (len(actions), 7) or not np.isfinite(objects).all():
        raise ValueError("object_pose must have shape [frames, 7] and be finite")
    if not 0.0 < grasp_lift_threshold_m < task_lift_threshold_m:
        raise ValueError("lift thresholds must satisfy 0 < grasp < task lift")
    widths = actions[:, 7]
    closed_mask = widths < closed_width_threshold_m
    if not closed_mask.any():
        raise ValueError("source episode has no gripper close transition")

    starts = np.flatnonzero(closed_mask & np.r_[True, ~closed_mask[:-1]])
    releases = np.flatnonzero(~closed_mask & np.r_[False, closed_mask[:-1]])
    intervals: list[tuple[int, int]] = []
    for start in starts:
        later_releases = releases[releases > start]
        if len(later_releases):
            intervals.append((int(start), int(later_releases[0])))
    if not intervals:
        raise ValueError("source episode has no release after close")

    initial_z = float(objects[0, 2])
    lift = objects[:, 2].astype(np.float64) - initial_z
    valid = [
        (start, release)
        for start, release in intervals
        if float(np.max(lift[start:release])) >= grasp_lift_threshold_m
    ]
    if not valid:
        raise ValueError("source episode has no close interval that physically lifts the cube")
    if len(valid) > 1:
        raise ValueError("source episode has multiple physically valid grasp intervals")

    close_frame, release_frame = valid[0]
    if close_frame == 0 or release_frame >= len(widths) - 1:
        raise ValueError("source episode must include approach and post-release frames")
    grasp_candidates = np.flatnonzero(
        (np.arange(len(lift)) >= close_frame)
        & (np.arange(len(lift)) < release_frame)
        & (lift >= grasp_lift_threshold_m)
    )
    lift_candidates = np.flatnonzero(
        (np.arange(len(lift)) >= close_frame)
        & (np.arange(len(lift)) < release_frame)
        & (lift >= task_lift_threshold_m)
    )
    if len(lift_candidates) == 0:
        raise ValueError("selected grasp interval never reaches the task lift threshold")
    return GripperPhaseBoundaries(
        close_frame=close_frame,
        grasp_frame=int(grasp_candidates[0]),
        lift_frame=int(lift_candidates[0]),
        release_frame=release_frame,
    )


def _validate_cube_lift_episode(episode: CanonicalEpisode, expected_uuid: str) -> None:
    manifest = episode.manifest
    if manifest.episode_uuid != expected_uuid:
        raise ValueError(
            f"source UUID mismatch: expected {expected_uuid}, got {manifest.episode_uuid}"
        )
    if manifest.scene.scene_id != "rigid_cube_lift":
        raise ValueError(f"source scene must be rigid_cube_lift, got {manifest.scene.scene_id}")
    if manifest.operator_success is not True:
        raise ValueError("source episode must be operator-successful")
    if not manifest.evaluator.available or manifest.evaluator.success is not True:
        raise ValueError("source episode must pass its native automatic evaluator")
    if manifest.lineage.generation_method != "teleoperation":
        raise ValueError("v1 sources must be original teleoperation episodes")


def load_source_trajectory(
    source: SourceEpisodeRefV1,
    *,
    grasp_lift_threshold_m: float = 0.02,
    task_lift_threshold_m: float = 0.08,
) -> SourceTrajectory:
    episode = open_canonical_episode(
        source.path,
        array_keys={"ee_pose", "object_pose", "applied_action"},
    )
    _validate_cube_lift_episode(episode, source.episode_uuid)
    ee_pose = np.asarray(episode.arrays["ee_pose"], dtype=np.float32)
    object_pose = np.asarray(episode.arrays["object_pose"], dtype=np.float32)
    applied_action = np.asarray(episode.arrays["applied_action"], dtype=np.float32)
    expected_length = episode.length
    if ee_pose.shape != (expected_length, 7):
        raise ValueError("source ee_pose must have shape [frames, 7]")
    if object_pose.shape != (expected_length, 7):
        raise ValueError("source object_pose must have shape [frames, 7]")
    if applied_action.shape != (expected_length, 8):
        raise ValueError("source applied_action must have shape [frames, 8]")
    if not all(np.isfinite(array).all() for array in (ee_pose, object_pose, applied_action)):
        raise ValueError("source trajectory arrays must be finite")
    return SourceTrajectory(
        path=episode.path,
        episode_uuid=episode.manifest.episode_uuid,
        task=episode.manifest.task,
        fps=episode.manifest.fps,
        ee_pose=ee_pose,
        object_pose=object_pose,
        applied_action=applied_action,
        phases=segment_gripper_phases(
            applied_action,
            object_pose,
            grasp_lift_threshold_m=grasp_lift_threshold_m,
            task_lift_threshold_m=task_lift_threshold_m,
        ),
    )
