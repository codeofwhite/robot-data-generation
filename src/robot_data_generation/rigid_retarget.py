from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source_episode import SourceTrajectory


@dataclass(frozen=True)
class RetargetedTrajectory:
    target_ee_pose: np.ndarray
    gripper_closed: np.ndarray
    source_frame_index: np.ndarray
    source_frame_coordinate: np.ndarray
    phase_id: np.ndarray
    close_waypoint_index: int
    object_start_position_m: tuple[float, float, float]

    @property
    def length(self) -> int:
        return int(self.target_ee_pose.shape[0])


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("controller target contains a zero quaternion")
    return quaternions / norms


def _slerp_pair(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path quaternion interpolation for wxyz quaternions."""

    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(dot))
    scale = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / scale * first
        + np.sin(fraction * angle) / scale * second
    )


def _resample_pose(
    poses: np.ndarray,
    source_coordinates: np.ndarray,
) -> np.ndarray:
    frame_axis = np.arange(len(poses), dtype=np.float64)
    positions = np.column_stack(
        [np.interp(source_coordinates, frame_axis, poses[:, axis]) for axis in range(3)]
    )
    source_quaternions = _normalize_quaternions(poses[:, 3:7].astype(np.float64))
    quaternions = np.empty((len(source_coordinates), 4), dtype=np.float64)
    for index, coordinate in enumerate(source_coordinates):
        lower = min(int(np.floor(coordinate)), len(poses) - 1)
        upper = min(lower + 1, len(poses) - 1)
        quaternions[index] = _slerp_pair(
            source_quaternions[lower],
            source_quaternions[upper],
            float(coordinate - lower),
        )
    return np.concatenate((positions, quaternions), axis=1).astype(np.float32)


def _phase_ids(source: SourceTrajectory, coordinates: np.ndarray) -> np.ndarray:
    phases = source.phases
    result = np.zeros(len(coordinates), dtype=np.int8)
    result[coordinates >= phases.close_frame] = 1
    result[coordinates >= phases.grasp_frame] = 2
    result[coordinates >= phases.lift_frame] = 3
    result[coordinates >= phases.release_frame] = 4
    return result


def retarget_rigid_cube_lift(
    source: SourceTrajectory,
    controller_target_pose: np.ndarray,
    object_xy_offset_m: tuple[float, float],
    *,
    transition_frames: int,
    target_hz: int,
    closed_width_threshold_m: float = 0.04,
) -> RetargetedTrajectory:
    """Build an object-centric waypoint trajectory from command-aligned FK poses.

    The offset stays zero through the early approach, then is blended to its full
    value over ``transition_frames`` before contact. This preserves the demonstrated
    home pose and avoids an instantaneous target jump while ensuring that contact
    and all later phases remain expressed relative to the randomized object position.
    Pose interpolation is performed at ``target_hz`` while gripper transitions remain
    discrete at their demonstrated source-frame boundaries.
    """

    controller_targets = np.asarray(controller_target_pose, dtype=np.float32)
    if controller_targets.shape != (source.length, 7):
        raise ValueError("controller_target_pose must have shape [source frames, 7]")
    if not np.isfinite(controller_targets).all():
        raise ValueError("controller_target_pose must be finite")
    offset = np.asarray((*object_xy_offset_m, 0.0), dtype=np.float32)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("object XY offset must contain two finite values")
    if transition_frames < 2:
        raise ValueError("transition_frames must be at least two")
    if target_hz < source.fps or target_hz % source.fps:
        raise ValueError("target_hz must be an integer multiple of source fps")
    close_frame = source.phases.close_frame
    transition_start = max(0, close_frame - transition_frames)
    blend = np.zeros(source.length, dtype=np.float32)
    blend[transition_start : close_frame + 1] = _smoothstep(
        np.linspace(
            0.0, 1.0, close_frame - transition_start + 1, dtype=np.float32
        )
    )
    blend[close_frame + 1 :] = 1.0
    source_targets = controller_targets.copy()
    source_targets[:, :3] += blend[:, None] * offset[None, :]

    ratio = target_hz // source.fps
    source_coordinates = np.arange(
        (source.length - 1) * ratio + 1, dtype=np.float64
    ) / ratio
    target = _resample_pose(source_targets, source_coordinates)
    source_frame_index = np.floor(source_coordinates + 1e-9).astype(np.int64)
    source_closed = source.applied_action[:, 7] < closed_width_threshold_m
    gripper_closed = source_closed[source_frame_index]
    close_indices = np.flatnonzero(source_coordinates >= close_frame)
    if not len(close_indices):
        raise ValueError("resampled trajectory has no close waypoint")
    object_start = source.object_pose[0, :3] + offset
    return RetargetedTrajectory(
        target_ee_pose=target,
        gripper_closed=gripper_closed.astype(np.bool_),
        source_frame_index=source_frame_index,
        source_frame_coordinate=source_coordinates.astype(np.float32),
        phase_id=_phase_ids(source, source_coordinates),
        close_waypoint_index=int(close_indices[0]),
        object_start_position_m=tuple(float(value) for value in object_start),
    )
