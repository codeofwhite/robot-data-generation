from pathlib import Path

import numpy as np

from robot_data_generation.rigid_retarget import retarget_rigid_cube_lift
from robot_data_generation.source_episode import GripperPhaseBoundaries, SourceTrajectory


def _source() -> SourceTrajectory:
    frames = 12
    ee_pose = np.zeros((frames, 7), dtype=np.float32)
    ee_pose[:, 0] = np.linspace(0.3, 0.5, frames)
    ee_pose[:, 3] = 1.0
    object_pose = np.zeros((frames, 7), dtype=np.float32)
    object_pose[:, :3] = (0.55, -0.10, 0.277)
    object_pose[:, 3] = 1.0
    actions = np.zeros((frames, 8), dtype=np.float32)
    actions[:, 7] = 0.08
    actions[6:10, 7] = 0.0
    return SourceTrajectory(
        path=Path("/tmp/source"),
        episode_uuid="source-uuid",
        task="lift",
        fps=10,
        ee_pose=ee_pose,
        object_pose=object_pose,
        applied_action=actions,
        phases=GripperPhaseBoundaries(
            close_frame=6, grasp_frame=7, lift_frame=8, release_frame=10
        ),
    )


def _controller_targets(source: SourceTrajectory) -> np.ndarray:
    targets = source.ee_pose.copy()
    targets[:, 0] += 0.1
    return targets


def test_retarget_preserves_home_and_reaches_full_offset_before_contact() -> None:
    source = _source()

    controller_targets = _controller_targets(source)
    result = retarget_rigid_cube_lift(
        source,
        controller_targets,
        (0.02, -0.01),
        transition_frames=2,
        target_hz=20,
    )

    np.testing.assert_allclose(result.target_ee_pose[0], controller_targets[0])
    np.testing.assert_allclose(result.target_ee_pose[6], controller_targets[3])
    np.testing.assert_allclose(
        result.target_ee_pose[source.phases.close_frame * 2, :3],
        controller_targets[source.phases.close_frame, :3] + (0.02, -0.01, 0.0),
    )
    np.testing.assert_allclose(
        result.target_ee_pose[::2, 3:], controller_targets[:, 3:]
    )
    np.testing.assert_allclose(
        result.object_start_position_m, (0.57, -0.11, 0.277)
    )
    assert result.length == 23
    assert result.close_waypoint_index == 12
    assert result.gripper_closed[12:20].all()
    assert not result.gripper_closed[11]
    assert not result.gripper_closed[20]
    assert result.phase_id[12] == 1
    assert result.phase_id[16] == 3


def test_zero_offset_is_an_exact_pose_replay() -> None:
    source = _source()
    controller_targets = _controller_targets(source)
    result = retarget_rigid_cube_lift(
        source,
        controller_targets,
        (0.0, 0.0),
        transition_frames=2,
        target_hz=20,
    )
    np.testing.assert_array_equal(result.target_ee_pose[::2], controller_targets)


def test_quaternion_resampling_uses_shortest_path_and_stays_normalized() -> None:
    source = _source()
    targets = _controller_targets(source)
    targets[1, 3:] = (-0.9998477, 0.0, 0.0, -0.0174524)
    result = retarget_rigid_cube_lift(
        source, targets, (0.0, 0.0), transition_frames=2, target_hz=20
    )

    np.testing.assert_allclose(
        np.linalg.norm(result.target_ee_pose[:, 3:], axis=1), 1.0, atol=1e-6
    )
    assert abs(float(result.target_ee_pose[1, 3])) > 0.99
