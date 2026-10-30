import numpy as np
import pytest

from robot_data_generation.source_episode import segment_gripper_phases


def test_segment_gripper_phases_requires_close_then_release() -> None:
    actions = np.zeros((8, 8), dtype=np.float32)
    actions[:, 7] = 0.08
    actions[2:6, 7] = 0.0
    objects = np.zeros((8, 7), dtype=np.float32)
    objects[:, 3] = 1.0
    objects[:, 2] = [0.277, 0.277, 0.277, 0.28, 0.30, 0.37, 0.37, 0.277]

    phases = segment_gripper_phases(actions, objects)

    assert phases.close_frame == 2
    assert phases.grasp_frame == 4
    assert phases.lift_frame == 5
    assert phases.release_frame == 6


def test_segment_gripper_phases_rejects_missing_release() -> None:
    actions = np.zeros((5, 8), dtype=np.float32)
    actions[0, 7] = 0.08
    objects = np.zeros((5, 7), dtype=np.float32)
    objects[:, 3] = 1.0

    with pytest.raises(ValueError, match="no release"):
        segment_gripper_phases(actions, objects)


def test_segment_ignores_accidental_close_without_object_motion() -> None:
    actions = np.zeros((14, 8), dtype=np.float32)
    actions[:, 7] = 0.08
    actions[2:4, 7] = 0.0
    actions[6:12, 7] = 0.0
    objects = np.zeros((14, 7), dtype=np.float32)
    objects[:, 3] = 1.0
    objects[:, 2] = 0.277
    objects[9:13, 2] = (0.30, 0.37, 0.38, 0.38)

    phases = segment_gripper_phases(actions, objects)

    assert phases.close_frame == 6
    assert phases.release_frame == 12


def test_segment_rejects_two_physically_valid_grasp_intervals() -> None:
    actions = np.zeros((16, 8), dtype=np.float32)
    actions[:, 7] = 0.08
    actions[2:7, 7] = 0.0
    actions[9:14, 7] = 0.0
    objects = np.zeros((16, 7), dtype=np.float32)
    objects[:, 3] = 1.0
    objects[:, 2] = 0.277
    objects[4:7, 2] = (0.30, 0.37, 0.37)
    objects[11:14, 2] = (0.30, 0.37, 0.37)

    with pytest.raises(ValueError, match="multiple physically valid"):
        segment_gripper_phases(actions, objects)
