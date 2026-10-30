from __future__ import annotations

import numpy as np

from robot_data_generation.trajectory_alignment import (
    _angular_distance,
    _command_close_frame,
    _phase_events,
)


def test_phase_events_resolve_first_boundary() -> None:
    phase = np.asarray([[0], [0], [1], [1], [2], [3], [3], [4]], dtype=np.int8)
    assert _phase_events(phase) == {"close": 2, "grasp": 4, "lift": 5, "release": 7}


def test_angular_distance_ignores_quaternion_sign() -> None:
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert _angular_distance(quaternion, -quaternion) == 0.0


def test_command_close_frame_uses_applied_width_not_phase_start() -> None:
    actions = np.zeros((6, 8), dtype=np.float32)
    actions[:, 7] = [0.08, 0.08, 0.08, 0.08, 0.0, 0.0]
    assert _command_close_frame(actions) == 4
