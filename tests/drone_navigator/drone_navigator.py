"""
Unit tests for drone_navigator module verifying move_toward behavior
"""

import math
import pytest
from tello_obstacle_detection.drone_navigator.drone_navigator import move_toward


class MockDrone:
    def __init__(self):
        self.actions = []

    def rotate_clockwise(self, angle_cm):
        self.actions.append(('cw', angle_cm))

    def rotate_counter_clockwise(self, angle_cm):
        self.actions.append(('ccw', angle_cm))

    def move_forward(self, distance_cm):
        self.actions.append(('fwd', distance_cm))


@pytest.fixture
def drone():
    """Provides a fresh mock drone for each test"""
    return MockDrone()


def test_no_movement_when_at_target(drone):
    """If the drone is already at the waypoint, it should not rotate or move"""
    start_pos = (1.0, 2.0)
    heading = 45.0
    end_pos, end_heading = move_toward(drone, target=start_pos, pos=start_pos, heading=heading)
    assert end_pos == start_pos
    assert end_heading == heading
    assert drone.actions == []


def test_move_forward_on_y_axis(drone):
    """Target directly ahead on +Y should result in zero turn and forward movement"""
    start_pos = (0.0, 0.0)
    heading = 0.0  # facing +Y
    target = (0.0, 1.0)
    new_pos, new_heading = move_toward(drone, target, start_pos, heading)
    # Expect no rotation and a 100cm forward move
    assert drone.actions[0] == ('cw', 0)
    assert drone.actions[1] == ('fwd', 100)
    assert pytest.approx(new_pos[0], abs=1e-6) == 0.0
    assert pytest.approx(new_pos[1], abs=1e-6) == 1.0
    assert new_heading == 0.0


def test_turn_and_move_right(drone):
    """Target on +X should rotate +90° then move forward"""
    start_pos = (0.0, 0.0)
    heading = 0.0  # facing +Y
    target = (1.0, 0.0)
    new_pos, new_heading = move_toward(drone, target, start_pos, heading)
    # First action should be a 90° clockwise turn
    assert ('cw', 90) in drone.actions
    # Then a 100cm forward move
    assert ('fwd', 100) in drone.actions
    assert pytest.approx(new_pos[0], abs=1e-6) == 1.0
    assert pytest.approx(new_pos[1], abs=1e-6) == 0.0
    assert new_heading == 90.0


def test_turn_and_move_left(drone):
    """Target on -X should rotate counter‑clockwise 90° then move forward"""
    start_pos = (0.0, 0.0)
    heading = 0.0  # facing +Y
    target = (-1.0, 0.0)
    new_pos, new_heading = move_toward(drone, target, start_pos, heading)
    # First action should be a 90° counter‑clockwise turn
    assert ('ccw', 90) in drone.actions
    # Then a 100cm forward move
    assert ('fwd', 100) in drone.actions
    assert pytest.approx(new_pos[0], abs=1e-6) == -1.0
    assert pytest.approx(new_pos[1], abs=1e-6) == 0.0
    # Heading should wrap to 270°
    assert new_heading == 270.0
