"""Top-level orchestrator for Tello obstacle‑avoidance demo.

This module stitches together the different building blocks in the repository
to run an adaptive navigation loop on a DJI Tello.  The focus is on showing how
the components interact rather than providing a production‑ready flight
controller.  The flow follows the high‑level algorithm described in the
repository documentation:

1.  Initialise the drone and the video/depth pipelines.
2.  Build a navigation graph and plan a path to the goal.
3.  Take off and repeatedly:
    * acquire the latest depth map
    * choose a safe direction
    * adapt (prune, change altitude, rotate or replan) as required
    * move towards the next waypoint
4.  Land and shut everything down when finished.

The code is intentionally conservative – many functions simply call into helper
modules and the drone movement primitives come from ``drone_navigator``.  It is
written so the file can be executed directly while keeping imports cheap when
the module is merely imported (e.g. during unit tests).
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Tuple

from djitellopy import Tello

from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
from tello_obstacle_detection.midas_pipeline import input_pipeline, output_pipeline
from tello_obstacle_detection.path_calculator.path_calculator import find_path
from tello_obstacle_detection.drone_navigator.direction_navigator import (
    Direction,
    choose_safe_direction,
)
from tello_obstacle_detection.drone_navigator.drone_navigator import (
    is_facing_target,
    move_toward,
)


# ---------------------------------------------------------------------------
# Configuration parameters – tweak as needed for the environment
# ---------------------------------------------------------------------------
WIDTH = 3.0  # metres east–west span
LENGTH = 5.0  # metres north–south span
SPACING = 0.5  # distance between grid nodes (metres)
GOAL = (2.5, 4.0)  # target coordinate in metres
ALTITUDE = 1.0  # starting altitude in metres

DEPTH_THRESHOLD = 0.5  # metres, threshold for "clear" in depth map
ALTITUDE_STEP = 0.2  # metres to climb/descend when adjusting altitude
MIN_ALTITUDE = 0.5  # metres – don't descend below this
CONTROL_INTERVAL = 0.1  # seconds between control loop iterations


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def rotate_drone_90(drone: Tello, direction: Direction) -> None:
    """Rotate the drone 90° left or right."""
    if direction == Direction.LEFT:
        drone.rotate_counter_clockwise(90)
    elif direction == Direction.RIGHT:
        drone.rotate_clockwise(90)


def prune_facing_node(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Drop the current front waypoint from the path."""
    return path[1:] if len(path) > 1 else []


# ---------------------------------------------------------------------------
# Main adaptive navigation routine
# ---------------------------------------------------------------------------
def run():
    """Entry point for the obstacle‑aware navigation demo."""

    # ----- Startup -----
    drone = Tello()
    drone.connect()
    drone.streamon()

    # Input pipeline: mirror frames to MJPEG server for debugging
    frame_reader = drone.get_frame_read()
    input_pipeline.start_flask_server()
    threading.Thread(
        target=input_pipeline.mirror_djitellopy_frames,
        args=(frame_reader,),
        daemon=True,
    ).start()

    # Output pipeline: background depth fetcher
    depth_stop_event = output_pipeline.start_background_fetcher()

    # Build navigation graph and initial plan
    G, nodes = build_grid_x_graph(WIDTH, LENGTH, SPACING)
    path = find_path(G, nodes, start=(0.0, 0.0), goal=GOAL)

    # Take off and climb
    drone.takeoff()
    time.sleep(1)
    drone.move_up(int(ALTITUDE * 100))

    current_pos = (0.0, 0.0)
    heading = 0.0  # degrees, 0 = north
    altitude = ALTITUDE

    # ----- Adaptive Navigation Loop -----
    while path:
        depth = output_pipeline.get_latest_depth()
        if depth is None:
            # Depth missing or stale – trigger refresh and retry
            output_pipeline.fetch_depth_once()
            time.sleep(0.1)
            continue

        direction, status = choose_safe_direction(depth, DEPTH_THRESHOLD)

        if direction == Direction.CENTER:
            next_wp = path[0]

        elif direction == Direction.UP:
            drone.move_up(int(ALTITUDE_STEP * 100))
            altitude += ALTITUDE_STEP
            output_pipeline.fetch_depth_once()
            continue

        elif direction == Direction.DOWN and altitude - ALTITUDE_STEP >= MIN_ALTITUDE:
            drone.move_down(int(ALTITUDE_STEP * 100))
            altitude -= ALTITUDE_STEP
            output_pipeline.fetch_depth_once()
            continue

        elif direction in (Direction.LEFT, Direction.RIGHT):
            rotate_drone_90(drone, direction)
            heading = (heading + (90 if direction == Direction.RIGHT else -90)) % 360
            output_pipeline.fetch_depth_once()
            new_depth = output_pipeline.get_latest_depth()
            if new_depth is None:
                continue
            new_dir, _ = choose_safe_direction(new_depth, DEPTH_THRESHOLD)
            if new_dir == Direction.CENTER:
                next_wp = path[0]
            else:
                path = prune_facing_node(path)
                path = find_path(G, nodes, current_pos, GOAL)
                continue

        elif not status.get("center", False):
            path = prune_facing_node(path)
            path = find_path(G, nodes, current_pos, GOAL)
            continue

        else:
            # Ambiguous/block – request new depth and retry
            output_pipeline.fetch_depth_once()
            time.sleep(0.1)
            continue

        # Facing waypoint? grab fresh depth for better alignment
        if is_facing_target(heading, current_pos, next_wp):
            output_pipeline.fetch_depth_once()

        # Move towards waypoint
        current_pos, heading = move_toward(drone, next_wp, current_pos, heading)
        if math.isclose(current_pos[0], next_wp[0], abs_tol=1e-3) and math.isclose(
            current_pos[1], next_wp[1], abs_tol=1e-3
        ):
            path.pop(0)

        time.sleep(CONTROL_INTERVAL)

    # ----- Termination -----
    drone.land()
    depth_stop_event.set()


if __name__ == "__main__":
    run()

