import math
import time
import torch
from tello_obstacle_detection.depth_pipeline.midas_trigger import capture_and_compute_depth, init_depth_model
from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
from tello_obstacle_detection.path_calculator.path_calculator import find_path
from djitellopy import Tello

def is_facing_target(heading, target, pos, tolerance_deg=10):
    """
    Returns True if the drone's heading is within `tolerance_deg` degrees of
    the direction toward target. 0° = +Y forward, +90° = +X right (same as your system).
    """
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return True  # already at target

    desired_angle = math.degrees(math.atan2(dx, dy)) % 360
    delta = (desired_angle - heading + 360) % 360
    if delta > 180:
        delta = 360 - delta
    return delta <= tolerance_deg

def rotate_toward(drone, target, pos, heading):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return heading  # no change
    desired_angle = math.degrees(math.atan2(dx, dy)) % 360
    turn = (desired_angle - heading + 360) % 360
    if turn > 180:
        delta = 360 - turn
        drone.rotate_counter_clockwise(int(delta))
        heading = (heading - delta) % 360
    else:
        drone.rotate_clockwise(int(turn))
        heading = (heading + turn) % 360
    return heading

def move_toward(
    drone,
    target,
    pos,
    heading,
    device,
    model,
    model_type,
    transform,
    net_w,
    net_h,
    optimize,
    depth_callback=None,
    alignment_tolerance=10,
):
    """
    Rotate and move the drone from current pos/heading toward target waypoint.
    If facing the target within `alignment_tolerance` degrees, capture a depth map
    via MiDaS before proceeding. Returns updated position and heading.
    depth_callback: optional callable(depth_array, rgb_frame) for post-processing.
    """
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = math.hypot(dx, dy)
    dist = math.hypot(dx, dy)

    if dist < 1e-3:
        return pos, heading  # already there

    # Compute desired angle
    angle = math.degrees(math.atan2(dx, dy)) % 360

    # If already roughly facing the target, trigger depth capture once
    if is_facing_target(heading, target, pos, tolerance_deg=alignment_tolerance):
        try:
            rgb_frame, depth = capture_and_compute_depth(
                drone,  # assuming tello-like object is passed here
                device,
                model,
                model_type,
                transform,
                net_w,
                net_h,
                optimize,
            )
            if depth_callback:
                depth_callback(depth, rgb_frame)
        except Exception as e:
            # log or handle depth capture failure but don't block movement
            print(f"[move_toward] depth capture failed: {e}")

    # Rotate toward target
    turn = (angle - heading + 360) % 360
    if turn > 180:
        delta = 360 - turn
        if delta > 0:
            drone.rotate_counter_clockwise(int(delta))
            heading = (heading - delta) % 360
    else:
        if turn > 0:
            drone.rotate_clockwise(int(turn))
            heading = (heading + turn) % 360

    drone.move_forward(int(dist * 100))

    new_x = pos[0] + dist * math.sin(math.radians(angle))
    new_y = pos[1] + dist * math.cos(math.radians(angle))
    return (new_x, new_y), heading

def execute_simple_route(width, length, spacing, goal, altitude=1.0):
    """
    Plan a path on a fixed-spacing grid and fly the Tello along it.
    No obstacle avoidance.

    width, length (m): arena dimensions on ground plane
    spacing (m): grid node gap
    goal (x,y) in meters: destination coordinate
    altitude (m): takeoff altitude to maintain
    """
    # 1. build mesh and plan
    G, nodes = build_grid_x_graph(width, length, spacing)
    path = find_path(G, nodes, start=(0.0, 0.0), goal=goal)
    print("Planned waypoints:", path)

    # 2. connect and takeoff
    drone = Tello()
    drone.connect()
    drone.streamon()
    drone.takeoff() # ascend to default ~0.25m

    # 3. init depth model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, net_w, net_h = init_depth_model(
        device,
        "src/tello_obstacle_detection/weights/dpt_swin2_large_384.pt",
        model_type="dpt_swin2_large_384",
        optimize=False,
    )

    depth_context = {
        "device": device,
        "model": model,
        "model_type": "dpt_swin2_large_384",
        "transform": transform,
        "net_w": net_w,
        "net_h": net_h,
        "optimize": False,
    }
    # 3. follow waypoints
    pos = (0.0, 0.0)
    heading = 0.0  # assume facing +X
    for wp in path:
        pos, heading = move_toward(
        drone,
        wp,
        pos,
        heading,
        depth_context["device"],
        depth_context["model"],
        depth_context["model_type"],
        depth_context["transform"],
        depth_context["net_w"],
        depth_context["net_h"],
        depth_context["optimize"],
        )
    time.sleep(0.5)
    
    # 4. land when done
    drone.land()