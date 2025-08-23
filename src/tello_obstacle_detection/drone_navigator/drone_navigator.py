import math
import time
import torch
import threading
from typing import Callable
from tello_obstacle_detection.depth_pipeline.midas_trigger import capture_and_compute_depth, init_depth_model
from tello_obstacle_detection.depth_pipeline.post_processor import safe_direction, normalize_midas_depth
from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
from tello_obstacle_detection.path_calculator.path_calculator import find_path, path_re_planner
from djitellopy import Tello

class DroneKeepAlive:
    """Background thread to prevent Tello auto-landing"""
    def __init__(self, drone, interval=10):
        self.drone = drone
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _keep_alive_loop(self):
        while self.running:
            try:
                battery = self.drone.get_battery()
                print(f"[KeepAlive] Battery: {battery}%")
                time.sleep(self.interval)
            except Exception as e:
                print(f"[KeepAlive] Error: {e}")
                time.sleep(1)

def is_facing_target(heading, target, pos, tolerance_deg=10):
    """Check if drone is facing toward the target within tolerance"""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return True

    desired_angle = math.degrees(math.atan2(dx, dy)) % 360
    delta = (desired_angle - heading + 360) % 360
    if delta > 180:
        delta = 360 - delta
    return delta <= tolerance_deg

def rotate_toward(drone, target, pos, heading):
    """Rotate drone toward target direction"""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return heading
    
    desired_angle = math.degrees(math.atan2(dx, dy)) % 360
    turn = (desired_angle - heading + 360) % 360
    
    if turn > 180:
        delta = 360 - turn
        if delta > 5:  # Set minimum rotation angle
            drone.rotate_counter_clockwise(int(delta))
            heading = (heading - delta) % 360
    else:
        if turn > 5:  # Set minimum rotation angle
            drone.rotate_clockwise(int(turn))
            heading = (heading + turn) % 360
    
    return heading

def safe_depth_capture(drone, depth_context, max_retries=3):
    """Safe depth capture with retry logic and keep-alive pings."""
    for attempt in range(max_retries):
        try:
            print(f"[DepthCapture] Attempt {attempt + 1}/{max_retries}")
            time.sleep(0.1)  # small delay before frame capture
            
            rgb_frame, depth = capture_and_compute_depth(
                drone,
                depth_context["device"],
                depth_context["model"],
                depth_context["model_type"],
                depth_context["transform"],
                depth_context["net_w"],
                depth_context["net_h"],
                depth_context["optimize"],
            )
            print(f"[DepthCapture] Success - Depth shape: {depth.shape}")
            return rgb_frame, depth

        except Exception as e:
            print(f"[DepthCapture] Attempt {attempt + 1} failed: {e}")

            # Send keep-alive immediately after failure
            try:
                drone.get_battery()
            except Exception as ka_err:
                print(f"[DepthCapture] Keep-alive failed: {ka_err}")

            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print("[DepthCapture] All attempts failed")
                return None, None

def move_toward_with_depth(drone, target, pos, path, heading, depth_context, depth_callback):
    """Move toward target with depth capture"""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = math.hypot(dx, dy)

    if dist < 0.1:  # Consider arrived if within 10cm
        return pos, heading

    print(f"[Navigation] Moving to {target}, distance: {dist:.2f}m")

    # 1. Rotate toward target
    heading = rotate_toward(drone, target, pos, heading)
    
    # 2. Wait for stabilization after rotation
    time.sleep(0.5)
    
    # 3. Depth capture (before moving)
    rgb_frame, depth = safe_depth_capture(drone, depth_context)
    if depth_callback and depth is not None:
        depth_callback(depth, rgb_frame)

    # 4. Print out the safe direction analysed
    if depth is not None:
        depth_normalisation = normalize_midas_depth(depth)
        direction = safe_direction(depth_normalisation, near_thresh=0.35, min_free=0.65)
        print(f"[Obstacle Check] Safe direction: {direction}")
    else:
        print("[Obstacle Check] Depth is not captured")

    # 5. Adjust the path or proceed
    if direction == 'center':
        # Move (in cm units, max 500cm limit)
        move_cm = min(int(dist*100), 500)
        if move_cm <= 10:
            return pos, heading
        drone.move_forward(move_cm)
        time.sleep(move_cm/50.0)

        # Update new position
        new_position = (target[0], target[1])
        return new_position, heading, path, True
    else:
        path = path_re_planner(direction, path, pos)
        return pos, heading, path, False

def execute_simple_route(
    drone,
    width: float,
    length: float,
    spacing: float,
    start: tuple[float, float],
    goal: tuple[float, float],
    altitude: float = 1.0,
    depth_context: dict | None = None,
    depth_callback: Callable | None = None,
) -> None:
    """ 
    Execute a grid-planned waypoint route on an already-initialized airborne drone while delegating keep-alive, connection, and landing to the caller. 
        Args: 
            drone: Tello-compatible drone instance already connected, streaming, and airborne. 
            width: Width of the coverage area in meters used to build the grid. 
            length: Length of the coverage area in meters used to build the grid. 
            spacing: Grid spacing in meters between adjacent nodes along both axes. 
            goal: Target (x,y) position in meters in the same local frame as the planner; start is assumed to be (0.0,0.0). 
            altitude: Desired flight altitude in meters; reserved for downstream motion logic if utilized. 
            depth_callback: Optional callable invoked by move_toward_with_depth for depth outputs or telemetry; may be None. Returns: None. 
            Behavior: Builds a grid graph with build_grid_x_graph, computes a path with find_path, and iteratively calls move_toward_with_depth to advance toward each waypoint; 
            
    logs progress and continues past per-waypoint errors. Raises: Propagates exceptions thrown before the waypoint loop (e.g., planning failures); per-waypoint exceptions are caught, logged, and the loop continues.
    
    """

    G, nodes = build_grid_x_graph(width, length, spacing)
    path = find_path(G, nodes, start, goal)
    print(f"Planned waypoints: {path}")

    position = start
    heading = 0.0
    i = 1  # index through path

    # Loop until all waypoints completed
    while i < len(path):
        target = path[i]
        print(f"\n[Route] Waypoint {i+1}/{len(path)}: {target}")
        try:
            position, heading, path, proceed = move_toward_with_depth(
                drone, target, position, path, heading, depth_context, depth_callback
            )
        
            # Identify whether drone moved
            if proceed == True:
                i += 1
            else:
                path = path

        except Exception as e:
            print(f"[Route] Error at waypoint {target}: {e}")

    print("\n[Route] All waypoints completed!")
