import math
import time
import torch
import threading
from typing import Callable
from tello_obstacle_detection.depth_pipeline.midas_trigger import capture_and_compute_depth, init_depth_model
from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
from tello_obstacle_detection.path_calculator.path_calculator import find_path
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
    """Safe depth capture with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"[DepthCapture] Attempt {attempt + 1}/{max_retries}")
            
            # Wait briefly before frame capture
            time.sleep(0.1)
            
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
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait before retry
            else:
                print("[DepthCapture] All attempts failed")
                return None, None

def move_toward_with_depth(drone, target, pos, heading, depth_context, depth_callback):
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
    
    # 4. Move (in cm units, max 500cm limit)
    move_cm = min(int(dist*100), 500)
    if move_cm <= 10:
        return pos, heading
    drone.move_forward(move_cm)
    time.sleep(move_cm/50.0)
    
    # 5. Calculate new position
    move_m = move_cm/100.0
    ang = math.radians(heading)
    new_x = pos[0] + move_m*math.sin(ang)
    new_y = pos[1] + move_m*math.cos(ang)
    return (new_x, new_y), heading

def execute_simple_route(
    drone,
    width: float,
    length: float,
    spacing: float,
    goal: tuple[float,float],
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
    G,nodes=build_grid_x_graph(width,length,spacing)
    path=find_path(G,nodes,start=(0.0,0.0),goal=goal)
    print(f"Planned waypoints: {path}")
    pos=(0.0,0.0);heading=0.0
    for i,waypoint in enumerate(path):
        print(f"\n[Route] Waypoint {i+1}/{len(path)}: {waypoint}")
        try:
            pos,heading=move_toward_with_depth(drone,waypoint,pos,heading,depth_context,depth_callback)
            print(f"[Route] Reached waypoint {i+1}, current pos: {pos}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[Route] Error at waypoint {waypoint}: {e}")
            continue
    print("\n[Route] All waypoints completed!")
