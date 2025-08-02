import math
import time

FACING_THRESHOLD_DEG = 10

def is_facing_target(current_heading, src_pos, target_pos, threshold_deg=FACING_THRESHOLD_DEG):
    dx = target_pos[0] - src_pos[0]
    dy = target_pos[1] - src_pos[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return True
    desired_angle = math.degrees(math.atan2(dx, dy)) % 360
    diff = (desired_angle - current_heading + 360) % 360
    if diff > 180:
        diff = 360 - diff
    return diff <= threshold_deg

def move_toward(drone, target, pos, heading):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return pos, heading
    angle = math.degrees(math.atan2(dx, dy)) % 360
    turn = (angle - heading + 360) % 360
    if turn > 180:
        amount = int(360 - turn)
        drone.rotate_counter_clockwise(amount)
        heading = (heading - amount) % 360
    else:
        amount = int(turn)
        drone.rotate_clockwise(amount)
        heading = (heading + amount) % 360
    drone.move_forward(int(dist * 100))
    new_x = pos[0] + dist * math.sin(math.radians(angle))
    new_y = pos[1] + dist * math.cos(math.radians(angle))
    return (new_x, new_y), heading

def execute_simple_route(drone, width, length, spacing, goal, snapshot_hook, altitude=1.0):
    from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
    from tello_obstacle_detection.path_calculator.path_calculator import find_path
    G, nodes = build_grid_x_graph(width, length, spacing)
    path = find_path(G, nodes, start=(0.0, 0.0), goal=goal)
    print("Planned waypoints:", path)
    drone.takeoff()
    time.sleep(1)
    drone.move_up(int(altitude * 100))
    pos = (0.0, 0.0)
    heading = 0.0
    for wp in path:
        if is_facing_target(heading, pos, wp):
            print(f"[NAV] Facing waypoint {wp} before moving, triggering snapshot.")
            snapshot_hook()
        pos, heading = move_toward(drone, wp, pos, heading)
        if is_facing_target(heading, pos, wp):
            print(f"[NAV] Facing waypoint {wp} after move, triggering snapshot.")
            snapshot_hook()
        time.sleep(0.5)
    print("[NAV] Route complete. Landing.")
    drone.land()
