import math
import time
from djitellopy import Tello
from gird_map.grid_map_builder import build_grid_x_graph
from path_calculator.path_calculator import find_path

def move_toward(drone, target, pos, heading):
    """
    Rotate and move the drone from current pos/heading toward target waypoint.
    Returns updated position and heading.
    """
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    dist = math.hypot(dx, dy)

    if dist < 1e-3: # if the drone is already at (or extremely close to) target
        return pos, heading

    # 0° = +Y forward, +90° = +X right
    angle = math.degrees(math.atan2(dx, dy))
    turn = (angle - heading + 360) % 360
    if turn > 180:
        drone.rotate_counter_clockwise(int(360 - turn))
        heading = (heading - (360 - turn)) % 360
    else:
        drone.rotate_clockwise(int(turn))
        heading = (heading + turn) % 360
    dist = math.hypot(dx, dy)

    # move full distance to the waypoint
    drone.move_forward(int(dist * 100))  # convert m to cm

    # Correct XY update for 0°=forward(+Y)
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
    # 1) build mesh and plan
    G, nodes = build_grid_x_graph(width, length, spacing)
    path = find_path(G, nodes, start=(0.0, 0.0), goal=goal)
    print("Planned waypoints:", path)

    # 2) connect and takeoff
    drone = Tello()
    drone.connect()
    drone.takeoff()                       # ascend to default ~0.25m
    # optionally gain extra altitude
    drone.move_up(int(altitude * 100))    # ascend to desired altitude

    # 3) follow waypoints
    pos = (0.0, 0.0)
    heading = 0.0  # assume facing +X
    for wp in path:
        pos, heading = move_toward(drone, wp, pos, heading)
        time.sleep(0.5)  # small pause between moves

    # 4) land when done
    drone.land()