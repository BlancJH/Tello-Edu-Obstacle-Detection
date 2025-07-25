from drone_navigator.drone_navigator import execute_simple_route

if __name__ == "__main__":
    # user parameters
    WIDTH = 3.0       # meters east-west span
    LENGTH = 5.0      # meters north-south span
    SPACING = 0.5     # meters between grid nodes
    GOAL = (0.0, 3.0) # target coordinate in meters
    ALTITUDE = 1.0    # meters above ground

    execute_simple_route(WIDTH, LENGTH, SPACING, GOAL, ALTITUDE)
