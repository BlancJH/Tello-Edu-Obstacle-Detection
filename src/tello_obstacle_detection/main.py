import time
import torch
from djitellopy import Tello
from tello_obstacle_detection.depth_pipeline.midas_trigger import init_depth_model
from drone_navigator.drone_navigator import execute_simple_route
from tello_obstacle_detection.drone_navigator.drone_navigator import DroneKeepAlive


def make_depth_context():
    """Create and return the MiDaS depth_context dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, net_w, net_h = init_depth_model(
        device,
        "src/tello_obstacle_detection/weights/dpt_swin2_tiny_256.pt",
        model_type="dpt_swin2_tiny_256",
        optimize=False,
    )
    return {
        "device": device,
        "model": model,
        "model_type": "dpt_swin2_tiny_256",
        "transform": transform,
        "net_w": net_w,
        "net_h": net_h,
        "optimize": False,
    }


def main():
    WIDTH = 3.0
    LENGTH = 5.0
    SPACING = 0.5
    START = (0.0, 0.0)
    GOAL = (0.0, 4.0)
    ALTITUDE = 1.0
    drone = Tello()
    ka = None
    try:
        drone.connect()
        drone.streamon()
        time.sleep(2)

        # Load the depth model before taking off
        depth_ctx = make_depth_context()

        # Take off and immediately start the keep-alive thread
        drone.takeoff()
        ka = DroneKeepAlive(drone, interval=10)
        ka.start()
        while not ka.thread or not ka.thread.is_alive():
            time.sleep(0.1)
        time.sleep(3)

        # Proceed to waypoint calculation once model is loaded and keep-alive running
        execute_simple_route(
            drone,
            WIDTH,
            LENGTH,
            SPACING,
            START,
            GOAL,
            ALTITUDE,
            depth_ctx,
            depth_callback=None,
        )
    finally:
        if ka:
            ka.stop()
        try:
            drone.land()
            time.sleep(3)
            drone.streamoff()
        except Exception as e:
            print(f"[Cleanup] {e}")


if __name__ == "__main__":
    main()
