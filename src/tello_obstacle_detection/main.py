"""CLI entry point for running edge detection on the Tello video stream."""
import argparse
import cv2

from . import config
from .drone_controller import DroneController
from .vision_utils import to_gray, blur
from .edge_detector import canny_edges


def parse_args():
    parser = argparse.ArgumentParser(description="Tello obstacle detection demo")
    parser.add_argument('--low', type=int, default=config.CANNY_THRESHOLD_LOW,
                        help='Canny low threshold')
    parser.add_argument('--high', type=int, default=config.CANNY_THRESHOLD_HIGH,
                        help='Canny high threshold')
    return parser.parse_args()


def main():
    args = parse_args()

    drone = DroneController()
    drone.connect()
    drone.start_stream()

    try:
        while True:
            frame = drone.get_frame()
            if frame is None:
                continue

            gray = to_gray(frame)
            blurred = blur(gray)
            edges = canny_edges(blurred, args.low, args.high)

            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([frame, edges_bgr])
            cv2.imshow('Tello Feed | Edges', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
