"""Wrapper around djitellopy.Tello for simplified control."""
from djitellopy import Tello


class DroneController:
    def __init__(self) -> None:
        self.tello = Tello()

    def connect(self) -> None:
        self.tello.connect()

    def start_stream(self) -> None:
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()

    def get_frame(self):
        return self.frame_read.frame

    def stop(self) -> None:
        self.tello.streamoff()
        self.tello.end()
