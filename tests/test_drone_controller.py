from tello_obstacle_detection.drone_controller import DroneController


def test_controller_init():
    controller = DroneController()
    assert controller.tello is not None
