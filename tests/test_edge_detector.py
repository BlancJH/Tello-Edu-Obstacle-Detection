import numpy as np
from tello_obstacle_detection.edge_detector import canny_edges


def test_canny_edges_shape():
    img = np.zeros((10, 10), dtype=np.uint8)
    edges = canny_edges(img, 50, 150)
    assert edges.shape == img.shape
