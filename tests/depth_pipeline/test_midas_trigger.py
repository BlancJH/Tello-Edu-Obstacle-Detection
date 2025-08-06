import os
import sys
import time
import pytest
import numpy as np
import torch

# Import the symbols from your module; adjust the import path if the file is named differently
from pathlib import Path

# Assuming your module is in src/B/example/example.py, and its filename is e.g. depth_tello.py
# e.g., from example.depth_tello import (
#     init_depth_model,
#     capture_tello_frame,
#     get_depth_array_from_frame,
#     capture_and_compute_depth,
# )
# Replace below with the actual module path/name:
import tello_obstacle_detection.depth_pipeline.midas_trigger as dt

class DummyFrame:
    def __init__(self, array):
        self.frame = array  # BGR frame

class DummyFrameReader:
    def __init__(self, frame):
        self.frame = frame
        self.counter = 0

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, val):
        self._frame = val

class DummyTello:
    def __init__(self, rgb_array):
        # Simulate get_frame_read().frame returns BGR image
        self._frame = cvt_rgb_to_bgr(rgb_array)
        self._frame_read = type("FR", (), {"frame": self._frame})

    def get_frame_read(self):
        # Return an object with .frame property
        return self._frame_read

    def connect(self):
        pass

    def streamon(self):
        pass

def cvt_rgb_to_bgr(rgb):
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

@pytest.fixture
def dummy_rgb_image():
    # Create a small synthetic RGB image (e.g., 64x64)
    h, w = 64, 64
    img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return img

@pytest.fixture
def fake_model_and_transform(monkeypatch):
    # Fake load_model to return dummy objects
    def fake_load_model(device, model_weights, model_type, optimize):
        # Return: model, transform, net_w, net_h
        def fake_transform(d):
            # expects dict with "image": array
            return {"image": d["image"]}  # identity
        model = object()
        transform = fake_transform
        net_w, net_h = 128, 128
        return model, transform, net_w, net_h

    monkeypatch.setattr(dt, "load_model", fake_load_model)
    return fake_load_model  # not used directly

@pytest.fixture
def fake_process(monkeypatch):
    # Fake process to return a deterministic depth map
    def fake_process(device, model, model_type, image, net_dims, orig_wh, optimize, use_camera):
        # Return a depth array same shape as input width/height (simulate)
        # image is a tensor-like; for simplicity return a constant array
        h, w = orig_wh[1], orig_wh[0]
        return np.full((h, w), 2.5, dtype=np.float32)

    monkeypatch.setattr(dt, "process", fake_process)
    return fake_process

def test_init_depth_model_calls_load_model(monkeypatch):
    # Ensure init_depth_model returns what load_model would give
    called = {}

    def spy_load_model(device, model_weights, model_type, optimize):
        called['args'] = (device, model_weights, model_type, optimize)
        fake_transform = lambda d: {"image": d["image"]}
        return "model_obj", fake_transform, 32, 32

    monkeypatch.setattr(dt, "load_model", spy_load_model)
    device = torch.device("cpu")
    model, transform, net_w, net_h = dt.init_depth_model(device, "weights.pt", "my_model", optimize=True)

    assert model == "model_obj"
    assert callable(transform)
    assert net_w == 32 and net_h == 32
    assert called['args'] == (device, "weights.pt", "my_model", True)

def test_get_depth_array_from_frame_returns_expected(fake_model_and_transform, fake_process, dummy_rgb_image):
    device = torch.device("cpu")
    # Prepare fake model/loading
    model, transform, net_w, net_h = dt.init_depth_model(device, "unused", "type", optimize=False)
    # Call and verify output shape/content
    depth = dt.get_depth_array_from_frame(
        device,
        model,
        "type",
        transform,
        net_w,
        net_h,
        dummy_rgb_image,
        optimize=False,
    )
    # Should match fake_process output which is filled with 2.5
    h, w = dummy_rgb_image.shape[0], dummy_rgb_image.shape[1]
    assert isinstance(depth, np.ndarray)
    assert depth.shape == (h, w)
    assert np.allclose(depth, 2.5)

def test_capture_tello_frame_success(monkeypatch, dummy_rgb_image):
    import cv2

    # Create dummy Tello whose get_frame_read().frame returns a BGR image
    class FakeFrameRead:
        def __init__(self, frame):
            self.frame = frame

    class FakeTello:
        def __init__(self, frame):
            self._fr = FakeFrameRead(frame)

        def get_frame_read(self):
            return self._fr

    # Prepare a correct BGR image from random RGB
    rgb = dummy_rgb_image
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tello = FakeTello(bgr)

    # Should return RGB conversion
    returned = dt.capture_tello_frame(tello, timeout_s=1.0)
    assert returned.shape == rgb.shape
    assert np.array_equal(returned, rgb)

def test_capture_and_compute_depth_integration(monkeypatch, fake_model_and_transform, fake_process, dummy_rgb_image):
    import cv2
    # Fake Tello
    class FakeFrameRead:
        def __init__(self, frame):
            self.frame = frame

    class FakeTello:
        def __init__(self, rgb):
            self._bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._fr = FakeFrameRead(self._bgr)

        def get_frame_read(self):
            return self._fr

    device = torch.device("cpu")
    model, transform, net_w, net_h = dt.init_depth_model(device, "unused", "type", optimize=False)
    tello = FakeTello(dummy_rgb_image)

    rgb_frame, depth_map = dt.capture_and_compute_depth(
        tello, device, model, "type", transform, net_w, net_h, optimize=False
    )
    # Validate shapes
    assert rgb_frame.shape == dummy_rgb_image.shape
    h, w = dummy_rgb_image.shape[0], dummy_rgb_image.shape[1]
    assert depth_map.shape == (h, w)
    assert np.allclose(depth_map, 2.5)
