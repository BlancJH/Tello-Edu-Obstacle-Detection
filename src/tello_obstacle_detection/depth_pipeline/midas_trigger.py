import cv2
import torch
import numpy as np
import time
from dotenv import load_dotenv
from djitellopy import Tello

import os, sys

# Set submodule directory
midas_root = os.path.join(os.path.dirname(__file__), "../MiDaS-pipeline")
sys.path.insert(0, midas_root)

# Import midas as package
from midas.model_loader import load_model, default_models
from run import process, run

# Import env Virables
load_dotenv()
model_type = os.getenv("MODEL_NAME")

def init_depth_model(device, model_weights, model_type, optimize=False):
    """Load MiDaS model and preprocessing pipeline."""
    model, transform, net_w, net_h = load_model(device, model_weights, model_type, optimize)
    return model, transform, net_w, net_h

def capture_tello_frame(tello: Tello, timeout_s=5.0):
    """Return an RGB frame from Tello’s video stream."""
    start = time.time()
    while time.time() - start < timeout_s:
        frame = tello.get_frame_read().frame  # BGR
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.05)
    raise RuntimeError("Failed to get frame from Tello within timeout.")

def get_depth_array_from_frame(
    device, model, model_type, transform, net_w, net_h, image_rgb, optimize
):
    """Run MiDaS inference on a single RGB frame and return the depth array."""
    image = transform({"image": image_rgb / 255.0})["image"]
    with torch.no_grad():
        depth = process(
            device,
            model,
            model_type,
            image,
            (net_w, net_h),
            image_rgb.shape[1::-1],  # (width, height)
            optimize,
            True,  # use_camera flag
        )
    return depth

def capture_and_compute_depth(tello: Tello, device, model, model_type, transform, net_w, net_h, optimize):
    """Grab a frame from Tello and return (RGB frame, depth array)."""
    image_rgb = capture_tello_frame(tello)
    depth = get_depth_array_from_frame(
        device, model, model_type, transform, net_w, net_h, image_rgb, optimize
    )
    return image_rgb, depth
