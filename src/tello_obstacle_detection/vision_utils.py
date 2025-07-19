"""Utility functions for image pre-processing."""
import cv2


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur(frame, kernel_size=(5, 5)):
    return cv2.GaussianBlur(frame, kernel_size, 0)
