"""Edge detection helpers."""
import cv2


def canny_edges(gray_frame, low_threshold, high_threshold):
    return cv2.Canny(gray_frame, low_threshold, high_threshold)
