import numpy as np
from enum import Enum
from typing import Tuple, Dict, Optional

class Direction(Enum):
    CENTER = "center"
    UP = "top"
    RIGHT = "right"
    LEFT = "left"
    DOWN = "bottom"
    BLOCK = "block"

def choose_safe_direction(
    depth_map: np.ndarray,
    threshold: float,
    sample_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Direction, Dict[str, bool]]:
    """
    Decide safe direction with priority:
      1. Center if clear.
      2. If only one direction is clear, take it.
      3. If only center is blocked (i.e., left, right, top, bottom all clear) -> UP.
      4. If top, center, bottom are blocked -> RIGHT.
      5. Fallback priority: Center > Up > Right > Left > Bottom.
    Returns (chosen_direction, per-point status).
    """
    h, w = depth_map.shape[:2]
    if sample_offsets is None:
        sample_offsets = {
            "left": (-0.3, 0.0),
            "center": (0.0, 0.0),
            "right": (0.3, 0.0),
            "top": (0.0, -0.3),
            "bottom": (0.0, 0.3),
        }

    def to_pixel(offset):
        ox, oy = offset
        px = int(np.clip((ox * 0.5 + 0.5) * (w - 1), 0, w - 1))
        py = int(np.clip((oy * 0.5 + 0.5) * (h - 1), 0, h - 1))
        return px, py

    status: Dict[str, bool] = {}
    for name, offset in sample_offsets.items():
        x, y = to_pixel(offset)
        depth_value = depth_map[y, x]
        status[name] = depth_value >= threshold

    # 1. Center if clear
    if status.get("center", False):
        return Direction.CENTER, status

    # 2. Only one clear direction
    clear = [d for d, ok in status.items() if ok]
    if len(clear) == 1:
        sole = clear[0]
        # map name to Direction
        mapping = {
            "center": Direction.CENTER,
            "top": Direction.UP,
            "right": Direction.RIGHT,
            "left": Direction.LEFT,
            "bottom": Direction.DOWN,
        }
        return mapping.get(sole, Direction.BLOCK), status

    # 3. Only center blocked => UP
    others = ["left", "right", "top", "bottom"]
    if (not status.get("center", False)) and all(status.get(o, False) for o in others):
        return Direction.UP, status

    # 4. Top, center, bottom blocked => RIGHT
    if (not status.get("top", False)) and (not status.get("center", False)) and (not status.get("bottom", False)):
        return Direction.RIGHT, status

    # 5. Fallback by priority: Center > Up > Right > Left > Bottom
    if status.get("center", False):
        return Direction.CENTER, status
    if status.get("top", False):
        return Direction.UP, status
    if status.get("right", False):
        return Direction.RIGHT, status
    if status.get("left", False):
        return Direction.LEFT, status
    if status.get("bottom", False):
        return Direction.DOWN, status

    return Direction.BLOCK, status
