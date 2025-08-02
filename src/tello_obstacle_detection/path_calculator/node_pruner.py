from typing import List, Tuple
import numpy as np

from drone_navigator import Direction  # adjust import path if needed
from midas_pipeline.output_pipeline import choose_safe_direction  # import the function you implemented

def prune_facing_node_if_center_blocked(
    path: List[Tuple[float, float]],
    current_pos: Tuple[float, float],
    heading: float,
    depth_map: np.ndarray,
    threshold: float,
) -> List[Tuple[float, float]]:
    """
    If the next waypoint (facing node) is coming and the center sample is blocked,
    prune that node from the path so the navigator can replan or skip it.

    Rules:
      - Evaluate safe direction from depth_map.
      - If center is blocked and the chosen direction is not CENTER, assume the facing node is unsafe and remove it.
      - Otherwise, leave path intact.

    Args:
        path: list of waypoint (x,y) tuples; first element is next/facing node.
        current_pos: current (x,y) position (unused here but may be for context).
        heading: current heading in degrees (unused here but could be used to decide “facing” more robustly).
        depth_map: 2D numpy depth array.
        threshold: depth threshold for clearance.

    Returns:
        Possibly shortened path with the facing node removed if blocked.
    """
    if not path:
        return path  # nothing to prune

    chosen_direction, status = choose_safe_direction(
        depth_map=depth_map,
        threshold=threshold,
    )

    # If center is blocked (i.e., status["center"] is False) and result is not CENTER,
    # that implies we're facing something unsafe ahead: prune the first waypoint.
    if not status.get("center", False) and chosen_direction != Direction.CENTER:
        pruned = path[1:]
        print(f"[PRUNER] Center blocked, direction {chosen_direction.value}. Pruning facing node {path[0]}. New path: {pruned}")
        return pruned

    return path