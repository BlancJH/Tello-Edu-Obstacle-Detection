import math
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np

# Direction and clearance logic live with the navigation helpers, not the
# MIDAS depth pipeline. Importing from the correct module keeps
# path_calculator lightweight and avoids unnecessary dependency on the depth
# fetching package.
from tello_obstacle_detection.drone_navigator.direction_navigator import (
    Direction,
    choose_safe_direction,
)

def find_path(
    G: nx.Graph,
    nodes: dict,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    depth_map: Optional[np.ndarray] = None,
    depth_threshold: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Find the shortest path from start → goal over graph G using A*, then optionally
    prune the facing node if the depth_map indicates the center is blocked.

    Parameters:
        G (networkx.Graph): nodes have 'coord' = (x, y).
        nodes (dict): maps (x, y) → node index in G.
        start (tuple): (x, y) start position.
        goal  (tuple): (x, y) goal position.
        depth_map: optional depth numpy array for immediate obstacle check.
        depth_threshold: threshold to decide clearance in choose_safe_direction.

    Returns:
        List[tuple]: ordered list of (x, y) waypoints from start to goal, possibly pruned.
    """
    def euclidean(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # Snap to nearest graph nodes
    start_idx = min(nodes.values(),
                    key=lambda i: euclidean(G.nodes[i]['coord'], start))
    goal_idx = min(nodes.values(),
                   key=lambda i: euclidean(G.nodes[i]['coord'], goal))

    # A* search
    path_idxs = nx.astar_path(
        G,
        start_idx,
        goal_idx,
        heuristic=lambda u, v: euclidean(G.nodes[u]['coord'], G.nodes[v]['coord']),
        weight='weight'
    )

    path = [G.nodes[i]['coord'] for i in path_idxs]

    # Prune facing node if depth_map indicates center blocked (and direction not CENTER)
    if depth_map is not None and len(path) >= 1:
        direction, status = choose_safe_direction(depth_map=depth_map, threshold=depth_threshold)
        if not status.get("center", False) and direction != Direction.CENTER:
            # Drop the first waypoint (facing node)
            pruned = path[1:]
            # log or annotate
            print(f"[NODE_PRUNER] Center blocked, direction={direction.value}. Pruning facing node {path[0]}. New path: {pruned}")
            path = pruned

    return path
