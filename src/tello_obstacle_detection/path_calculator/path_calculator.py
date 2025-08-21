import math
import networkx as nx

def find_path(G, nodes, start, goal):
    """
    Find the shortest path from start → goal over graph G using A*.

    Parameters:
        G (networkx.Graph): nodes have 'coord' = (x, y).
        nodes (dict): maps (x, y) → node index in G.
        start (tuple): (x, y) start position, e.g. (0, 0).
        goal  (tuple): (x, y) goal position within the flight area.

    Returns:
        List[tuple]: ordered list of (x, y) waypoints from start to goal.
    """
    def euclidean(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # Snap to graph nodes
    start_idx = min(nodes.values(),
                    key=lambda i: euclidean(G.nodes[i]['coord'], start))
    goal_idx  = min(nodes.values(),
                    key=lambda i: euclidean(G.nodes[i]['coord'], goal))

    # A* search
    path_idxs = nx.astar_path(
        G,
        start_idx,
        goal_idx,
        heuristic=lambda u, v: euclidean(G.nodes[u]['coord'], G.nodes[v]['coord']),
        weight='weight'
    )

    # Map back to coordinates
    return [G.nodes[i]['coord'] for i in path_idxs]

def path_re_planner(direction: str, path: list[tuple[float, float]], position: tuple, shift: float = 0.5) -> list[tuple[float, float]]:
    """Adjust the nodes according to the safe direction detected."""
    if not path:
        print("[Route] No Path to re-plan.")

    goal = path[-1]

    if direction == "right":
        # Push all the nodes (x + 0.5, y)
        return [(x + 0.5, y) for (x, y) in path[:-1]] + [goal]

    elif direction == 'left':
        # Push all the nodes (x - 0.5, y)
        return [(x - 0.5, y) for (x, y) in path[:-1]] + [goal]

    elif direction == 'none':
        # Add one node to the side and push all nodes except the goal to the following side.
        detour_node = (position[0] + shift, position[1])
        shifted_node = [(x + shift, y) for (x, y) in path[:-1]]
        return [detour_node] + shifted_node + [goal]
    
    return path