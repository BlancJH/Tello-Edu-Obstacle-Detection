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