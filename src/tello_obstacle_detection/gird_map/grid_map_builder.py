import math
import networkx as nx

def build_grid_x_graph(W, L, Nx, Ny):
    """
    Build a navigation graph over a flat, rectangular area on the ground.

    The drone takes off to a fixed altitude and then navigates only
    in the horizontal plane, where:

        x ∈ [ -width/2, +width/2 ] (meters east–west)
        y ∈ [0, length ] (meters north–south)

    The rectangle is subdivided into Nx columns (along x) and Ny rows
    (along y), and each cell gets an “X” (diagonals from its center to corners).

    Parameters:
        width (float): total span along the x‑axis (meters).
        length (float): total span along the y‑axis (meters).
        Nx (int): how many columns to split the width into.
        Ny (int): how many rows to split the length into.

    Returns:
        G (networkx.Graph):
            Each node has attribute 'coord' = (x, y) on the ground.
        nodes (dict of (x, y) → int):
            Maps each waypoint coordinate to its node index in G.
    """
    dx = W / Nx # cell size in x (meters)
    dy = L / Ny # cell size in y (meters)

    # 1. Create all nodes: grid corners + cell centers
    nodes = {}
    idx = 0

    # Grid corners
    for i in range(Nx + 1):
        x = -W/2 + i * dx
        for j in range(Ny + 1):
            y = j * dy
            coord = (x, y) # coord is the (x, y) point in space
            nodes[coord] = idx
            idx += 1

    # Cell centers (where diagonals cross)
    for i in range(Nx):
        cx = -W/2 + (i + 0.5) * dx
        for j in range(Ny):
            cy = (j + 0.5) * dy
            coord = (cx, cy)
            nodes[coord] = idx
            idx += 1

    # 2. Build the graph and wire up center→corner edges
    G = nx.Graph()
    for coord, node_id in nodes.items():
        G.add_node(node_id, coord=coord)

    # For each cell, connect its center to each of its four corners
    for i in range(Nx):
        x0 = -W/2 + i * dx
        x1 = x0 + dx
        for j in range(Ny):
            y0 = j * dy
            y1 = y0 + dy

            corners = [
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            ]
            center = (-W/2 + (i + 0.5) * dx, (j + 0.5) * dy)
            c_idx = nodes[center]

            for corner in corners:
                corner_idx = nodes[corner]
                # Euclidean distance in same units as W/H
                dist = math.hypot(corner[0] - center[0],
                                  corner[1] - center[1])
                G.add_edge(c_idx, corner_idx, weight=dist)

    return G, nodes