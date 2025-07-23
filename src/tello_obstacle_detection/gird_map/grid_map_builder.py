import math
import networkx as nx

def build_grid_x_graph(W, H, Nx, Ny):
    """
    Builds a navigation graph over a rectangular area subdivided into a grid,  
    with each cell containing an “X” (diagonals) for possible flight paths.

    Parameters:
        W (float): Width of the rectangle in meters
        H (float): Height of the rectangle in meters
        Nx (int): Number of columns
        Ny (int): Number of rows

    Returns:
        G (networkx.Graph): An undirected graph where:
            - Each node represents a key waypoint (grid corner or cell center).
            - Each edge connects a cell center to each of its four corners.
            - Edge weights are the Euclidean distances (in the same units as W/H).
        nodes (dict): Mapping from coordinate tuple → node index.
            - Key `'coord'` in node attributes is the (x, y) position of that node.
    """
    # Compute cell dimensions
    dx = W / Nx # width of each cell
    dy = H / Ny # height of each cell

    # 1. Create all nodes: grid corners + cell centers
    nodes = {} # maps (x, y) coordinate → unique integer node index
    node_idx = 0 # incremental ID

    # Add every grid corner
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            coord = (i * dx, j * dy) # coord is the (x, y) point in the plane
            nodes[coord] = node_idx
            node_idx += 1

    # Add each cell’s center (intersection of the “X”)
    for i in range(Nx):
        for j in range(Ny):
            coord = (i * dx + dx / 2, j * dy + dy / 2)
            nodes[coord] = node_idx
            node_idx += 1

    # 2. Build the graph and connect centers to corners
    G = nx.Graph()
    # Add nodes with their coordinate attribute
    for coord, idx in nodes.items():
        G.add_node(idx, coord=coord)

    # For each cell, connect its center node to each of the four corners
    for i in range(Nx):
        for j in range(Ny):
            # List the four corner coordinates of cell (i, j)
            corners = [
                (i * dx, j * dy),
                ((i + 1) * dx, j * dy),
                ((i + 1) * dx, (j + 1) * dy),
                (i * dx, (j + 1) * dy),
            ]
            center = (i * dx + dx / 2, j * dy + dy / 2)
            center_idx = nodes[center]
            # Connect center to each corner, weighting by Euclidean distance
            for corner in corners:
                corner_idx = nodes[corner]
                dist = math.hypot(corner[0] - center[0],
                                  corner[1] - center[1])
                G.add_edge(corner_idx, center_idx, weight=dist)

    return G, nodes
