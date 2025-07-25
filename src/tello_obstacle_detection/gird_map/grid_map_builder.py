import math
import networkx as nx

def build_grid_x_graph(width, length, spacing=0.5):
    """
    Build a navigation graph over a flat, rectangular area on the ground
    with all grid nodes exactly `spacing` meters apart (e.g. 0.5 m).
    The drone flies at fixed altitude and navigates in the horizontal (x, y) plane.

    Parameters:
        width (float): east–west span in meters.
        length (float): north–south span in meters.
        spacing (float): distance between adjacent grid nodes in meters.

    Returns:
        G (networkx.Graph): graph; each node has attribute 'coord' = (x, y).
        nodes (dict): mapping from (x, y) to node index in G.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if width <= 0 or length <= 0:
        raise ValueError("width and length must be positive")

    Nx = int(round(width / spacing))
    Ny = int(round(length / spacing))
    if Nx < 1 or Ny < 1:
        raise ValueError("width and length must each be at least spacing")

    actual_width = Nx * spacing
    actual_length = Ny * spacing
    dx = spacing
    dy = spacing

    nodes = {}
    idx = 0
    # 1. Grid corners
    for i in range(Nx + 1):
        x = -actual_width / 2 + i * dx
        for j in range(Ny + 1):
            y = j * dy
            nodes[(x, y)] = idx
            idx += 1

    # 2. Cell centers (where diagonals cross)
    for i in range(Nx):
        cx = -actual_width / 2 + (i + 0.5) * dx
        for j in range(Ny):
            cy = (j + 0.5) * dy
            nodes[(cx, cy)] = idx
            idx += 1

    # 3. Build graph and connect each center to its four corners
    G = nx.Graph()
    for coord, node_id in nodes.items():
        G.add_node(node_id, coord=coord)

    for i in range(Nx):
        x0 = -actual_width / 2 + i * dx
        x1 = x0 + dx
        for j in range(Ny):
            y0 = j * dy
            y1 = y0 + dy
            center = (-actual_width / 2 + (i + 0.5) * dx, (j + 0.5) * dy)
            c_idx = nodes[center]
            for corner in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                corner_idx = nodes[corner]
                dist = math.hypot(corner[0] - center[0], corner[1] - center[1])
                G.add_edge(c_idx, corner_idx, weight=dist)

    # 4. Connect corner-to-corner horizontally and vertically
    for coord, nid in list(nodes.items()):
        x, y = coord
        # only if this is a corner (multiples of spacing)
        if abs((x + actual_width/2) % spacing) < 1e-6 and abs(y % spacing) < 1e-6:
            # neighbor offsets east and north
            for dx_off, dy_off in [(spacing, 0), (0, spacing)]:
                nbr = (x + dx_off, y + dy_off)
                if nbr in nodes:
                    nid2 = nodes[nbr]
                    G.add_edge(nid, nid2, weight=math.hypot(dx_off, dy_off))
                    
    return G, nodes