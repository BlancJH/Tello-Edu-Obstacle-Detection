import math
import pytest
import networkx as nx
from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph

def expected_counts(width, length, spacing):
    Nx = int(round(width  / spacing))
    Ny = int(round(length / spacing))
    nodes = (Nx + 1) * (Ny + 1) + Nx * Ny
    # total edges = center–corner + corner–corner (horiz + vert)
    center_corner = 4 * Nx * Ny
    corner_corner = Nx * (Ny + 1) + (Nx + 1) * Ny
    edges = center_corner + corner_corner
    return Nx, Ny, nodes, edges

@pytest.mark.parametrize("width,length,spacing", [
    (1,   1,   1),     # minimal
    (2,   2,   1),     # square 2×2
    (4,   2,   1),     # non‑square rect
    (3.5, 2.5, 0.5),   # fractional dims
    (10,  1,   1),     # extreme aspect ratio
    (5,   2,   0.5),   # single row
    (2,   5,   0.5),   # single column
])
def test_node_and_edge_counts(width, length, spacing):
    G, nodes = build_grid_x_graph(width, length, spacing)
    Nx, Ny, exp_nodes, exp_edges = expected_counts(width, length, spacing)

    # node‑count sanity
    assert len(nodes) == exp_nodes

    # now matches both center–corner and corner–corner edges
    assert G.number_of_edges() == exp_edges

    # Spot‑check one center-to-corner weight
    actual_width = Nx * spacing
    corner = (-actual_width/2, 0.0)
    center = (corner[0] + spacing/2, spacing/2)

    assert corner in nodes
    assert center in nodes

    w = G[nodes[center]][nodes[corner]]['weight']
    expected_w = math.hypot(spacing/2, spacing/2)
    assert pytest.approx(expected_w, rel=1e-6) == w

def test_invalid_inputs():
    with pytest.raises(ValueError):
        build_grid_x_graph(5, 5, 0)       # zero spacing
    with pytest.raises(ValueError):
        build_grid_x_graph(0, 5, 0.5)     # zero width
    with pytest.raises(ValueError):
        build_grid_x_graph(5, 0, 0.5)     # zero length
    with pytest.raises(ValueError):
        build_grid_x_graph(0.1, 0.1, 1.0) # spacing > dimensions

def test_type_errors():
    with pytest.raises(TypeError):
        build_grid_x_graph("5", 3,   0.5)
    with pytest.raises(TypeError):
        build_grid_x_graph(5,   "3", 0.5)
    with pytest.raises(TypeError):
        build_grid_x_graph(5,   3,   "0.5")

def test_large_mesh_performance():
    width, length, spacing = 100.0, 100.0, 1.0
    G, nodes = build_grid_x_graph(width, length, spacing)
    Nx, Ny, exp_nodes, exp_edges = expected_counts(width, length, spacing)

    assert len(nodes) == exp_nodes
    assert G.number_of_edges() == exp_edges
