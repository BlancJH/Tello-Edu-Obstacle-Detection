"""
Unit tests for path_calculator.find_path using a simple grid graph.
"""

import math
import pytest
import networkx as nx
from tello_obstacle_detection.gird_map.grid_map_builder import build_grid_x_graph
from tello_obstacle_detection.path_calculator.path_calculator import find_path


def test_straight_line_path():
    """
    On a 2×2 m grid with 1 m spacing, the shortest path from (0,0) to (1,2)
    should follow the straight column of corner‐to‐corner edges.
    """
    G, nodes = build_grid_x_graph(width=2, length=2, spacing=1)
    path = find_path(G, nodes, start=(0, 0), goal=(1, 2))
    assert path == [
        (0.0, 0),
        (0.5, 0.5),
        (1.0, 1),
        (1.0, 2)
    ]


def test_diagonal_path_goes_through_center():
    """
    On a 1×1 m grid with 1 m spacing, going from the SW corner (–0.5, 0)
    to the NE corner (0.5, 1) must pass through the cell center (0, 0.5).
    """
    G, nodes = build_grid_x_graph(width=1, length=1, spacing=1)
    start = (-0.5, 0.0)
    goal  = (0.5,  1.0)
    path = find_path(G, nodes, start=start, goal=goal)
    assert path == [
        (-0.5, 0.0),
        ( 0.0, 0.5),
        ( 0.5, 1.0)
    ]


def test_snapping_of_off_grid_points():
    """
    If start/goal aren’t exactly on nodes, they snap to the nearest node
    before pathfinding. On a 1×1 grid, (0.2,0.3) → (–0.5,0) and (0.2,0.8) → (–0.5,1).
    """
    G, nodes = build_grid_x_graph(width=1, length=1, spacing=1)
    # Both off by small amounts
    path = find_path(G, nodes, start=(0.2, 0.3), goal=(0.2, 0.8))
    assert path == [
        (0.0, 0.5),
        (0.5, 1)
    ]


def test_no_path_raises_when_disconnected():
    """
    If the graph has no connection between the snapped start and goal nodes,
    networkx.astar_path should raise NetworkXNoPath.
    """
    # Build a tiny graph with two isolated nodes
    G = nx.Graph()
    G.add_node(0, coord=(0.0, 0.0))
    G.add_node(1, coord=(1.0, 1.0))
    nodes = {(0.0, 0.0): 0, (1.0, 1.0): 1}

    with pytest.raises(nx.NetworkXNoPath):
        find_path(G, nodes, start=(0.0, 0.0), goal=(1.0, 1.0))
