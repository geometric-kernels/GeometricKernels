from math import sqrt as sr
from pathlib import Path

import numpy as np

TEST_MESH_PATH = str(Path(__file__).parent.resolve() / "teddy.obj")

TEST_GRAPH_ADJACENCY = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
).astype(np.float64)


TEST_GRAPH_LAPLACIAN = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0],
        [-1, 4, -1, -1, -1, 0, 0],
        [0, -1, 2, 0, 0, -1, 0],
        [0, -1, 0, 2, -1, 0, 0],
        [0, -1, 0, -1, 2, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
).astype(
    np.float64
)  # corresponds to TEST_GRAPH_ADJACENCY, unnormalized Laplacian

TEST_GRAPH_LAPLACIAN_NORMALIZED = np.array(
    [
        [1, -0.5, 0, 0, 0, 0, 0],  # noqa: E241
        [-0.5, 1, -1 / sr(2) / 2, -1 / sr(2) / 2, -1 / sr(2) / 2, 0, 0],  # noqa: E241
        [0, -1 / sr(2) / 2, 1, 0, 0, -1 / sr(2), 0],  # noqa: E241
        [0, -1 / sr(2) / 2, 0, 1, -0.5, 0, 0],  # noqa: E241
        [0, -1 / sr(2) / 2, 0, -0.5, 1, 0, 0],  # noqa: E241
        [0, 0, -1 / sr(2), 0, 0, 1, 0],  # noqa: E241
        [0, 0, 0, 0, 0, 0, 0],  # noqa: E241
    ]
).astype(
    np.float64
)  # corresponds to TEST_GRAPH_ADJACENCY, normalized Laplacian

TEST_GRAPH_EDGES_NUM_NODES = 5

TEST_GRAPH_EDGES_ADJACENCY = np.array(
    [
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ]
).astype(np.int32)

TEST_GRAPH_EDGES_UP_LAPLACIAN = np.array(
    [
        [1, -1, 0, 1, 0, 0, 0],
        [-1, 1, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
).astype(np.float64)

TEST_GRAPH_EDGES_DOWN_LAPLACIAN = np.array(
    [
        [2, 1, 1, -1, -1, 0, 0],
        [1, 2, 1, 1, 0, -1, 0],
        [1, 1, 2, 0, 0, 0, 1],
        [-1, 1, 0, 2, 1, -1, 0],
        [-1, 0, 0, 1, 2, 1, -1],
        [0, -1, 0, -1, 1, 2, -1],
        [0, 0, 1, 0, -1, -1, 2],
    ]
).astype(np.float64)

TEST_GRAPH_EDGES_ORIENTED_EDGES = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 3],
        [3, 4],
    ]
).astype(np.int32)

TEST_GRAPH_EDGES_TRIANGLES = [(0, 1, 2)]

TEST_GRAPH_EDGES_ORIENTED_TRIANGLES = np.array(
    [
        [1, 4, -2],
    ]
).astype(np.int32)

TEST_GRAPH_EDGES_ORIENTED_TRIANGLES_AUTO = np.array(
    [
        [1, 4, -2],
        [4, 6, -5],
    ]
).astype(np.int32)
