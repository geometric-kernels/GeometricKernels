import warnings

import networkx as nx
import numpy as np

from geometric_kernels.jax import *  # noqa
from geometric_kernels.spaces.graph_edge import GraphEdge
from geometric_kernels.torch import *  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)

triangles = [(1, 2, 3)]
B1 = nx.incidence_matrix(G).toarray()
B2 = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [0.0]])
sc = GraphEdge(B1, B2)
m = sc.num_edges
eigs = np.array(
    [
        [6.21724894e-15],
        [1.38196601e00],
        [2.38196601e00],
        [3.00000000e00],
        [3.61803399e00],
        [4.61803399e00],
    ]
)


def test_get_eigenvalues(tol=1e-7):
    ##############################################
    # Eigendecomposition checks
    evals = sc.get_eigenvalues(m)
    # check vals
    np.testing.assert_allclose(evals, eigs, atol=tol, rtol=tol)
