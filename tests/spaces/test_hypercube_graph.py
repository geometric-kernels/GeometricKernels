import lab as B
import numpy as np
import pytest
from plum import Tuple

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import HypercubeGraph
from geometric_kernels.utils.kernel_formulas import hypercube_graph_heat_kernel

from ..helper import check_function_with_backend


@pytest.fixture(params=[1, 2, 3, 5, 10])
def inputs(request) -> Tuple[B.Numeric]:
    """
    Returns a tuple (space, eigenfunctions, X, X2) where:
    - space is a HypercubeGraph object with dimension equal to request.param,
    - eigenfunctions is the respective Eigenfunctions object with at most 5 levels,
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space,
    - weights is an array of positive numbers of shape (eigenfunctions.num_levels, 1).
    """
    d = request.param
    space = HypercubeGraph(d)
    eigenfunctions = space.get_eigenfunctions(min(space.dim + 1, 5))

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    # These weights are used for testing the weighted outerproduct, they
    # should be positive.
    weights = np.random.rand(eigenfunctions.num_levels, 1) ** 2 + 1e-5

    return space, eigenfunctions, X, X2, weights


def test_numbers_of_eigenfunctions(inputs):
    space, eigenfunctions, _, _, _ = inputs
    num_levels = eigenfunctions.num_levels

    # If the number of levels is maximal, check that the number of
    # eigenfunctions is equal to the number of binary vectors of size `space.dim`.
    if num_levels == space.dim + 1:
        assert eigenfunctions.num_eigenfunctions == 2**space.dim


@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_against_analytic_heat_kernel(inputs, lengthscale, backend):
    space, _, X, X2, _ = inputs
    lengthscale = np.array([lengthscale])
    result = hypercube_graph_heat_kernel(lengthscale, X, X2)

    kernel = MaternGeometricKernel(space)

    # Check that MaternGeometricKernel on HypercubeGraph with nu=infinity
    # coincides with the closed form expression for the heat kernel on the
    # hypercube graph.
    check_function_with_backend(
        backend,
        result,
        kernel.K,
        {"nu": np.array([np.inf]), "lengthscale": lengthscale},
        X,
        X2,
        atol=1e-2,
    )
