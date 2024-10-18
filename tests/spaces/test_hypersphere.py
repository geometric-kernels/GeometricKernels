import lab as B
import numpy as np
import pytest
from plum import Tuple

from geometric_kernels.spaces import Hypersphere
from geometric_kernels.spaces.hypersphere import SphericalHarmonics
from geometric_kernels.utils.utils import chain


class Consts:
    seed = 42
    dimension = 2
    num_data = 7
    num_data2 = 5
    num_eigenfunctions = 9


@pytest.fixture(name="eigenfunctions")
def _eigenfunctions_fixture():
    return SphericalHarmonics(Consts.dimension, Consts.num_eigenfunctions)


@pytest.fixture(name="inputs")
def _inputs_fixure(request) -> Tuple[B.Numeric]:
    def _norm(v):
        return np.sum(v**2, axis=-1, keepdims=True) ** 0.5

    np.random.seed(Consts.seed)
    value = np.random.randn(Consts.num_data, Consts.dimension + 1)
    value = value / _norm(value)
    value2 = np.random.randn(Consts.num_data2, Consts.dimension + 1)
    value2 = value2 / _norm(value2)
    return value, value2


@pytest.mark.parametrize(
    "dim, num, expected_num, expected_num_levels",
    [(2, 3, 9, 3), (2, 4, 16, 4), (3, 3, 14, 3), (3, 4, 30, 4), (8, 2, 10, 2)],
)
def test_shape_eigenfunctions(dim, num, expected_num, expected_num_levels):
    sph_harmonics = SphericalHarmonics(dim, num)
    assert len(sph_harmonics._spherical_harmonics) == sph_harmonics.num_eigenfunctions
    assert sph_harmonics.num_eigenfunctions == expected_num
    assert sph_harmonics.num_levels == expected_num_levels


def test_call_eigenfunctions(
    inputs: Tuple[B.Numeric, B.Numeric],
    eigenfunctions: SphericalHarmonics,
):
    inputs, _ = inputs
    output = eigenfunctions(inputs)
    assert output.shape == (Consts.num_data, eigenfunctions.num_eigenfunctions)


def test_eigenfunctions_shape(eigenfunctions: SphericalHarmonics):
    num_eigenfunctions_manual = np.sum(eigenfunctions.num_eigenfunctions_per_level)
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions
    assert len(eigenfunctions.num_eigenfunctions_per_level) == eigenfunctions.num_levels


def test_weighted_outerproduct_with_addition_theorem(
    inputs, eigenfunctions: SphericalHarmonics
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, inputs2 = inputs
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    chained_weights = chain(
        weights_per_level, eigenfunctions.num_eigenfunctions_per_level
    )
    weights = B.reshape(weights_per_level, -1, 1)
    actual = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs2))

    Phi_X = eigenfunctions(inputs)
    Phi_X2 = eigenfunctions(inputs2)
    expected = B.einsum("ni,ki,i->nk", Phi_X, Phi_X2, chained_weights)
    np.testing.assert_array_almost_equal(actual, expected)


def test_weighted_outerproduct_with_addition_theorem_same_input(
    inputs, eigenfunctions: SphericalHarmonics
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, _ = inputs
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    weights = B.reshape(weights_per_level, -1, 1)
    first = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs))
    second = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, None))
    np.testing.assert_array_almost_equal(first, second)


def test_weighted_outerproduct_diag_with_addition_theorem(
    inputs, eigenfunctions: SphericalHarmonics
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, _ = inputs
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    chained_weights = chain(
        weights_per_level, eigenfunctions.num_eigenfunctions_per_level
    )
    weights = B.reshape(weights_per_level, -1, 1)
    actual = eigenfunctions.weighted_outerproduct_diag(weights, inputs)

    Phi_X = eigenfunctions(inputs)
    expected = B.einsum("ni,i->n", Phi_X**2, chained_weights)
    np.testing.assert_array_almost_equal(B.to_numpy(actual), B.to_numpy(expected))


def test_sphere_heat_kernel():
    import torch

    import geometric_kernels.torch  # noqa
    from geometric_kernels.kernels import MaternKarhunenLoeveKernel
    from geometric_kernels.utils.manifold_utils import manifold_laplacian

    _TRUNCATION_LEVEL = 10

    # Parameters
    grid_size = 4
    nb_samples = 10
    dimension = 3

    # Create manifold
    hypersphere = Hypersphere(dim=dimension)

    # Generate samples
    ts = torch.linspace(0.1, 1, grid_size, requires_grad=True)
    xs = torch.tensor(
        np.array(hypersphere.random_point(nb_samples)), requires_grad=True
    )
    ys = xs

    # Define kernel
    kernel = MaternKarhunenLoeveKernel(hypersphere, _TRUNCATION_LEVEL, normalize=False)
    params = kernel.init_params()
    params["nu"] = torch.tensor([torch.inf])

    # Define heat kernel function
    def heat_kernel(t, x, y):
        params["lengthscale"] = B.reshape(B.sqrt(2 * t), 1)
        return kernel.K(params, x, y)

    for t in ts:
        for x in xs:
            for y in ys:
                # Compute the derivative of the kernel function wrt t
                dfdt, _, _ = torch.autograd.grad(
                    heat_kernel(t, x[None], y[None]), (t, x, y)
                )
                # Compute the Laplacian of the kernel on the manifold
                egrad = lambda u: torch.autograd.grad(  # noqa
                    heat_kernel(t, u[None], y[None]), (t, u, y)
                )[
                    1
                ]  # noqa
                fx = lambda u: heat_kernel(t, u[None], y[None])  # noqa
                ehess = lambda u, h: torch.autograd.functional.hvp(fx, u, h)[1]  # noqa
                lapf = manifold_laplacian(x, hypersphere, egrad, ehess)

                # Check that they match
                assert np.isclose(dfdt.detach().numpy(), lapf, atol=1.0e-3)
