import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels.matern_kernel import default_num
from geometric_kernels.lab_extras import complex_conj
from geometric_kernels.spaces import SpecialOrthogonal, SpecialUnitary

from ..helper import check_function_with_backend, compact_matrix_lie_groups


@pytest.fixture(
    params=compact_matrix_lie_groups(),
    ids=str,
)
def inputs(request):
    """
    Returns a tuple (space, eigenfunctions, X, X2) where:
    - space = request.param,
    - eigenfunctions = space.get_eigenfunctions(num_levels), with reasonable num_levels
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space,
    """
    space = request.param
    num_levels = min(10, default_num(space))
    eigenfunctions = space.get_eigenfunctions(num_levels)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=100 + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    return space, eigenfunctions, X, X2


def get_dtype(group):
    if isinstance(group, SpecialOrthogonal):
        return np.double
    elif isinstance(group, SpecialUnitary):
        return np.cdouble
    else:
        raise ValueError()


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_group_inverse(inputs, backend):
    group, _, X, _ = inputs

    result = np.eye(group.n, dtype=get_dtype(group))
    result = np.broadcast_to(result, (X.shape[0], group.n, group.n))

    check_function_with_backend(
        backend,
        result,
        lambda X: B.matmul(X, group.inverse(X)),
        X,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_character_conj_invariant(inputs, backend):
    group, eigenfunctions, X, G = inputs

    # Truncate X and G to have the same length
    n_xs = min(X.shape[0], G.shape[0])
    X = X[:n_xs, :, :]
    G = G[:n_xs, :, :]

    def gammas_diff(X, G, chi):
        conjugates = B.matmul(B.matmul(G, X), group.inverse(G))
        conj_gammas = eigenfunctions._torus_representative(conjugates)

        xs_gammas = eigenfunctions._torus_representative(X)

        return chi(xs_gammas) - chi(conj_gammas)

    for chi in eigenfunctions._characters:
        check_function_with_backend(
            backend,
            np.zeros((n_xs,)),
            lambda X, G: gammas_diff(X, G, chi),
            X,
            G,
            atol=1e-3,
        )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_character_at_identity(inputs, backend):
    group, eigenfunctions, _, _ = inputs

    for chi, dim in zip(eigenfunctions._characters, eigenfunctions._dimensions):
        check_function_with_backend(
            backend,
            np.array([dim], dtype=get_dtype(group)),
            lambda X: B.real(chi(eigenfunctions._torus_representative(X))),
            np.eye(group.n, dtype=get_dtype(group))[None, ...],
        )

        check_function_with_backend(
            backend,
            np.array([0], dtype=get_dtype(group)),
            lambda X: B.imag(chi(eigenfunctions._torus_representative(X))),
            np.eye(group.n, dtype=get_dtype(group))[None, ...],
        )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_characters_orthogonal(inputs, backend):
    group, eigenfunctions, _, _ = inputs

    num_samples = 10000
    key = np.random.RandomState(0)
    _, X = group.random(key, num_samples)

    def all_char_vals(X):
        gammas = eigenfunctions._torus_representative(X)
        values = [
            chi(gammas)[..., None]  # [num_samples, 1]
            for chi in eigenfunctions._characters
        ]

        return B.concat(*values, axis=-1)

    check_function_with_backend(
        backend,
        np.eye(eigenfunctions.num_levels, dtype=get_dtype(group)),
        lambda X: complex_conj(B.T(all_char_vals(X))) @ all_char_vals(X) / num_samples,
        X,
        atol=0.4,  # very loose, but helps make sure the diagonal is close to 1 while the rest is close to 0
    )
