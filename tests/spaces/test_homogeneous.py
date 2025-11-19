import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.kernels.matern_kernel import MaternGeometricKernel
from geometric_kernels.spaces.stiefel import Stiefel

from ..helper import check_function_with_backend, create_random_state, np_to_backend


def _choose_lengthscale_ratio_two(
    kernel: MaternKarhunenLoeveKernel, params: dict
) -> float:
    """
    Choose a lengthscale such that the first spectral weight divided by the
    second equals 2 for the given kernel. Returns the scalar lengthscale.

    - If nu == inf: w(s) = exp(-ls^2 * s / 2) => ls^2 = 2 ln 2 / (s1 - s0)
    - If nu < inf:  w(s) = (2 nu / ls^2 + s)^(-nu - d/2)
                    => let p = nu + d/2, a = 2 nu / ls^2
                       ((a+s1)/(a+s0))^p = 2 => a = (s1 - 2^(1/p) s0) / (2^(1/p)-1)
                       => ls^2 = 2 nu / a
    """
    # Extract the first two Laplacian eigenvalues (shape [L, 1])
    s = np.asarray(kernel.eigenvalues_laplacian).reshape(-1)
    if s.shape[0] < 2:
        # Fallback: if not enough levels, just return default 1.0
        return 1.0
    s0, s1 = float(s[0]), float(s[1])

    # Pull parameters
    nu = float(np.asarray(params["nu"]).reshape(()))
    d = int(kernel.space.dimension)

    if np.isinf(nu):
        denom = s1 - s0
        l2 = 2.0 * np.log(2.0) / denom
        return np.sqrt(np.array([l2])), np.array([nu])
    else:
        p = nu + d / 2.0
        two_pow = 2.0 ** (1.0 / p)
        denom = two_pow - 1.0
        a = (s1 - two_pow * s0) / denom
        l2 = 2.0 * nu / a
        return np.sqrt(np.array([l2])), np.array([nu])


@pytest.fixture(params=[(4, 2), (5, 2), (6, 3)], ids=lambda p: f"V({p[1]},{p[0]})")
def stiefel_space(request):
    n, m = request.param
    key = np.random.RandomState(0)
    # Increase stabilizer samples to reduce Monte Carlo variance in averaging
    key, space = Stiefel(n, m, key, average_order=500)
    return key, space


@pytest.mark.parametrize(
    "backend", ["numpy", "tensorflow", "torch", "jax"]
)  # Limit to numpy to avoid optional deps
def test_stiefel_kernel(stiefel_space, backend):
    key, space = stiefel_space
    space.samples_H = np_to_backend(space.samples_H, backend)
    space.matrix_complement = np_to_backend(space.matrix_complement, backend)

    # Number of levels for RFF kernel (created via MaternGeometricKernel for homogeneous spaces)
    num_levels = 2
    rs = create_random_state(backend)
    kernel_rff = MaternGeometricKernel(space, num=num_levels, normalize=True, key=rs)
    params_rff = kernel_rff.init_params()
    G = space.G
    kernel_G = MaternKarhunenLoeveKernel(G, num_levels, normalize=True)
    params_G = kernel_G.init_params()
    tuned_params = _choose_lengthscale_ratio_two(kernel_G, params_G)
    tuned_ls = np_to_backend(tuned_params[0], backend)
    tuned_nu = np_to_backend(tuned_params[1], backend)

    params_rff["lengthscale"] = tuned_ls
    params_G["lengthscale"] = tuned_ls
    params_rff["nu"] = tuned_nu
    params_G["nu"] = tuned_nu

    # Stabilizer elements embedded into G
    h_emb = space.embed_stabilizer(space.samples_H)
    H = h_emb.shape[0]

    # Sample N points on the Stiefel manifold
    N = 5
    key, X = space.random(key, N, project=True)

    # Compare K_rff(X,X) vs average_h K_G(gx, gy @ h), with renormalization by mean diagonal
    def diff(X):
        nX = X.shape[0]
        gX = space.embed_manifold(X)  # [N, n, n]
        # [N, H, n, n] of gy @ h
        YH = B.matmul(B.expand_dims(gX, axis=1), B.expand_dims(h_emb, axis=0))
        X2_big = B.reshape(YH, -1, space.n, space.n)  # [N*H, n, n]
        KG = kernel_G.K(params_G, gX, X2_big)  # [N, N*H]
        KG_reshaped = B.reshape(KG, nX, nX, H)  # [N, N, H]
        K_avg = B.mean(B.transpose(KG_reshaped, [0, 2, 1]), axis=1)  # [N, N]
        # Renormalize averaged kernel to unit mean diagonal
        identity = B.eye(B.dtype(K_avg), nX, nX)
        mean_diag = B.sum(B.real(K_avg * identity)) / nX
        K_avg = K_avg / mean_diag

        K_rff = kernel_rff.K(params_rff, X, X)
        return B.sum(B.abs(K_rff - K_avg), squeeze=False) / (
            nX * (nX - 1)
        )  # mean exclude diagonal

    check_function_with_backend(
        backend,
        np.zeros((1, 1)),
        lambda X: diff(X),
        X,
        atol=5e-1,
    )
