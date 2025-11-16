import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Grassmannian

from ..helper import check_function_with_backend, np_to_backend


def make_tuned_kernels(grass: Grassmannian, num_levels: int, normalize: bool = False):
    """
    Build kernels on G=SO(n) and M=Gr(n,m) with the same `num_levels`, then
    intersect their signatures and set the final Grassmannian kernel to use
    exactly as many levels as the size of that intersection.

    Returns (kernel_M, kernel_G).
    """
    # Build both kernels with the same truncation level
    kernel_G = MaternKarhunenLoeveKernel(
        grass.G, num_levels=num_levels, normalize=normalize
    )
    kernel_M_tmp = MaternKarhunenLoeveKernel(
        grass, num_levels=num_levels, normalize=normalize
    )

    group_sigs = kernel_G.eigenfunctions._signatures
    m_sigs = kernel_M_tmp.eigenfunctions._signatures

    # Direct intersection of signatures without special casing even n
    group_sig_set = set(group_sigs)
    overlap = [s for s in m_sigs if s in group_sig_set]
    m_levels = max(1, len(overlap))

    # Final Grassmannian kernel with levels equal to the intersection size
    kernel_M = MaternKarhunenLoeveKernel(
        grass, num_levels=m_levels, normalize=normalize
    )

    return kernel_M, kernel_G


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


@pytest.fixture(
    scope="module",
    params=[
        (5, 2),
        (6, 3),
        (7, 3),
    ],
    ids=lambda p: f"Gr({p[0]},{p[1]})",
)
def inputs(request):
    """Build kernels, params, samples for a Grassmannian Gr(n,m)."""
    key = np.random.RandomState(0)
    n, m = request.param
    N = 8
    grass = Grassmannian(n, m)
    num_levels = 20
    # Build both with `num_levels`, intersect signatures, and size kernel_M to the overlap.
    # Use normalize=True as required for the averaging identity.
    kernel_M, kernel_G = make_tuned_kernels(
        grass, num_levels=num_levels, normalize=True
    )
    params_M = kernel_M.init_params()
    params_G = kernel_G.init_params()
    # Tune params_M so that the first weight / second weight = 2, and
    # set the same lengthscale for the group kernel parameters.
    tuned_params = _choose_lengthscale_ratio_two(kernel_M, params_M)

    # Representatives for N Grassmannian points using SO(n).random from so.py
    key, g = grass.G.random(key, N)  # [N, n, n]
    x = grass.project_to_manifold(g)  # [N, n, m]

    # Pre-sample stabilizer elements with the built-in sampler (H samples)
    num_h = 8000
    key, h = grass.H.random(key, num_h)  # [H, n, n]

    # Return everything needed
    return kernel_M, kernel_G, params_M, params_G, tuned_params, x, g, h

@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_grassmannian_kernel_averaging(inputs, backend):
    """Grassmannian kernel equals the stabilizer-averaged SO(n) kernel (renormalized)."""
    kernel_M, kernel_G, params_M, params_G, tuned_params, x, g, h = inputs
    tuned_ls = np_to_backend(tuned_params[0], backend)
    nu = np_to_backend(tuned_params[1], backend)
    params_M["lengthscale"] = tuned_ls
    params_G["lengthscale"] = tuned_ls
    params_M["nu"] = nu
    params_G["nu"] = nu
    # Expect zero difference matrix between analytic and averaged group covariance.
    expected = np.zeros((x.shape[0], x.shape[0]))

    def diff(x, g, h):
        # Analytic covariance on the Grassmannian
        K_analytic = kernel_M.K(params_M, x, x)  # [N, N]
        # Build all right-coset transforms G @ h for each point and each stabilizer sample
        # Shapes: G (N, n, n), h (H, n, n)
        g_ = B.expand_dims(g, axis=0)  # [1, N, n, n]
        h_ = B.expand_dims(h, axis=1)  # [H, 1, n, n]
        gh = B.matmul(g_, h_)  # [H, N, n, n]

        # Flatten to pass through the kernel in one shot: [H*N, n, n]
        N = g.shape[0]
        n = g.shape[1]
        H = h.shape[0]
        gh = B.reshape(gh, H * N, n, n)

        # Compute group kernel between all Gi and all (Gj @ h)
        K_gh = kernel_G.K(params_G, g, gh)  # [N, H*N]
        K_gh = B.reshape(K_gh, N, H, N)  # [N, H, N] where axis 1 indexes h
        K_avg = B.mean(K_gh, axis=1)  # [N, N]

        # Renormalize average over H so that diagonal matches normalized Grassmannian
        # kernel (the integral over H is not 1 by default).
        eye_mask = B.eye(B.dtype(K_avg), N)
        diag_mean = B.sum(K_avg * eye_mask) / B.cast(B.dtype(K_avg), np.array(N))
        K_avg = K_avg / diag_mean
        return K_analytic - K_avg

    check_function_with_backend(
        backend,
        expected,
        diff,
        x,
        g,
        h,
        atol=1e-1,
    )
