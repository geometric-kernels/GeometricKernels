import lab as B
import numpy as np

from geometric_kernels.spaces import Hypersphere


def test_sphere_heat_kernel():
    # Tests that the heat kernel on the sphere solves the heat equation. This
    # test only uses torch, as lab doesn't support backend-independent autodiff.
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
                np.testing.assert_allclose(dfdt.detach().numpy(), lapf, atol=1e-3)
