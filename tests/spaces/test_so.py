import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import SpecialOrthogonalGroup


def random_so_matrix(dim, num):
    h = np.random.randn(num, dim, dim)  # [N, D, D]
    # q, r = np.linalg.qr(h)
    qr = [
        np.linalg.qr(h[i]) for i in range(len(h))
    ]  # backward compatibility with np<1.22
    q = np.stack([Q for Q, _ in qr])  # [N, D, D]
    r = np.stack([R for _, R in qr])  # [N, D, D]
    diag_sign = np.diagonal(np.sign(r), axis1=-2, axis2=-1)  # [N, D]
    q = q * diag_sign[:, None, :]  # [N, D, D]
    det_sign, _ = np.linalg.slogdet(q)  # [N, ]
    sign_matrix = np.eye(dim).reshape(-1, dim, dim).repeat(num, axis=0)  # [N, D, D]
    sign_matrix[:, 0, 0] = det_sign  # [N, D, D]
    q = q @ sign_matrix  # [N, D, D]
    return q


# parametrize by dimension
@pytest.fixture(scope="module", params=[3, 5, 6, 7, 8, 9, 10])
def so_kernel(request):
    so_group = SpecialOrthogonalGroup(request.param)

    kernel = MaternKarhunenLoeveKernel(so_group, 16)
    params, state = kernel.init_params_and_state()

    return kernel, params, state


# parametrize by size
@pytest.mark.parametrize("size", [5, 6, 7, 8, 9, 10])
def test_so_group(so_kernel, size):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension
    rank = dim // 2
    so_dim = 2 * rank ** 2 + rank if dim % 2 else 2 * rank ** 2 - rank
    print(f"\ntesting dim={dim} (dimSO={so_dim}, rank={rank})")

    m = random_so_matrix(dim, size)

    K = kernel.K(params, state, m, m)

    assert K.shape == (size, size)

    np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))


def test_so_kernel_torch(so_kernel):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension

    m = random_so_matrix(dim=dim, num=3)

    M = torch.tensor(m)  # [7, 6, 6]
    K = kernel.K(params, state, M, M)

    torch.linalg.cholesky(K + torch.tensor(1e-6) * torch.eye(K.shape[0]))


def test_so_kernel_tf(so_kernel):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension

    m = random_so_matrix(dim=dim, num=3)

    M = tf.convert_to_tensor(m)
    K = kernel.K(params, state, M, M)

    tf.linalg.cholesky(K + 1e-6 * tf.eye(K.shape[0], dtype=K.dtype))


def test_so_kernel_jax(so_kernel):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension

    m = random_so_matrix(dim=dim, num=3)

    M = jnp.array(m)
    K = kernel.K(params, state, M, M)

    jnp.linalg.cholesky(K + 1e-7 * jnp.eye(K.shape[0]))


if __name__ == "__main__":
    import lab.jax  # noqa
    import lab.tensorflow  # noqa

    import geometric_kernels.jax  # noqa
    import geometric_kernels.lab_extras.tensorflow  # noqa
    import geometric_kernels.torch  # noqa

    so_group = SpecialOrthogonalGroup(n=5)

    kernel = MaternKarhunenLoeveKernel(so_group, 16)
    params, state = kernel.init_params_and_state()
    test_so_group((kernel, params, state), size=3)

    test_so_kernel_torch((kernel, params, state))

    test_so_kernel_tf((kernel, params, state))

    test_so_kernel_jax((kernel, params, state))
