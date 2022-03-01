import jax.numpy as jnp
import lab as B
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


def to_typed_tensor(value, backend):
    if backend == "tensorflow":
        return tf.convert_to_tensor(value)
    elif backend == "torch":
        return torch.tensor(value)
    elif backend == "numpy":
        return value
    elif backend == "jax":
        return jnp.array(value)
    else:
        raise ValueError("Unknown backend: {}".format(backend))


# parametrize by dimension
@pytest.fixture(scope="module", params=[3, 5, 6, 7, 8, 9, 10])
def so_kernel(request):
    so_group = SpecialOrthogonalGroup(request.param)

    kernel = MaternKarhunenLoeveKernel(so_group, 16)
    params, state = kernel.init_params_and_state()

    return kernel, params, state


# parametrize by backend
# parametrize by size
@pytest.mark.parametrize("backend", ["numpy", "jax", "tensorflow", "torch"])
@pytest.mark.parametrize("size", [5, 6, 7, 8, 9, 10])
def test_so_group(so_kernel, backend, size):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension

    m = random_so_matrix(dim, size)

    M = to_typed_tensor(m, backend)

    K = kernel.K(params, state, M, M)

    k = B.to_numpy(K)

    assert k.shape == (size, size)

    np.linalg.cholesky(k + 1e-6 * np.eye(K.shape[0]))
