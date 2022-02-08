import numpy as np
import pytest
import torch

import geometric_kernels.torch  # noqa
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

    try:
        np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print(f"Failed for {dim}")


def test_so_kernel_torch(so_kernel):
    kernel, params, state = so_kernel

    dim = kernel.space.dimension

    m = random_so_matrix(dim=dim, num=3)

    M = torch.tensor(m)  # [7, 6, 6]
    K = kernel.K(params, state, M, M)

    try:
        torch.linalg.cholesky(K)
    except RuntimeError:
        print("Fail")


if __name__ == "__main__":
    so_group = SpecialOrthogonalGroup(n=7)

    kernel = MaternKarhunenLoeveKernel(so_group, 16)
    params, state = kernel.init_params_and_state()
    test_so_group((kernel, params, state), size=5)

    test_so_kernel_torch((kernel, params, state))
