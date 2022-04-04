import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.mesh import Mesh


def to_typed_ndarray(value, dtype):
    if dtype == "float32":
        return value.astype(np.float32)
    elif dtype == "float64":
        return value.astype(np.float64)
    else:
        raise ValueError("Unknown dtype: {}".format(dtype))


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


def mesh_point():
    n_vertices = 10
    vertices = np.array(
        [(1.0 * (i % 2), 1.0 * (i // 2), 0.0) for i in range(n_vertices)]
    )
    faces = np.array([(i, i + 1, i + 2) for i in range(n_vertices - 2)])
    mesh = Mesh(vertices, faces)
    point = np.array([0]).reshape(1, 1)

    return mesh, point


def circle_point():
    circle = Circle()

    point = np.array([0]).reshape(1, 1)

    return circle, point


def hypersphere_point():
    hypersphere = Hypersphere(dim=2)

    point = hypersphere.random_point(1).reshape(1, -1)

    return hypersphere, point


@pytest.fixture(name="spacepoint", params=["circle", "hypersphere", "mesh"])
def _spacepoint_fixture(request):
    if request.param == "circle":
        return circle_point()
    elif request.param == "hypersphere":
        return hypersphere_point()
    elif request.param == "mesh":
        return mesh_point()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_dtype(spacepoint, dtype, backend):
    space, point = spacepoint
    # if not isinstance(space, Mesh):
    point = to_typed_ndarray(point, dtype)
    point = to_typed_tensor(point, backend)

    kernel = MaternKarhunenLoeveKernel(space, 3)

    params, state = kernel.init_params_and_state()
    params["nu"] = to_typed_tensor(to_typed_ndarray(np.r_[0.5], dtype), backend)
    params["lenghtscale"] = to_typed_tensor(
        to_typed_ndarray(np.r_[0.5], dtype), backend
    )

    # make sure that it just runs
    kernel.K(params, state, point)
