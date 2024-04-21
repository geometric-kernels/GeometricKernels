import sys

import jax.numpy as jnp
import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch

from geometric_kernels.feature_maps import (
    DeterministicFeatureMapCompact,
    RandomPhaseFeatureMapCompact,
    RandomPhaseFeatureMapNoncompact,
    RejectionSamplingFeatureMapHyperbolic,
    RejectionSamplingFeatureMapSPD,
)
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import (
    Circle,
    Hyperbolic,
    Hypersphere,
    Mesh,
    SymmetricPositiveDefiniteMatrices,
)


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
    elif backend in ["torch", "pytorch"]:
        return torch.tensor(value)
    elif backend == "numpy":
        return value
    elif backend == "jax":
        return jnp.array(value)
    else:
        raise ValueError("Unknown backend: {}".format(backend))


def mesh_point():
    n_base = 4
    n_vertices = 2 * n_base
    vertices = np.array(
        [
            (
                1.0 * (i % 2),
                np.cos(2 * np.pi * (i // 2) / n_base),
                np.sin(2 * np.pi * (i // 2) / n_base),
            )
            for i in range(n_vertices)
        ]
    )
    faces = np.array(
        [
            (i % n_vertices, (i + 1) % n_vertices, (i + 2) % n_vertices)
            for i in range(n_vertices)
        ]  # box without sides
        + [
            (i % 2, (i + 2) % n_vertices, (i + 4) % n_vertices)
            for i in range(n_vertices - 4)
        ]  # sides
    )
    # this is just a box

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


def hyperbolic_point():
    hyperboloid = Hyperbolic(dim=2)

    point = hyperboloid.random_point(1).reshape(1, -1)

    return hyperboloid, point


def spd_point():
    spd = SymmetricPositiveDefiniteMatrices(2)

    point = spd.random_point(1).reshape(1, 2, 2)

    return spd, point


@pytest.fixture(name="noncompact_spacepoint", params=["hyperbolic", "spd"])
def _noncompact_spacepoint(request):
    if request.param == "hyperbolic":
        return hyperbolic_point()
    elif request.param == "spd":
        return spd_point()
    else:
        raise ValueError("Unknown space {}".format(request.param))


@pytest.fixture(name="kl_spacepoint", params=["circle", "hypersphere", "mesh"])
def _kl_spacepoint_fixture(request):
    if request.param == "circle":
        return circle_point()
    elif request.param == "hypersphere":
        return hypersphere_point()
    elif request.param == "mesh":
        return mesh_point()
    else:
        raise ValueError("Unknown space {}".format(request.param))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_karhunen_loeve_dtype(kl_spacepoint, dtype, backend):
    space, point = kl_spacepoint
    point = to_typed_ndarray(point, dtype)
    point = to_typed_tensor(point, backend)

    kernel = MaternKarhunenLoeveKernel(space, 3)

    params = kernel.init_params()
    params["nu"] = to_typed_tensor(to_typed_ndarray(np.r_[0.5], dtype), backend)
    params["lengthscale"] = to_typed_tensor(
        to_typed_ndarray(np.r_[0.5], dtype), backend
    )

    # make sure that it just runs
    kernel.K(params, point)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "tensorflow"])
def test_feature_map_dtype(kl_spacepoint, dtype, backend):
    space, point = kl_spacepoint
    point = to_typed_ndarray(point, dtype)
    point = to_typed_tensor(point, backend)

    num_levels = 3
    kernel = MaternKarhunenLoeveKernel(space, num_levels)

    params = kernel.init_params()
    params["nu"] = to_typed_tensor(to_typed_ndarray(np.r_[0.5], dtype), backend)
    params["lengthscale"] = to_typed_tensor(
        to_typed_ndarray(np.r_[0.5], dtype), backend
    )

    # make sure it runs
    feature_map = DeterministicFeatureMapCompact(space, num_levels)
    feature_map(point, params)

    # make sure it runs
    key = B.create_random_state(B.dtype(point), seed=1234)
    feature_map = RandomPhaseFeatureMapCompact(space, num_levels)
    feature_map(point, params, key=key)


@pytest.fixture(params=["naive", "rs"])
def feature_map_on_noncompact(request, noncompact_spacepoint):
    space = noncompact_spacepoint[0]
    if request.param == "naive":
        feature_map = RandomPhaseFeatureMapNoncompact(space, 10)
    elif request.param == "rs" and isinstance(space, Hyperbolic):
        feature_map = RejectionSamplingFeatureMapHyperbolic(space, 10)
    elif request.param == "rs" and isinstance(space, SymmetricPositiveDefiniteMatrices):
        feature_map = RejectionSamplingFeatureMapSPD(space, 10)
    else:
        raise ValueError(f"Unknown feature map {request.param}")
    return noncompact_spacepoint + (feature_map,)


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="requires newer numpy version, unavailable in Python<=3.7",
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "jax", "torch", "tensorflow"])
@pytest.mark.parametrize("nu", [0.5, np.inf])
def test_feature_map_noncompact_dtype(feature_map_on_noncompact, dtype, backend, nu):
    space, point, feature_map = feature_map_on_noncompact
    point = to_typed_ndarray(point, dtype)
    point = to_typed_tensor(point, backend)

    params = {}
    params["nu"] = to_typed_tensor(to_typed_ndarray(np.r_[nu], dtype), backend)
    params["lengthscale"] = to_typed_tensor(
        to_typed_ndarray(np.r_[0.5], dtype), backend
    )

    # make sure it runs
    key = B.create_random_state(B.dtype(point), seed=1234)
    feature_map(point, params, key=key)
