import lab as B
import numpy as np
from beartype.door import die_if_unbearable, is_bearable
from beartype.typing import Any, Callable, List, Optional, Union
from plum import ModuleType, resolve_type_hint

from geometric_kernels.lab_extras import SparseArray
from geometric_kernels.spaces import (
    Circle,
    CompactMatrixLieGroup,
    DiscreteSpectrumSpace,
    Graph,
    Hyperbolic,
    HypercubeGraph,
    Hypersphere,
    Mesh,
    NoncompactSymmetricSpace,
    ProductDiscreteSpectrumSpace,
    Space,
    SpecialOrthogonal,
    SpecialUnitary,
    SymmetricPositiveDefiniteMatrices,
)

from .data import TEST_GRAPH_ADJACENCY, TEST_MESH_PATH

EagerTensor = ModuleType("tensorflow.python.framework.ops", "EagerTensor")


def compact_matrix_lie_groups() -> List[CompactMatrixLieGroup]:
    return [
        SpecialOrthogonal(3),
        SpecialOrthogonal(8),
        SpecialUnitary(2),
        SpecialUnitary(5),
    ]


def product_discrete_spectrum_spaces() -> List[ProductDiscreteSpectrumSpace]:
    return [
        ProductDiscreteSpectrumSpace(Circle(), Hypersphere(3), Circle()),
        ProductDiscreteSpectrumSpace(
            Circle(), Graph(np.kron(TEST_GRAPH_ADJACENCY, TEST_GRAPH_ADJACENCY))
        ),  # TEST_GRAPH_ADJACENCY is too small for default parameters of the ProductDiscreteSpectrumSpace
        ProductDiscreteSpectrumSpace(Mesh.load_mesh(TEST_MESH_PATH), Hypersphere(2)),
    ]


def discrete_spectrum_spaces() -> List[DiscreteSpectrumSpace]:
    return (
        [
            Circle(),
            HypercubeGraph(1),
            HypercubeGraph(3),
            HypercubeGraph(6),
            Hypersphere(2),
            Hypersphere(3),
            Hypersphere(10),
            Mesh.load_mesh(TEST_MESH_PATH),
            Graph(TEST_GRAPH_ADJACENCY, normalize_laplacian=False),
            Graph(TEST_GRAPH_ADJACENCY, normalize_laplacian=True),
        ]
        + compact_matrix_lie_groups()
        + product_discrete_spectrum_spaces()
    )


def noncompact_symmetric_spaces() -> List[NoncompactSymmetricSpace]:
    return [
        Hyperbolic(2),
        Hyperbolic(3),
        Hyperbolic(8),
        Hyperbolic(9),
        SymmetricPositiveDefiniteMatrices(2),
        SymmetricPositiveDefiniteMatrices(3),
        SymmetricPositiveDefiniteMatrices(6),
        SymmetricPositiveDefiniteMatrices(7),
    ]


def spaces() -> List[Space]:
    return discrete_spectrum_spaces() + noncompact_symmetric_spaces()


def np_to_backend(value: B.NPNumeric, backend: str):
    """
    Converts a numpy array to the desired backend.

    :param value:
        A numpy array.
    :param backend:
        The backend to use, one of the strings "tensorflow", "torch", "numpy", "jax".

    :raises ValueError:
        If the backend is not recognized.

    :return:
        The array `value` converted to the desired backend.
    """
    if backend == "tensorflow":
        import tensorflow as tf

        return tf.convert_to_tensor(value)
    elif backend in ["torch", "pytorch"]:
        import torch

        return torch.tensor(value)
    elif backend == "numpy":
        return value
    elif backend == "jax":
        import jax.numpy as jnp

        return jnp.array(value)
    elif backend == "scipy_sparse":
        import scipy.sparse as sp

        return sp.csr_array(value)
    else:
        raise ValueError("Unknown backend: {}".format(backend))


def create_random_state(backend: str, seed: int = 0):
    dtype = B.dtype(np_to_backend(np.array([1.0]), backend))
    return B.create_random_state(dtype, seed=seed)


def array_type(backend: str):
    """
    Returns the array type corresponding to the given backend.

    :param backend:
        The backend to use, one of the strings "tensorflow", "torch", "numpy",
        "jax", "scipy_sparse".

    :return:
        The array type corresponding to the given backend.
    """
    if backend == "tensorflow":
        return resolve_type_hint(Union[B.TFNumeric, EagerTensor])
    elif backend in ["torch", "pytorch"]:
        return resolve_type_hint(B.TorchNumeric)
    elif backend == "numpy":
        return resolve_type_hint(B.NPNumeric)
    elif backend == "jax":
        return resolve_type_hint(B.JAXNumeric)
    elif backend == "scipy_sparse":
        return resolve_type_hint(SparseArray)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def apply_recursive(data: Any, func: Callable[[Any], Any]) -> Any:
    """
    Apply a function recursively to a nested data structure. Supports lists and
    dictionaries.

    :param data:
        The data structure to apply the function to.
    :param func:
        The function to apply.

    :return:
        The data structure with the function applied to each element.
    """
    if isinstance(data, dict):
        return {key: apply_recursive(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [apply_recursive(element, func) for element in data]
    else:
        return func(data)


def check_function_with_backend(
    backend: str,
    result: Any,
    f: Callable,
    *args: Any,
    compare_to_result: Optional[Callable] = None,
    atol=1e-4,
):
    """
    1. Casts the arguments `*args` to the backend `backend`.
    2. Runs the function `f` on the casted arguments.
    3. Checks that the result is of the backend `backend`.
    4. If no `compare_to_result` kwarg is provided, checks that the result,
       casted back to numpy backend, coincides with the given `result`.
       If `compare_to_result` is provided, checks if
       `compare_to_result(result, f_output)` is True.

    :param backend:
        The backend to use, one of the strings "tensorflow", "torch", "numpy", "jax".
    :param result:
        The expected result of the function, if no `compare_to_result` kwarg is
        provided, expected to be a numpy array. Otherwise, can be anything.
    :param f:
        The backend-independent function to run.
    :param args:
        The arguments to pass to the function `f`, expected to be numpy arrays
        or non-array arguments.
    :param compare_to_result:
        A function that takes two arguments, the computed result and the
        expected result, and returns a boolean.
    :param atol:
        The absolute tolerance to use when comparing the computed result with
        the expected result.
    """

    def cast(arg):
        if is_bearable(arg, B.Numeric):
            # We only expect numpy arrays here
            die_if_unbearable(arg, B.NPNumeric)
            return np_to_backend(arg, backend)
        else:
            return arg

    args_casted = (apply_recursive(arg, cast) for arg in args)

    f_output = f(*args_casted)
    assert is_bearable(
        f_output, array_type(backend)
    ), f"The output is not of the expected type. Expected: {array_type(backend)}, got: {type(f_output)}"
    if compare_to_result is None:
        # we convert `f_output` to numpy array to compare with `result``
        if is_bearable(f_output, SparseArray):
            f_output = f_output.toarray()
        else:
            f_output = B.to_numpy(f_output)
        assert (
            result.shape == f_output.shape
        ), f"Shapes do not match: {result.shape} (for result) vs {f_output.shape} (for f_output)"
        np.testing.assert_allclose(f_output, result, atol=atol)
    else:
        assert compare_to_result(
            result, f_output
        ), f"compare_to_result(result, f_output) failed with\n result:\n{result}\n\nf_output:\n{f_output}"
