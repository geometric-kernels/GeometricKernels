"""
Composite routines missing in eagerpy.
"""
from importlib import import_module
from typing import Callable, List, Tuple, Union

import eagerpy as ep
import einops
import numpy as np
from eagerpy.tensor import Tensor
from multipledispatch import Dispatcher
from opt_einsum import contract

__all__ = [
    "abs",
    "cast_to_int",
    "einsum",
    "rearrange",
    "reduce",
    "repeat",
    "take_along_axis",
    "cos",
    "sin",
]


tf = None
torch = None

cos = Dispatcher("sin")
sin = Dispatcher("cos")
cast_to_int = Dispatcher("cast_to_int")
take_along_axis = Dispatcher("take_along_axis")


def abs(t: Tensor) -> Tensor:
    # TODO(VD): dispatch to different backends for more efficient code.
    return (t ** 2) ** 0.5


def rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    Eagerpy wrapper for einops.rearrange.

    From: https://einops.rocks/api/rearrange/

    A reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation),
    reshape (view), squeeze, unsqueeze, stack, concatenate and other operations.

    :param tensor: list of tensors is also accepted, those should be of the same type and shape
    :param pattern: reduction pattern
    """
    return ep.astensor(einops.rearrange(tensor.raw, pattern, **axes_lengths))


def reduce(
    tensor: Union[Tensor, List[Tensor]],
    pattern: str,
    reduction: Union[str, Callable[[Tensor, Tuple[int]], Tensor]],
    **axes_lengths: int,
) -> Tensor:
    """
    Eagerpy wrapper for einops.reduce.

    From: https://einops.rocks/api/reduce/

    Provides combination of reordering and reduction using reader-friendly notation.

    :param tensor: list of tensors is also accepted, those should be of the same type and shape
    :param pattern: reduction pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean',
        'prod'), case-sensitive alternatively, a callable f(tensor, reduced_axes) ->
        tensor can be provided.
    :param axes_lengths: any additional specifications for dimensions
    """
    return ep.astensor(
        einops.reduce(tensor.raw, pattern=pattern, reduction=reduction, **axes_lengths)
    )


def repeat(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    Eagerpy wrapper for einops.repeat

    From: https://einops.rocks/api/repeat/

    A reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation),
    reshape (view), squeeze, unsqueeze, stack, concatenate and other operations.

    :param tensor: list of tensors is also accepted, those should be of the same type and shape
    :param pattern: reduction pattern
    """
    return ep.astensor(einops.repeat(tensor.raw, pattern, **axes_lengths))


def einsum(subscripts: str, *operands: List[Tensor]) -> Tensor:
    """
    Eagerpy wrapper for opt_einsum.contract.

    Does typical einsum as available tf.einsum.

    :param subscripts: Specifies the subscripts for summation.
    :param operands: (list of tensor) these are the arrays for the operation
    """
    return ep.astensor(contract(subscripts, *[o.raw for o in operands]))


###############
# cos
###############
@cos.register(ep.TensorFlowTensor)
def _cos_tf(t: ep.TensorFlowTensor) -> ep.TensorFlowTensor:
    global tf
    if tf is None:
        tf = import_module("tensorflow")
    return type(t)(tf.math.cos(t.raw))  # type: ignore[misc]


@cos.register(ep.PyTorchTensor)
def _cos_torch(t: ep.PyTorchTensor) -> ep.PyTorchTensor:
    global torch
    if torch is None:
        torch = import_module("torch")
    return type(t)(torch.cos(t.raw))  # type: ignore[misc]


@cos.register(ep.NumPyTensor)
def _cos_numpy(t: ep.NumPyTensor) -> ep.NumPyTensor:
    return type(t)(np.cos(t.raw))


###############
# sin
###############
@sin.register(ep.TensorFlowTensor)
def _sin_tf(t: ep.TensorFlowTensor) -> ep.TensorFlowTensor:
    global tf
    if tf is None:
        tf = import_module("tensorflow")
    return type(t)(tf.math.sin(t.raw))  # type: ignore[misc]


@sin.register(ep.PyTorchTensor)
def _sin_torch(t: ep.PyTorchTensor) -> ep.PyTorchTensor:
    global torch
    if torch is None:
        torch = import_module("torch")
    return type(t)(torch.sin(t.raw))  # type: ignore[misc]


@sin.register(ep.NumPyTensor)
def _sin_numpy(t: ep.NumPyTensor) -> ep.NumPyTensor:
    return type(t)(np.sin(t.raw))


###############
# cast to int
###############
@cast_to_int.register(ep.TensorFlowTensor)
def _cast_to_int_tf(t: ep.TensorFlowTensor):
    global tf
    if tf is None:
        tf = import_module("tensorflow")
    return ep.astensor(tf.cast(t.raw, tf.int64))  # type: ignore[misc]


@cast_to_int.register(ep.PyTorchTensor)
def _cast_to_int_torch(t: ep.PyTorchTensor):
    global torch
    if torch is None:
        torch = import_module("torch")
    return ep.astensor(t.raw.to(torch.int64))  # type: ignore[misc]


@cast_to_int.register(ep.NumPyTensor)
def _cast_to_int_numpy(t: ep.NumPyTensor):
    return t.astype(np.int64)


###############
# take along axis
###############
@take_along_axis.register(ep.PyTorchTensor, ep.PyTorchTensor)
def _take_along_axis_torch(t: ep.PyTorchTensor, index: ep.PyTorchTensor, axis=0):
    global torch
    if torch is None:
        torch = import_module("torch")

    return type(t)(torch.index_select(t.raw, axis, index.raw.ravel())).astype(  # type: ignore[misc]
        t.dtype
    )


@take_along_axis.register(ep.TensorFlowTensor, ep.TensorFlowTensor)
def _take_along_axis_tf(t: ep.TensorFlowTensor, index: ep.TensorFlowTensor, axis=0):
    global tf
    if tf is None:
        tf = import_module("tensorflow")

    return type(t)(
        tf.gather(t.raw, tf.reshape(index.raw, (-1,)), axis=axis)  # type: ignore[misc]
    ).astype(t.dtype)


@take_along_axis.register(ep.TensorFlowTensor, ep.NumPyTensor)
def _take_along_axis_tf_np(t: ep.TensorFlowTensor, index: ep.NumPyTensor, axis=0):
    global tf
    if tf is None:
        tf = import_module("tensorflow")

    return type(t)(tf.gather(t.raw, index.raw.ravel(), axis=axis)).astype(  # type: ignore[misc]
        t.dtype
    )


@take_along_axis.register(ep.NumPyTensor, ep.NumPyTensor)
def _take_along_axis_numpy(t: ep.NumPyTensor, index: ep.NumPyTensor, axis=0):
    return type(t)(np.take_along_axis(t.raw, index.raw, axis=axis))  # type: ignore[misc]
