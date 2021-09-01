from importlib import import_module

import eagerpy as ep
import numpy as np
from multipledispatch import dispatch

from .types import TensorLike

tf = None
torch = None


@dispatch(ep.TensorFlowTensor)
def cast_to_int(t: ep.TensorFlowTensor):
    global tf
    if tf is None:
        tf = import_module("tensorflow")
    return ep.astensor(tf.cast(t.raw, tf.int64))


@dispatch(ep.PyTorchTensor)
def cast_to_int(t: ep.PyTorchTensor):
    global torch
    if torch is None:
        torch = import_module("torch")
    return ep.astensor(t.raw.to(torch.int64))


@dispatch(ep.NumPyTensor)
def cast_to_int(t: ep.NumPyTensor):
    return t.astype(np.int64)


@dispatch(ep.PyTorchTensor, ep.PyTorchTensor)
def take_along_axis(t: ep.PyTorchTensor, index: ep.PyTorchTensor, axis=0):
    global torch
    if torch is None:
        torch = import_module("torch")

    return type(t)(torch.index_select(t.raw, axis, index.raw.ravel())).astype(t.dtype)


@dispatch(ep.TensorFlowTensor, ep.TensorFlowTensor)
def take_along_axis(t: ep.TensorFlowTensor, index: ep.TensorFlowTensor, axis=0):
    global tf
    if tf is None:
        tf = import_module("tensorflow")

    return type(t)(tf.gather(t.raw, tf.reshape(index.raw, (-1,)), axis=axis)).astype(t.dtype)


@dispatch(ep.TensorFlowTensor, ep.NumPyTensor)
def take_along_axis(t: ep.TensorFlowTensor, index: ep.NumPyTensor, axis=0):
    global tf
    if tf is None:
        tf = import_module("tensorflow")

    return type(t)(tf.gather(t.raw, index.raw.ravel(), axis=axis)).astype(t.dtype)


@dispatch(ep.NumPyTensor, ep.NumPyTensor)
def take_along_axis(t: ep.TensorFlowTensor, index: TensorLike, axis=0):
    return type(t)(np.take_along_axis(t.raw, index.raw, axis=axis))
