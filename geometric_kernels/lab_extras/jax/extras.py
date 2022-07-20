from typing import List

import jax.numpy as jnp
import lab as B
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.JAXNumeric]


@dispatch
def take_along_axis(a: Union[_Numeric, B.Numeric], index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    if not isinstance(a, jnp.ndarray):
        a = jnp.array(a)
    return jnp.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.JAXNumeric, b: Union[List, B.NPNumeric, B.Number, B.JAXNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return jnp.array(b)


@dispatch
def trapz(y: B.JAXNumeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the trapezoidal rule.
    """
    return jnp.trapz(y, x, dx, axis)


@dispatch
def logspace(start: B.JAXNumeric, stop: _Numeric, num: int = 50):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return jnp.logspace(start, stop, num)


@dispatch
def degree(a: B.JAXNumeric):  # type: ignore
    """
    Given a vector a, return a diagonal matrix with a as main diagonal.
    """
    degrees = a.sum(axis=0)  # type: ignore
    return jnp.diag(degrees)


@dispatch
def eigenpairs(L: B.JAXNumeric, k: int):
    """
    Obtain the k highest eigenpairs of a symmetric PSD matrix L.
    """
    l, u = jnp.linalg.eigh(L)
    return l[:k], u[:, :k]


@dispatch
def set_value(a: B.JAXNumeric, index: B.Numeric, value: B.Numeric):
    """
    Set a[index] = value.
    """
    a = a.at[index].set(value)
    return a
