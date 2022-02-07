from typing import List

import jax.numpy as jnp
import lab as B
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.JAXNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return jnp.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.JAXNumeric, b: Union[List, B.NPNumeric, B.Number, B.JAXNumeric]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return jnp.array(b)


@dispatch
def swapaxes(a: B.JAXNumeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """
    return jnp.swapaxes(a, axis1, axis2)


@dispatch
def copysign(a: B.JAXNumeric, b: _Numeric) -> B.Numeric:  # type: ignore
    """
    Change the sign of `a` to that of `b`, element-wise.
    """
    return jnp.copysign(a, b)
