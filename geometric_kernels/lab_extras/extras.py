from typing import List

import lab as B
from lab import dispatch
from lab.util import abstract
from plum import Union


@dispatch
@abstract()
def take_along_axis(a: B.Numeric, index: B.Numeric, axis: int = 0):
    """
    Gathers elements of `a` along `axis` at `index` locations.

    The index array will be flattened.

    :param a: [N1, ..., Nj, ..., Nn]
    :param index: [J...]

    :return: [N1, ..., J, ..., Nn]
    """


@dispatch
@abstract()
def take_along_last_axis(a: B.Numeric, indices: B.Numeric) -> B.Numeric:
    """
    Takes elements of `a` along the last axis. `indices` must be the same rank (ndim) as
    `a`. Useful in e.g. argsorting and then recovering the sorted array.
    ```

    E.g. for 3d case:
    output[i, j, k] = a[i, j, indices[i, j, k]]
    """


@dispatch
@abstract()
def from_numpy(_: B.Numeric, b: Union[List, B.Numeric, B.NPNumeric]):
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """


@dispatch
@abstract()
def swapaxes(a: B.Numeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """


@dispatch
@abstract()
def copysign(a: B.Numeric, b: B.Numeric) -> B.Numeric:
    """
    Change the sign of `a` to that of `b`, element-wise.
    """


def isclose(a: B.Numeric, b: B.Numeric, rtol=1e-5, atol=1e-8):
    """Returns a boolean array where two arrays are element-wise equal within a
    tolerance."""

    rtol = B.cast(B.dtype(a), rtol)
    atol = B.cast(B.dtype(a), atol)

    return B.lt(B.abs(a - b), from_numpy(a, atol) + from_numpy(a, rtol) * B.abs(b))
