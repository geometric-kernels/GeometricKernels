from typing import List

import lab as B
import numpy as np
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.NPNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return np.take_along_axis(a, index, axis=axis)


@dispatch
def from_numpy(_: B.NPNumeric, b: Union[List, B.NPNumeric, B.Number]):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    return np.array(b)


@dispatch
def swapaxes(a: B.NPNumeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """
    return np.swapaxes(a, axis1, axis2)


@dispatch
def copysign(a: B.NPNumeric, b: _Numeric) -> B.Numeric:  # type: ignore
    """
    Change the sign of `a` to that of `b`, element-wise.
    """
    return np.copysign(a, b)


@dispatch
def take_along_last_axis(a: B.NPNumeric, indices: _Numeric):  # type: ignore
    """
    Takes elements of `a` along the last axis. `indices` must be the same rank (ndim) as
    `a`. Useful in e.g. argsorting and then recovering the sorted array.
    ```

    E.g. for 3d case:
    output[i, j, k] = a[i, j, indices[i, j, k]]
    """
    return np.take_along_axis(a, indices, axis=-1)
