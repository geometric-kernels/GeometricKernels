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
