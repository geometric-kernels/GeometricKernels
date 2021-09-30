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
    """


@dispatch
@abstract()
def from_numpy(_: B.Numeric, b: Union[List, B.Numeric]):
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """


@dispatch
@abstract
def trapz(y: B.Numeric, x: B.Numeric, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the trapezoidal rule.
    """
