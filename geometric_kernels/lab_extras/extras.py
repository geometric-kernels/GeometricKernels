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
def from_numpy(_: B.Numeric, b: Union[List, B.Numeric, B.NPNumeric]):
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """


@dispatch
@abstract
def swapaxes(a: B.Numeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """


@dispatch
@abstract
def copysign(a: B.Numeric, b: B.Numeric) -> B.Numeric:
    """
    Change the sign of `a` to that of `b`, element-wise.
    """
