from typing import List

import lab as B
import torch
from lab import dispatch
from plum import Union

_Numeric = Union[B.Number, B.TorchNumeric]


@dispatch
def take_along_axis(a: _Numeric, index: _Numeric, axis: int = 0) -> _Numeric:  # type: ignore
    """
    Gathers elements of `a` along `axis` at `index` locations.
    """
    return torch.index_select(a, axis, B.flatten(index))


@dispatch
def from_numpy(
    _: B.TorchNumeric, b: Union[List, B.Number, B.NPNumeric, B.TorchNumeric]
):  # type: ignore
    """
    Converts the array `b` to a tensor of the same backend as `a`
    """
    print("HERE!!!")
    return torch.tensor(b)


@dispatch
def swapaxes(a: B.TorchNumeric, axis1: int, axis2: int) -> B.Numeric:
    """
    Interchange two axes of an array.
    """
    return torch.transpose(a, axis1, axis2)


@dispatch
def copysign(a: B.TorchNumeric, b: _Numeric) -> B.Numeric:  # type: ignore
    """
    Change the sign of `a` to that of `b`, element-wise.
    """
    return torch.copysign(a, b)
