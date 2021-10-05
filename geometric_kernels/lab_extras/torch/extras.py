from typing import Any, List, Optional

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
def trapz(y: _Numeric, x: _Numeric, dx: _Numeric = 1.0, axis: int = -1):  # type: ignore
    """
    Integrate along the given axis using the trapezoidal rule.
    """
    return torch.trapz(y, x, dim=axis)


@dispatch
def norm(x: _Numeric, ord: Optional[Any] = None, axis: Optional[int] = None):  # type: ignore
    """
    Matrix or vector norm.
    """
    return torch.linalg.norm(x, ord=ord, dim=axis)


@dispatch
def logspace(start: _Numeric, stop: _Numeric, num: int = 50, base: _Numeric = 10.0):  # type: ignore
    """
    Return numbers spaced evenly on a log scale.
    """
    return torch.logspace(start, stop, num, base)
